
"""Shared Whisper Large V3 preprocessing, encoder, emotion head, and offline inference."""
from pathlib import Path
import os, json, hashlib, subprocess
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import numpy as np
import soundfile as sf
import librosa
import torch
from torch import nn


def load_audio(path, config):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    sr_target = config['sample_rate']
    if path.suffix.lower() in {'.wav', '.flac', '.aif', '.aiff'}:
        info = sf.info(str(path))
        if not config['min_seconds'] <= info.duration <= config['max_seconds']:
            raise ValueError('Unsupported duration')
        wave, sr = sf.read(str(path), always_2d=True, dtype='float32')
        if not len(wave) or not np.isfinite(wave).all(): raise ValueError('Nonfinite or empty audio')
        clipping = float(np.mean(np.abs(wave) >= .999))
        mono = wave.mean(axis=1)
        if wave.shape[1] > 1:
            energies = np.mean(wave.astype(np.float64)**2, axis=0)
            if float(np.mean(mono.astype(np.float64)**2)) < .01 * energies.max():
                mono = wave[:, int(energies.argmax())]
        if sr != sr_target:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=sr_target, res_type='soxr_hq')
    else:
        import imageio_ffmpeg
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
            '-i', str(path), '-t', str(config['max_seconds'] + 1), '-f', 'f32le',
            '-acodec', 'pcm_f32le', '-ac', '1', '-ar', str(sr_target), 'pipe:1'],
            capture_output=True, timeout=90)
        if result.returncode: raise ValueError('Audio decoding failed')
        mono = np.frombuffer(result.stdout, dtype='<f4').copy()
        clipping = float(np.mean(np.abs(mono) >= .999)) if len(mono) else 0.
    if not len(mono) or not np.isfinite(mono).all(): raise ValueError('Nonfinite or empty audio')
    mono = np.asarray(mono - mono.mean(), dtype=np.float32)
    duration = len(mono) / sr_target
    rms = float(np.sqrt(np.mean(mono.astype(np.float64)**2)))
    if not config['min_seconds'] <= duration <= config['max_seconds']: raise ValueError('Unsupported duration')
    if rms < config['min_rms']: raise ValueError('Silent or too quiet')
    digest = hashlib.sha256(mono.astype('<f4').tobytes()).hexdigest()
    _, boundaries = librosa.effects.trim(mono, top_db=config['trim_db'], frame_length=1024, hop_length=160)
    margin = round(config['trim_margin_seconds'] * sr_target)
    start, end = max(0, int(boundaries[0])-margin), min(len(mono), int(boundaries[1])+margin)
    clean = np.ascontiguousarray(mono[start:end], dtype=np.float32)
    if len(clean) < config['min_seconds'] * sr_target: raise ValueError('Insufficient usable audio')
    return clean, {'audio_hash': digest, 'duration': duration, 'retained_seconds': len(clean)/sr_target,
                   'rms': rms, 'clipping_fraction': clipping}


def audio_chunks(audio, config):
    size = int(config['chunk_seconds'] * config['sample_rate'])
    if not 0 < size <= 480000: raise ValueError('Whisper window must be <=30 seconds')
    pieces = [audio[i:i+size] for i in range(0, len(audio), size)]
    if len(pieces) > 1 and len(pieces[-1]) < 8000:
        tail = np.r_[pieces[-2], pieces[-1]]
        mid = len(tail)//2
        pieces[-2:] = [tail[:mid], tail[mid:]]
    assert sum(map(len, pieces)) == len(audio) and max(map(len, pieces)) <= size
    return pieces


class WhisperFeatures:
    def __init__(self, encoder_dir, config, device='cpu'):
        from transformers import WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        self.config, self.device = config, device
        dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
        self.frontend = WhisperFeatureExtractor.from_pretrained(encoder_dir, local_files_only=True)
        self.encoder, info = WhisperEncoder.from_pretrained(encoder_dir, local_files_only=True,
            torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
            attn_implementation='sdpa', output_loading_info=True)
        if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise RuntimeError('Encoder key mismatch: ' + str(info))
        c = self.encoder.config
        if (c.encoder_layers, c.d_model, c.num_mel_bins) != (32, 1280, 128):
            raise ValueError('Expected the full Whisper Large V3 encoder')
        if (self.frontend.sampling_rate, self.frontend.feature_size, self.frontend.n_samples) != (16000,128,480000):
            raise ValueError('Incorrect Whisper frontend')
        self.encoder.requires_grad_(False).eval().to(device)
        self.effective_batch_size = config['encoder_batch_size']

    def extract_many(self, waveforms):
        windows = [(i,piece) for i,wave in enumerate(waveforms) for piece in audio_chunks(wave,self.config)]
        collected = [[] for _ in waveforms]
        position, batch_size = 0, self.effective_batch_size
        dtype = next(self.encoder.parameters()).dtype
        while position < len(windows):
            batch = windows[position:position+batch_size]
            # Separate frontend calls guarantee sample-local mel scaling regardless of batching.
            inputs = [self.frontend(piece, sampling_rate=16000, return_tensors='pt',
                padding='max_length', max_length=480000, truncation=True,
                return_attention_mask=True, do_normalize=False) for _,piece in batch]
            lengths = [min(1500,(int(x.attention_mask.sum())+1)//2) for x in inputs]
            mel = torch.cat([x.input_features for x in inputs],dim=0)
            if tuple(mel.shape[1:]) != (128,3000): raise RuntimeError('Incorrect Whisper mel shape')
            hidden = None
            try:
                with torch.inference_mode():
                    hidden = self.encoder(mel.to(self.device,dtype=dtype),return_dict=True).last_hidden_state
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1: raise
                del hidden, mel, inputs
                torch.cuda.empty_cache()
                batch_size = max(1,batch_size//2)
                self.effective_batch_size = batch_size
                continue
            for j, ((owner,_), length) in enumerate(zip(batch,lengths)):
                if length < 4: raise ValueError('Too few real frames')
                collected[owner].append(hidden[j,:length].to(device='cpu',dtype=torch.float32))
            position += len(batch)
            del hidden, mel, inputs
        rows = []
        for parts in collected:
            frames = torch.cat(parts,dim=0)
            mean = frames.mean(0)
            std = frames.std(0,unbiased=False)
            quarters = torch.stack([part.mean(0) for part in torch.tensor_split(frames,4)])
            vector = torch.cat([mean,std,(quarters-mean[None]).flatten()]).numpy()
            if vector.shape != (7680,) or not np.isfinite(vector).all(): raise ValueError('Invalid Whisper features')
            rows.append(vector)
        return np.stack(rows)


class EmotionHead(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=256, dropout=.4):
        super().__init__()
        if hidden_dim == 0:
            self.net = nn.Sequential(nn.Dropout(dropout),nn.Linear(input_dim,num_labels))
        else:
            self.net = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),
                nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,128),nn.GELU(),
                nn.Dropout(dropout/2),nn.Linear(128,num_labels))
    def forward(self,x): return self.net(x)


def normalize_features(X, mean, scale, clip):
    return np.clip((np.asarray(X,dtype=np.float32)-mean)/scale,-clip,clip).astype(np.float32)


def probabilities(logits, temperature):
    values = np.asarray(logits,dtype=np.float64)/float(temperature)
    values -= values.max(axis=1,keepdims=True)
    exp = np.exp(values)
    return exp/exp.sum(axis=1,keepdims=True)


class EmotionPredictor:
    def __init__(self, bundle, device='cpu'):
        from safetensors.torch import load_file
        root = Path(bundle)
        self.manifest = json.loads((root/'manifest.json').read_text())
        self.config, self.labels = self.manifest['audio_config'],self.manifest['labels']
        self.device = device
        self.features = WhisperFeatures(root/'encoder',self.config,device)
        with np.load(root/'normalization.npz',allow_pickle=False) as z:
            self.mean,self.scale = z['mean'],z['scale']
        self.head = EmotionHead(**self.manifest['head'])
        self.head.load_state_dict(load_file(str(root/'classifier.safetensors')),strict=True)
        self.head.eval().to(device)
    def predict(self, audio_path):
        wave, stats = load_audio(audio_path,self.config)
        x = self.features.extract_many([wave])
        x = normalize_features(x,self.mean,self.scale,self.config['normalization_clip'])
        with torch.inference_mode(): logits = self.head(torch.from_numpy(x).to(self.device)).float().cpu().numpy()
        p = probabilities(logits,self.manifest['temperature'])[0]
        best = int(p.argmax())
        return {'emotion':self.labels[best], 'confidence':float(p[best]),
                'probabilities':dict(zip(self.labels,map(float,p))), 'duration_seconds':stats['duration']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--audio',required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(json.dumps(EmotionPredictor(args.bundle,args.device).predict(args.audio)))
