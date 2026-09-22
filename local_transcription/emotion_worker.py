"""Private pipe worker: separate PyTorch/CUDA runtime, no audio files or network."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import base64
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'whisper_ser_deployment'))
import numpy as np
import torch
from predict import EmotionPredictor


def emit(value):
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = os.environ.get('EMOTION_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        predictor = EmotionPredictor(ROOT / 'whisper_ser_deployment', device)
        # Pay lazy CUDA/kernel and audio-library initialization costs before recording.
        predictor.predict_waveform((.01*np.sin(np.arange(32000)*2*np.pi*220/16000)).astype(np.float32))
        emit({'ready': True, 'device': device, 'message': 'Emotion model ready'})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                audio = np.frombuffer(base64.b64decode(request['audio'], validate=True), dtype='<f4').copy()
                if len(audio) > 8 * 16000:
                    raise ValueError('Emotion window exceeds eight seconds')
                started = time.perf_counter()
                result = predictor.predict_waveform(audio)
                emit({**result, 'inference_ms': round((time.perf_counter()-started)*1000)})
            except Exception as exc:
                emit({'error': str(exc)})
    except Exception as exc:
        emit({'ready': False, 'message': str(exc)})
