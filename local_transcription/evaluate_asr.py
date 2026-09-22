"""Paired live ASR experiment. Uses original reference, never generated ASR as ground truth."""
import argparse,asyncio,csv,hashlib,json,re,time,urllib.request
from pathlib import Path
import numpy as np
from faster_whisper.audio import decode_audio
from websockets.asyncio.client import connect


def words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",s.lower().replace('’',"'"))


def score(reference,hypothesis):
    r,h=words(reference),words(hypothesis)
    if not r: raise ValueError('Reference has no words')
    dp=[[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0]=i
    for j in range(len(h)+1): dp[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            dp[i][j]=min(dp[i-1][j-1]+(r[i-1]!=h[j-1]),dp[i-1][j]+1,dp[i][j-1]+1)
    i,j=len(r),len(h);errors=[]
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(r[i-1]!=h[j-1]):
            if r[i-1]!=h[j-1]: errors.append({'type':'S','reference':r[i-1],'hypothesis':h[j-1]})
            i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:
            errors.append({'type':'D','reference':r[i-1],'hypothesis':''});i-=1
        else:
            errors.append({'type':'I','reference':'','hypothesis':h[j-1]});j-=1
    counts={k:sum(e['type']==k for e in errors) for k in 'SDI'}
    return {**counts,'N':len(r),'wer':dp[-1][-1]/len(r),'errors':errors[::-1]}


def add_noise(audio,snr,seed):
    if snr is None:return audio.copy(),None
    rng=np.random.default_rng(seed);noise=rng.normal(size=len(audio)).astype(np.float32)
    noise*=np.sqrt(np.mean(audio.astype(np.float64)**2))/np.sqrt(np.mean(noise.astype(np.float64)**2))*10**(-snr/20)
    measured=20*np.log10(np.linalg.norm(audio.astype(np.float64))/np.linalg.norm(noise.astype(np.float64)))
    mixed=audio+noise;peak=np.max(np.abs(mixed))
    if peak>.98:mixed*=.98/peak # Scale signal and noise together; preserve SNR, avoid clipping.
    return mixed.astype(np.float32),float(measured)


async def stream(audio):
    async with connect('ws://127.0.0.1:8765/stream',origin='http://127.0.0.1:8765') as ws:
        ready=json.loads(await asyncio.wait_for(ws.recv(),15))
        if ready.get('type')!='ready':raise RuntimeError(ready)
        await ws.send(json.dumps({'type':'start','sample_rate':16000,'language':'en'}))
        start=time.perf_counter();stop=None;first=None;changed=[];previous='';decodes=[];emotion_errors=[];emotion_updates=0
        async def send():
            nonlocal stop
            for offset in range(0,len(audio),3200):
                await asyncio.sleep(max(0,start+offset/16000-time.perf_counter()))
                await ws.send(audio[offset:offset+3200].astype('<f4').tobytes())
            await asyncio.sleep(max(0,start+len(audio)/16000-time.perf_counter()))
            stop=time.perf_counter();await ws.send(json.dumps({'type':'stop'}))
        sender=asyncio.create_task(send())
        try:
            while True:
                e=json.loads(await asyncio.wait_for(ws.recv(),75));now=time.perf_counter()
                if e['type']=='error':raise RuntimeError(e)
                if e['type']=='emotion':
                    if 'error' in e:emotion_errors.append(e['error'])
                    else:emotion_updates+=1
                if e['type']=='transcript':
                    decodes.append(e['decode_ms']);text=' '.join(filter(None,[e['confirmed'],e['partial']]))
                    if text and text!=previous:
                        first=first if first is not None else now
                        changed.append(now);previous=text
                if e['type']=='done':
                    gaps=np.diff(changed)
                    return {'hypothesis':e['confirmed'],'first_text_seconds':None if first is None else first-start,
                        'partial_before_stop':first is not None and stop is not None and first<stop,
                        'completion_after_stop_seconds':now-stop,'elapsed_seconds':now-start,
                        'changed_text_updates':len(changed),'update_gap_p95_seconds':None if not len(gaps) else float(np.percentile(gaps,95)),
                        'decode_sum_seconds':sum(decodes)/1000,'emotion_updates':emotion_updates,'emotion_errors':emotion_errors}
        finally:
            if not sender.done():sender.cancel()
            try:await sender
            except asyncio.CancelledError:pass


async def main(args):
    folder=Path(args.sample).resolve().parent
    sample=json.loads(Path(args.sample).read_text(encoding='utf-8-sig'))
    audio_path=folder/sample['audio'];audio=decode_audio(str(audio_path),sampling_rate=16000)
    if not np.isfinite(audio).all() or not np.any(audio):raise ValueError('Invalid speech fixture')
    health=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health'))
    if not health['ready'] or not health.get('emotion',{}).get('ready'):raise RuntimeError('Wait for both models to be ready')
    results=[];conditions=[('clean',None),('noise_20dB',20),('noise_10dB',10)]
    out=folder/'results.json'
    base={'sample':sample,'audio_sha256':hashlib.sha256(audio_path.read_bytes()).hexdigest(),'duration_seconds':len(audio)/16000,
          'health':health,'noise':'Seeded stationary Gaussian white noise, SNR measured over entire clip including pauses.',
          'normalization':'Lowercase; Unicode apostrophe normalized; punctuation ignored; apostrophes inside words retained; no number expansion.',
          'limitations':'One synthetic voice and script. Reference is intended TTS text, not independently human-verified. Not general ASR accuracy or real-user evaluation. Timing includes localhost transport and concurrent emotion inference; completion includes final emotion wait. No microphone/browser/ngrok path.',
          'results':results}
    for repeat in range(args.repeats):
        # Rotate order to reduce systematic condition/order confounding; models are already loaded.
        for name,snr in conditions[repeat%3:]+conditions[:repeat%3]:
            seed=20260915+repeat;wave,actual=add_noise(audio,snr,seed)
            result=await stream(wave)
            row={'condition':name,'repeat':repeat+1,'seed':seed,'snr_db':actual,**result,**score(sample['reference'],result['hypothesis'])}
            results.append(row);out.write_text(json.dumps(base,indent=2),encoding='utf-8')
            print(json.dumps({k:row[k] for k in ['condition','repeat','wer','first_text_seconds','completion_after_stop_seconds']}),flush=True)
            await asyncio.sleep(.5)
    with (folder/'results.csv').open('w',newline='',encoding='utf-8') as f:
        keys=['condition','repeat','snr_db','S','D','I','N','wer','first_text_seconds','completion_after_stop_seconds','update_gap_p95_seconds','partial_before_stop','emotion_updates','hypothesis']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    print('Saved '+str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample',default=str(Path(__file__).parent/'evaluation/sample.json'));p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.repeats<=10:p.error('repeats must be 1–10')
    asyncio.run(main(args))
