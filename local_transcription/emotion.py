"""One bounded recent window; no pending queue of obsolete emotion predictions."""
import asyncio
import base64
import json
import os
from pathlib import Path
import numpy as np

RATE = 16000


class EmotionWindow:
    def __init__(self):
        self.audio = np.empty(0, np.float32)
        self.total = 0
        self.processed = 0

    def append(self, audio):
        self.audio = np.concatenate((self.audio, audio))[-8*RATE:]
        self.total += len(audio)

    def take(self, final=False):
        new = self.total - self.processed
        if len(self.audio) < 2*RATE or new < (RATE//2 if final else 4*RATE):
            return None
        self.processed = self.total
        # Gate on recent activity so a pause does not repeatedly classify old speech.
        recent = self.audio[-min(new, 2*RATE):]
        if np.sqrt(np.mean(recent.astype(np.float64)**2)) < .004:
            return None
        return self.audio.copy(), self.total / RATE


class EmotionService:
    def __init__(self):
        self.process = None
        self.lock = asyncio.Lock()
        self.status = {'ready': False, 'message': 'Loading emotion model…'}

    async def start(self):
        root = Path(__file__).resolve().parent
        python = Path(os.environ.get('EMOTION_PYTHON', str(root.parent / '.venv-emotion' / 'Scripts' / 'python.exe')))
        try:
            if not python.is_file():
                raise RuntimeError('Run Setup emotion.cmd to enable emotion detection.')
            env = os.environ.copy()
            # Do not inherit the ASR process's injected CUDA 12 DLL directories.
            env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep)
                                         if 'site-packages' + os.sep + 'nvidia' not in p.lower())
            self.process = await asyncio.create_subprocess_exec(str(python), '-u', str(root/'emotion_worker.py'),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, env=env)
            self.status = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 180))
            if not self.status.get('ready'):
                await self.close()
        except Exception as exc:
            self.status = {'ready': False, 'message': 'Emotion unavailable: ' + str(exc)}
            await self.close()

    async def predict(self, audio):
        async with self.lock:
            try:
                payload = json.dumps({'audio': base64.b64encode(audio.astype('<f4').tobytes()).decode()})
                self.process.stdin.write((payload+'\n').encode())
                await self.process.stdin.drain()
                return json.loads(await asyncio.wait_for(self.process.stdout.readline(), 45))
            except (Exception, asyncio.CancelledError):
                # A timed-out/cancelled pipe response must never become the next request's result.
                self.status = {'ready': False, 'message': 'Emotion worker stopped. Restart the app to retry.'}
                await self.close()
                raise

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            await self.process.wait()
