"""HTTP/WebSocket server with on-PC inference; remote clients may use an HTTPS tunnel."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
# NVIDIA runtime wheels, if installed, stay isolated in this virtual environment.
_dll_handles = []
if os.name == "nt":
    for folder in (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia").glob("*/bin"):
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        _dll_handles.append(os.add_dll_directory(str(folder)))

import numpy as np
import soxr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from engine import TranscriptBuffer, decode, RATE
from emotion import EmotionService, EmotionWindow
from origins import allowed_origin

model = None
model_status = {"ready": False, "message": "Loading local model…", "device": "", "model": "base"}
capture_lock = asyncio.Lock()
emotion_service = EmotionService()


def load_model():
    global model
    path = Path(os.environ.get("ASR_MODEL_PATH", str(ROOT / "models" / "base")))
    if not (path / "model.bin").is_file():
        raise RuntimeError("Local model missing. Run Setup transcription.cmd first.")
    requested = os.environ.get("ASR_DEVICE", "auto")
    devices = ["cuda", "cpu"] if requested == "auto" else [requested]
    failures = []
    for device in devices:
        try:
            candidate = WhisperModel(str(path), device=device,
                                     compute_type="float16" if device == "cuda" else "int8",
                                     cpu_threads=8, num_workers=1, local_files_only=True)
            # Force the lazy generator to run, catching unavailable CUDA DLLs now.
            segments, _ = candidate.transcribe(np.zeros(RATE, np.float32), language="en", beam_size=1)
            list(segments)
            model = candidate
            model_status.update(ready=True, device=device, model=path.name,
                                message="Ready on " + ("GPU" if device == "cuda" else "CPU") +
                                (" (GPU runtime unavailable)" if failures else ""))
            return
        except Exception as exc:
            failures.append(f"{device}: {exc}")
    raise RuntimeError("; ".join(failures))


@asynccontextmanager
async def lifespan(app):
    async def initialize():
        try:
            await asyncio.to_thread(load_model)
        except Exception as exc:
            model_status.update(message=str(exc))
    job = asyncio.create_task(initialize())
    emotion_job = asyncio.create_task(emotion_service.start())
    yield
    await job
    await emotion_job
    await emotion_service.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/capture-worklet.js")
def worklet():
    return FileResponse(ROOT / "capture-worklet.js", media_type="application/javascript")


@app.get("/health")
def health():
    return {**model_status, 'emotion': emotion_service.status}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    # ngrok preserves Host by default. Keep browser connections same-origin.
    origin = ws.headers.get("origin")
    if not allowed_origin(origin, ws.headers.get('host'), os.environ.get('APP_ALLOWED_ORIGINS', '')):
        await ws.close(code=1008)
        return
    await ws.accept()
    if not model_status["ready"] or capture_lock.locked():
        await ws.send_json({"type": "error", "message": "Model is loading or another recording is active."})
        await ws.close()
        return
    async with capture_lock:
        buffer = TranscriptBuffer()
        emotion_window = EmotionWindow()
        finished = asyncio.Event()
        disconnected = False
        resampler = None
        await ws.send_json({"type": "ready"})

        def append_audio(audio):
            buffer.append(audio)
            emotion_window.append(audio)

        async def wait_for_audio():
            try:
                await asyncio.wait_for(finished.wait(), timeout=.25)
            except asyncio.TimeoutError:
                pass

        async def emotions():
            while not disconnected and not session['error']:
                final = finished.is_set()
                if not emotion_service.status.get('ready'):
                    if final:
                        return
                    await wait_for_audio()
                    continue
                sample = emotion_window.take(final)
                if sample is not None:
                    audio, end_seconds = sample
                    try:
                        result = await emotion_service.predict(audio)
                        if disconnected:
                            return
                        await ws.send_json({'type': 'emotion', **result,
                                            'end_seconds': end_seconds,
                                            'window_seconds': len(audio)/RATE})
                    except Exception as exc:
                        if not disconnected:
                            await ws.send_json({'type': 'emotion', 'error': str(exc) or 'Emotion inference timed out.'})
                        return
                if final:
                    return
                await wait_for_audio()

        async def receive():
            nonlocal resampler, disconnected
            try:
                start = await ws.receive_json()
                rate = int(start.get("sample_rate", 0))
                if start.get("type") != "start" or not 8000 <= rate <= 96000:
                    raise ValueError("Unsupported microphone sample rate.")
                language = start.get("language", "en")
                if language not in {"en", "id", "auto"}:
                    raise ValueError("Unsupported language selection.")
                session["language"] = None if language == "auto" else language
                resampler = soxr.ResampleStream(rate, RATE, 1, dtype="float32", quality="HQ")
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        disconnected = True
                        break
                    if msg.get("bytes") is not None:
                        raw = msg["bytes"]
                        if len(raw) > 384000 or len(raw) % 4:
                            raise ValueError("Invalid microphone packet.")
                        append_audio(resampler.resample_chunk(np.frombuffer(raw, dtype="<f4")))
                    elif msg.get("text"):
                        if json.loads(msg["text"]).get("type") == "stop":
                            append_audio(resampler.resample_chunk(np.empty(0, np.float32), last=True))
                            break
            except WebSocketDisconnect:
                disconnected = True
            except Exception as exc:
                session["error"] = str(exc)
            finally:
                finished.set()

        session = {"language": "en", "error": None}
        receiver = asyncio.create_task(receive())
        emotion_task = asyncio.create_task(emotions())
        try:
            while True:
                try:
                    await asyncio.wait_for(finished.wait(), timeout=.9)
                except asyncio.TimeoutError:
                    pass
                if disconnected:
                    break
                if session["error"]:
                    await ws.send_json({"type": "error", "message": session["error"]})
                    break
                final = finished.is_set()
                audio, revision = buffer.snapshot()
                if len(audio) and (final or (revision != buffer.processed_revision and len(audio) >= RATE)):
                    started = time.perf_counter()
                    decoded = await asyncio.to_thread(decode, model, audio, session["language"])
                    buffer.apply(decoded, len(audio), revision, (time.perf_counter() - started) * 1000, final=final)
                    await ws.send_json({"type": "transcript", **buffer.state()})
                if final:
                    try:
                        await asyncio.wait_for(emotion_task, timeout=45)
                    except asyncio.TimeoutError:
                        await ws.send_json({'type': 'emotion', 'error': 'Final emotion estimate timed out; transcript is complete.'})
                    await ws.send_json({"type": "done", **buffer.state()})
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "message": "Transcription failed: " + str(exc)})
            except Exception:
                pass
        finally:
            disconnected = True
            finished.set()
            receiver.cancel()
            try:
                await receiver
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            try:
                await emotion_task
            except (Exception, asyncio.CancelledError):
                pass
            try:
                await ws.close()
            except (RuntimeError, WebSocketDisconnect):
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, ws_max_size=384000)
