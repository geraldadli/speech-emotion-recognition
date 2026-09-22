"""Replay a speech file at real-time pace to verify partial text precedes Stop.

Start the local server first. This tests real inference and WebSocket transport,
not the browser microphone, WER, or a real-user study. Input audio is not saved.
"""
import argparse
import asyncio
import json
import time
from faster_whisper.audio import decode_audio
from websockets.asyncio.client import connect


async def check(path, require_emotion=False):
    audio = decode_audio(path, sampling_rate=16000)
    async with connect("ws://127.0.0.1:8765/stream", origin="http://127.0.0.1:8765") as ws:
        assert json.loads(await ws.recv())["type"] == "ready"
        await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "language": "en"}))
        started = time.perf_counter()
        stopped = False
        partial_before_stop = False
        first_text = None
        updates = 0
        emotion_updates = 0
        emotion_before_stop = False

        async def send():
            nonlocal stopped
            for offset in range(0, len(audio), 3200):
                await ws.send(audio[offset:offset + 3200].astype("<f4").tobytes())
                await asyncio.sleep(.2)
            stopped = True
            await ws.send(json.dumps({"type": "stop"}))

        sender = asyncio.create_task(send())
        try:
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "error":
                    raise RuntimeError(event["message"])
                if event['type'] == 'emotion':
                    if 'error' in event:
                        raise RuntimeError(event['error'])
                    emotion_updates += 1
                    emotion_before_stop |= not stopped
                    assert len(event['probabilities']) == 8
                    assert abs(sum(event['probabilities'].values())-1) < 1e-5
                if event["type"] == "transcript":
                    updates += 1
                    if event["confirmed"] or event["partial"]:
                        first_text = first_text or time.perf_counter() - started
                        partial_before_stop |= not stopped
                if event["type"] == "done":
                    assert partial_before_stop, "No nonempty transcript before Stop"
                    assert event["confirmed"], "Final transcript empty"
                    if require_emotion:
                        assert emotion_before_stop, 'No emotion estimate before Stop'
                    print(json.dumps({"passed": True, "partial_before_stop": True,
                                      "first_text_seconds": round(first_text, 2),
                                      "updates": updates, "audio_seconds": event["audio_seconds"],
                                      "emotion_updates": emotion_updates, "emotion_before_stop": emotion_before_stop,
                                      "final_text": event["confirmed"]}, indent=2))
                    return
            raise RuntimeError("Connection closed before final transcript")
        finally:
            if not sender.done():
                sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="A spoken English audio file, preferably over 10 seconds")
    parser.add_argument('--require-emotion', action='store_true')
    args = parser.parse_args()
    asyncio.run(check(args.audio, args.require_emotion))
