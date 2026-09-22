"""Bounded rolling-window transcription with explicit provisional text.

Whisper is an offline model. Re-decoding the growing window yields provisional
text while capture continues. Silence commits an utterance; long utterances
commit words ending before a conservative time boundary. This is a prototype
streaming policy, not a claim that every provisional word is stable.
"""
from dataclasses import dataclass
import time
import numpy as np

RATE = 16000


@dataclass
class Decoded:
    text: str
    # End timestamps in seconds relative to the window.
    words: list[tuple[float, str]]


class TranscriptBuffer:
    def __init__(self):
        self.audio = np.empty(0, dtype=np.float32)
        self.confirmed = ""
        self.partial = ""
        self.revision = 0
        self.processed_revision = -1
        self.total_samples = 0
        self.first_speech_at = None
        self.first_partial_ms = None
        self.decode_ms = 0.0
        self.updates = 0

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if not np.isfinite(samples).all():
            raise ValueError("Microphone sent invalid audio samples.")
        if len(self.audio) + len(samples) > 30 * RATE:
            raise ValueError("Processing could not keep up. Stop and restart with a smaller model.")
        self.audio = np.concatenate((self.audio, np.clip(samples, -1, 1)))
        self.total_samples += len(samples)
        self.revision += 1
        if self.first_speech_at is None and len(samples) and np.sqrt(np.mean(samples**2)) > .006:
            self.first_speech_at = time.perf_counter()

    def snapshot(self):
        return self.audio.copy(), self.revision

    def apply(self, decoded, count, revision, elapsed_ms, final=False):
        self.processed_revision = revision
        self.decode_ms = elapsed_ms
        self.updates += 1
        self.partial = decoded.text.strip()
        if self.partial and self.first_partial_ms is None and self.first_speech_at is not None:
            self.first_partial_ms = (time.perf_counter() - self.first_speech_at) * 1000
        tail = self.audio[max(0, count - int(.8 * RATE)):count]
        quiet_end = len(tail) >= int(.8 * RATE) and np.sqrt(np.mean(tail**2)) < .004
        if final or quiet_end:
            self._commit(self.partial)
            self.partial = ""
            self.audio = self.audio[count:]
        elif count >= 8 * RATE:
            safe = [(end, word) for end, word in decoded.words if end <= count / RATE - 2]
            if safe:
                self._commit("".join(word for _, word in safe).strip())
                consumed = min(count, max(1, round(safe[-1][0] * RATE)))
                self.audio = self.audio[consumed:]
                self.partial = "".join(word for end, word in decoded.words if end > safe[-1][0]).strip()
            elif not decoded.text.strip():
                # Silent windows have no text to preserve.
                self.audio = self.audio[count:]

    def _commit(self, text):
        if text:
            self.confirmed = (self.confirmed + " " + text).strip()

    def state(self):
        return {"confirmed": self.confirmed, "partial": self.partial,
                "audio_seconds": round(self.total_samples / RATE, 1),
                "decode_ms": round(self.decode_ms),
                "first_partial_ms": round(self.first_partial_ms) if self.first_partial_ms is not None else None,
                "updates": self.updates}


def decode(model, audio, language):
    if len(audio) < RATE // 5 or np.sqrt(np.mean(audio**2)) < .003:
        return Decoded("", [])
    segments, _ = model.transcribe(audio, language=language, beam_size=1,
                                  condition_on_previous_text=False, temperature=0,
                                  word_timestamps=True, vad_filter=True,
                                  vad_parameters={"min_silence_duration_ms": 500})
    segments = list(segments)
    return Decoded("".join(segment.text for segment in segments).strip(),
                   [(word.end, word.word) for segment in segments for word in (segment.words or [])])
