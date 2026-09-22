"""Streaming-state regression checks; these do not claim recognition accuracy."""
import unittest
import numpy as np
import soxr
from engine import TranscriptBuffer, Decoded, RATE


class BufferTests(unittest.TestCase):
    def test_partial_is_emitted_without_stop(self):
        b = TranscriptBuffer()
        b.append(np.ones(2 * RATE, np.float32) * .1)
        data, revision = b.snapshot()
        b.apply(Decoded("Good morning", [(1, " Good"), (1.5, " morning")]), len(data), revision, 120)
        self.assertEqual(b.partial, "Good morning")
        self.assertEqual(b.confirmed, "")

    def test_final_preserves_audio_arriving_during_decode(self):
        b = TranscriptBuffer()
        b.append(np.ones(RATE, np.float32) * .1)
        data, revision = b.snapshot()
        b.append(np.ones(RATE, np.float32) * .1)
        b.apply(Decoded("first", [(1, " first")]), len(data), revision, 10, final=True)
        self.assertEqual(len(b.audio), RATE)
        self.assertEqual(b.confirmed, "first")

    def test_long_window_commits_prefix_and_keeps_tail(self):
        b = TranscriptBuffer()
        b.append(np.ones(8 * RATE, np.float32) * .1)
        data, revision = b.snapshot()
        b.apply(Decoded("one two three", [(2, " one"), (5, " two"), (7, " three")]),
                len(data), revision, 10)
        self.assertEqual(b.confirmed, "one two")
        self.assertEqual(b.partial, "three")
        self.assertEqual(len(b.audio), 3 * RATE)
        b.apply(Decoded("three", [(2, " three")]), len(b.audio), revision, 10, final=True)
        self.assertEqual(b.confirmed, "one two three")

    def test_invalid_audio_and_backlog_fail_explicitly(self):
        with self.assertRaises(ValueError):
            TranscriptBuffer().append(np.array([np.nan], np.float32))
        with self.assertRaises(ValueError):
            TranscriptBuffer().append(np.zeros(31 * RATE, np.float32))

    def test_stream_resampling_preserves_length(self):
        r = soxr.ResampleStream(48000, RATE, 1, dtype="float32")
        pieces = [r.resample_chunk(np.zeros(2048, np.float32)) for _ in range(24)]
        pieces.append(r.resample_chunk(np.empty(0, np.float32), last=True))
        self.assertLessEqual(abs(sum(map(len, pieces)) - 24 * 2048 / 3), 1)


if __name__ == "__main__":
    unittest.main()
