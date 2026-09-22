import unittest
import numpy as np
from emotion import EmotionWindow, RATE


class EmotionWindowTests(unittest.TestCase):
    def test_short_speech_waits_but_stop_accepts_two_seconds(self):
        window = EmotionWindow()
        window.append(np.full(2*RATE, .05, np.float32))
        self.assertIsNone(window.take())
        audio, end = window.take(final=True)
        self.assertEqual(end, 2)
        self.assertEqual(len(audio), 2*RATE)
        self.assertIsNone(window.take(final=True))

    def test_bounded_latest_window_skips_obsolete_queue(self):
        window = EmotionWindow()
        window.append(np.full(12*RATE, .05, np.float32))
        audio, end = window.take()
        self.assertEqual(len(audio), 8*RATE)
        self.assertEqual(end, 12)
        self.assertIsNone(window.take())

    def test_recent_silence_does_not_reclassify_old_speech(self):
        window = EmotionWindow()
        window.append(np.full(4*RATE, .05, np.float32))
        self.assertIsNotNone(window.take())
        window.append(np.zeros(4*RATE, np.float32))
        self.assertIsNone(window.take())


if __name__ == '__main__':
    unittest.main()
