import unittest
import numpy as np
from evaluate_asr import score,add_noise
class EvaluationTests(unittest.TestCase):
 def test_alignment(self):
  s=score('one two three','one four three extra');self.assertEqual((s['S'],s['D'],s['I'],s['N']),(1,0,1,3))
  self.assertEqual(score('one two','')['D'],2)
  self.assertEqual(score("Hello, WORLD!","hello world")['wer'],0)
 def test_noise(self):
  x=np.sin(np.arange(16000)/10).astype(np.float32)*.8
  a,snr=add_noise(x,10,42);self.assertAlmostEqual(snr,10,places=4);self.assertLessEqual(abs(a).max(),.981)
  np.testing.assert_array_equal(a,add_noise(x,10,42)[0])
