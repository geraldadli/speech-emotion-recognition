# Preliminary ASR evaluation results

One 23.6-second synthetic English utterance, 48 reference words, two runs per condition. Both ASR and emotion inference active on RTX 4060 Laptop GPU.

| Condition | WER per run | Pooled WER | First text mean (range), s | Stop-to-Done mean, s |
|---|---|---:|---|---:|
| clean | 0.00%, 0.00% | 0.00% | 1.58 (1.01-2.14) | 0.43 |
| noise_20dB | 0.00%, 0.00% | 0.00% | 0.98 (0.97-0.99) | 0.48 |
| noise_10dB | 6.25%, 4.17% | 5.21% | 1.00 (0.99-1.02) | 0.39 |

All six runs returned partial text before Stop and produced emotion estimates without reported emotion errors.

## Observed errors

- noise_10dB, repeat 1: I: (none) -> other; I: (none) -> the; I: (none) -> vibration
- noise_10dB, repeat 2: I: (none) -> the; I: (none) -> vibration

## Interpretation and limits

Stronger synthetic noise increased errors in this fixture. This does not establish human-speech accuracy or performance in real background noise. Zero WER here means a match to this short intended TTS reference, not perfect recognition in general. The reference audio has not been independently human-transcribed.

Timing measures first nonempty output, not first correct output. The first clean run was slower; order, warm-up and runtime variation are confounded with condition. Two repeats cannot establish a noise effect on responsiveness. Completion includes final emotion inference. Browser microphone capture and ngrok are excluded.

There are 96 reference tokens per condition across repeats, but only one unique 48-word utterance. No confidence interval or significance is claimed. See results.json for hypotheses and alignments, results.csv for metrics, environment.json for environment/code hashes, and README.md for reproduction and human-speech follow-up.
