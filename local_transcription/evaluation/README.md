# Preliminary streaming ASR experiment

This experiment replays one synthetic English utterance through the running app
at real-time pace. It compares clean audio with additive stationary Gaussian white
noise at 20 dB and 10 dB SNR (lower SNR means stronger noise). Each condition runs
twice; condition order rotates and noise seed changes between repeats. This is
six runs of one sample, not six independent speakers or utterances.

The reference in sample.json is the intended Windows System.Speech synthesis text,
not an independently transcribed human recording. Inspect/listen to the fixture
before using it as formal reference evidence. Artificial noise is a controlled
stress condition, not a recording of a real room or street.

## Reproduce

Start the app and wait until ASR and emotion are ready. Keep other recording
sessions closed. From the project root run:

```
.venv-transcription/Scripts/python.exe local_transcription/evaluate_asr.py
```

It writes results.json incrementally, then results.csv when complete. Re-running
overwrites those result files; copy them first if needed. Errors fail explicitly
rather than being recorded as successful runs. No model weights are changed.

## Definitions

WER = (substitutions + deletions + insertions) / reference word count. Both texts
are lowercased; punctuation is removed, internal apostrophes retained, curly
apostrophes standardized. Numbers are not expanded. Edit alignment uses unit-cost
Levenshtein distance, preferring diagonal then deletion when optimal paths tie.
Raw hypotheses and token-level errors are retained for inspection.

First-text time is measured from starting the replay to receiving the first
nonempty transcript. It includes fixture leading silence, transport, buffering
and inference; it is not per-word latency. Completion delay is receipt of Done
minus sending Stop, and includes the app's final emotion wait. The p95 update gap
uses intervals between changed, nonempty transcript events. Initial wait and final
Done are not included in that gap metric. Decode sums are inference-work estimates
for overlapping windows, not end-to-end real-time factor. No RTF is claimed.

The test uses a native localhost WebSocket client. It does not exercise browser
microphone capture, phone hardware, ngrok or acoustic noise suppression. Both models
run concurrently as in the app. Performance depends on GPU load and scheduling.

## Next: real speech

Create a new directory with a consented human speech WAV and a JSON manifest with
`id`, `audio` (relative filename), `source`, and a manually checked `reference`.
Run `evaluate_asr.py --sample path/to/sample.json --repeats 2`. Keep the original
reference independent of ASR output. Use several speakers and different utterances,
not just repeated runs of the same recording. Compare real background-noise
conditions separately if you want claims about the journal's actual environment.
Report failed sessions, sample counts, speaker counts and variability. User testing
and feedback are still required by the course rubric.
