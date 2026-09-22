# Personal voice reflection journal

Speak in English, review the live transcript, reflect on the suggested voice tone,
choose how you actually felt, and optionally save an entry. This is a personal
reflection aid, not a mental-health assessment. Call-review and rehearsal features
have been removed from the interface.

## Run

Run `Setup transcription.cmd` and `Setup emotion.cmd` once, then
`Start transcription.cmd`. Open http://127.0.0.1:8765. For phone testing, run
`ngrok http 8765` and open its HTTPS URL. Keep the host PC running. Only one
recording can run at a time; the microphone audio is processed on the PC.

## Journal workflow

Start speaking, then Finish. Review and correct transcription mistakes, select
your feeling (including mixed feelings or prefer not to say), and optionally
write a reflection. The app never selects your feeling from the model output.
The displayed tone describes only the most recent processed window, not a
whole-entry mood. The model was trained on acted English speech and is unvalidated
for spontaneous personal reflections.

Save on this device writes to localStorage only on request. Entries appear below
and can be reopened, downloaded or deleted. Saving the same entry again updates
it instead of duplicating it. Downloads preserve original ASR text separately from
corrected words, the user-selected feeling, note, date and model estimates.
Interrupted entries are marked incomplete. No audio is saved.

Storage is unencrypted and accessible to people with access to that browser
profile. There are no accounts or cloud sync. Browser clearing/private mode may
remove entries, and a changed ngrok domain is a different storage origin.
Download entries for a portable copy. A storage failure does not claim success;
download remains available. Save or download before a new entry or page refresh.

The intended users are people who prefer speaking to typing personal reflections.
Validate usefulness with actual volunteers; use non-sensitive sample reflections
for demonstrations and avoid collecting private entries unnecessarily. The course
still requires technical ASR evaluation and real user testing, not generated reviews.

## Runtime and verification

ASR uses `.venv-transcription` and pinned local Base weights. The emotion worker
uses `.venv-emotion` and `whisper_ser_deployment`. Setup reuses system Python
packages where available, pins PyTorch 2.4.1 from the CUDA 11.8 wheel index and
Transformers 4.49.0. The ASR CUDA 12 runtime stays in its separate environment.
Use `ASR_DEVICE`, `ASR_MODEL_PATH`, `EMOTION_DEVICE` or `EMOTION_PYTHON` to override
runtime choices. An emotion worker failure leaves ASR available; restart to retry.

Run `node local_transcription/test_journal.cjs` and
`.venv-transcription/Scripts/python.exe -m unittest discover -s local_transcription -p "test_*.py"`.
A running server can be tested with `local_transcription/check_stream.py` using
spoken English audio and `--require-emotion`. Synthetic replay previously verified
both streams before Stop; it is not a real-user study or WER evaluation.

The rubric still requires ASR WER, a baseline plus an experimental condition,
responsiveness measurements, error analysis, at least five relevant real users,
feedback analysis, and the eight-section report with AI usage log/declaration.
See PROJECT_CRITERIA.md. This is a usable prototype, not a validated production system.
