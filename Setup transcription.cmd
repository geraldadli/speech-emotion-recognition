@echo off
cd /d "%~dp0"
if not exist ".venv-transcription\Scripts\python.exe" py -3.12 -m venv .venv-transcription
if errorlevel 1 goto failed
".venv-transcription\Scripts\python.exe" -m pip install -r local_transcription\requirements.txt
if errorlevel 1 goto failed
".venv-transcription\Scripts\python.exe" local_transcription\prepare_model.py
if errorlevel 1 goto failed
echo Setup complete. Double-click Start transcription.cmd.
pause
exit /b 0
:failed
echo Setup failed. Read the message above before retrying.
pause
exit /b 1
