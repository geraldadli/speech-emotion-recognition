@echo off
cd /d "%~dp0"
echo Preparing the separate emotion runtime. Existing Python 3.12 packages may be reused.
if not exist ".venv-emotion\Scripts\python.exe" py -3.12 -m venv --system-site-packages .venv-emotion
if errorlevel 1 goto failed
".venv-emotion\Scripts\python.exe" -m pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto failed
".venv-emotion\Scripts\python.exe" -m pip install -r local_transcription\requirements-emotion.txt
if errorlevel 1 goto failed
echo Ready. Restart Start transcription.cmd to load both models.
pause
exit /b
:failed
echo Setup failed. Check the error above. Python 3.12 is required.
pause
exit /b 1
