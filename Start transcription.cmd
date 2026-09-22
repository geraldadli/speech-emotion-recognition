@echo off
cd /d "%~dp0"
if not exist ".venv-transcription\Scripts\python.exe" goto setup
if not exist "local_transcription\models\base\model.bin" goto setup
echo Open http://127.0.0.1:8765 in Chrome or Edge.
echo Keep this window open while using the app. Press Ctrl+C to stop the server.
".venv-transcription\Scripts\python.exe" local_transcription\app.py
pause
exit /b
:setup
echo Run Setup transcription.cmd first.
pause
