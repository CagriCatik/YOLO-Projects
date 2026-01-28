@echo off
cd /d "%~dp0"
echo Starting Live Webcam Inference...
echo Make sure your webcam is connected.
echo Press 'Q' in the video window to stop.
.\venv\Scripts\python.exe -m scripts.live_inference
pause
