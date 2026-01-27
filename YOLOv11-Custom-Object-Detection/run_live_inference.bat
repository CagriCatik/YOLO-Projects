@echo off
echo Starting Live Webcam Inference...
echo Make sure your webcam is connected.
echo Press 'Q' in the video window to stop.
call .\venv\Scripts\activate.bat
python scripts/live_inference.py
pause
