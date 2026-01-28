@echo off
cd /d "%~dp0"
echo Starting YOLO Training Pipeline...
.\venv\Scripts\python.exe train_pipeline.py
pause
