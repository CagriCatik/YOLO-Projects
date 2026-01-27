@echo off
echo Starting YOLO Training Pipeline...
call .\venv\Scripts\activate.bat
python train_pipeline.py
pause
