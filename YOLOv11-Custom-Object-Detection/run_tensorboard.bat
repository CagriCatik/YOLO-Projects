@echo off
cd /d "%~dp0"
echo Starting TensorBoard...
echo Visit http://localhost:6006 to view results.
.\venv\Scripts\python.exe -m tensorboard.main --logdir runs/
pause
