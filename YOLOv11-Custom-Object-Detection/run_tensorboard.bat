@echo off
echo Starting TensorBoard...
echo Visit http://localhost:6006 to view results.
call .\venv\Scripts\activate.bat
python -m tensorboard.main --logdir runs/
pause
