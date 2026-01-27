@echo off
echo Running Model Validation...
call .\venv\Scripts\activate.bat
python scripts/validate_model.py
pause
