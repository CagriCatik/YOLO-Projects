@echo off
cd /d "%~dp0"
echo Running Model Validation...
.\venv\Scripts\python.exe -m scripts.validate_model
pause
