@echo off
cd /d "%~dp0"
echo Verifying Dataset Labels...
echo ======================================================
echo INSTRUCTIONS:
echo 1. Check if boxes align with objects.
echo 2. Check if labels match the signs.
echo 3. Press ANY KEY in the image window to see the next sample.
echo ======================================================
.\venv\Scripts\python.exe -m scripts.visualize_labels
pause
