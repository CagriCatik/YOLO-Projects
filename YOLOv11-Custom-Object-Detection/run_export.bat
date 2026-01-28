@echo off
cd /d "%~dp0"
echo Exporting Best Weights to FP16 ONNX...
.\venv\Scripts\python.exe -m scripts.export_model
pause
