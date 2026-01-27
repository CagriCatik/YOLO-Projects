@echo off
echo Exporting Best Weights to FP16 ONNX...
call .\venv\Scripts\activate.bat
python scripts/export_model.py
pause
