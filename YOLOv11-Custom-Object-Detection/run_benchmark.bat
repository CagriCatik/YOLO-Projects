@echo off
echo Starting Speed Benchmark (PT vs ONNX)...
call .\venv\Scripts\activate.bat
python scripts/benchmark_comparison.py
pause
