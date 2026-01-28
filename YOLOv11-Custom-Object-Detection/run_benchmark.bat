@echo off
cd /d "%~dp0"
echo Starting Speed Benchmark (PT vs ONNX)...
.\venv\Scripts\python.exe -m scripts.benchmark_comparison
pause
