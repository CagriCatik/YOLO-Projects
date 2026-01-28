@echo off
setlocal
:menu
cls
echo ======================================================
echo       YOLO11 Traffic Sign Detection Master Menu
echo ======================================================
echo 0. Verify Dataset Labels    (run_label_check.bat)
echo 1. Start Training pipeline      (run_train.bat)
echo 2. Run Validation               (run_validate.bat)
echo 3. Export to ONNX (FP16)        (run_export.bat)
echo 4. Run Speed Benchmark          (run_benchmark.bat)
echo 5. Start Live Webcam Inference  (run_live_inference.bat)
echo 6. Start TensorBoard Dashboard  (run_tensorboard.bat)
echo 7. Exit
echo ======================================================
set /p choice="Enter your choice (0-7): "

if "%choice%"=="0" (
    call run_label_check.bat
    goto menu
)

if "%choice%"=="1" (
    call run_train.bat
    goto menu
)
if "%choice%"=="2" (
    call run_validate.bat
    goto menu
)
if "%choice%"=="3" (
    call run_export.bat
    goto menu
)
if "%choice%"=="4" (
    call run_benchmark.bat
    goto menu
)
if "%choice%"=="5" (
    call run_live_inference.bat
    goto menu
)
if "%choice%"=="6" (
    call run_tensorboard.bat
    goto menu
)
if "%choice%"=="7" (
    exit
)

echo Invalid choice, try again.
pause
goto menu
