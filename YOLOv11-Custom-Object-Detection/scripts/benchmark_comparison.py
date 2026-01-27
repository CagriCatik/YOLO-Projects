import time
import torch
import os
from ultralytics import YOLO

def run_benchmark():
    # 1. Configuration
    pt_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt")
    onnx_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.onnx")
    test_img = os.path.abspath("dataset/valid/images/Test_img-36-_jpg.rf.02ffc1aa413ca8edd972fb53a71a576a.jpg")
    iterations = 100
    warmup = 10

    if not os.path.exists(pt_path) or not os.path.exists(onnx_path):
        print("Error: Models not found. Ensure both .pt and .onnx exist.")
        return

    print(f"--- Starting Speed Benchmark ({iterations} iterations) ---")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")

    # 2. PyTorch Benchmark
    print("\nBenchmarking PyTorch (.pt)...")
    model_pt = YOLO(pt_path)
    # Warmup
    for _ in range(warmup):
        model_pt.predict(test_img, device=0, verbose=False)
    
    start_time = time.time()
    for _ in range(iterations):
        model_pt.predict(test_img, device=0, verbose=False)
    pt_avg = (time.time() - start_time) / iterations * 1000

    # 3. ONNX Benchmark
    print("Benchmarking ONNX (.onnx)...")
    model_onnx = YOLO(onnx_path, task='detect')
    # Warmup
    for _ in range(warmup):
        model_onnx.predict(test_img, verbose=False)
    
    start_time = time.time()
    for _ in range(iterations):
        model_onnx.predict(test_img, verbose=False)
    onnx_avg = (time.time() - start_time) / iterations * 1000

    # 4. Results
    print("\n" + "="*50)
    print("      INFERENCE SPEED COMPARISON (AVG)")
    print("="*50)
    print(f"  PyTorch (.pt):  {pt_avg:.2f} ms")
    print(f"  ONNX (.onnx):   {onnx_avg:.2f} ms")
    print("-" * 50)
    
    diff = pt_avg - onnx_avg
    if diff > 0:
        print(f"  RESULT: ONNX is {diff:.2f} ms faster ({ (pt_avg/onnx_avg - 1)*100:.1f}%)")
    else:
        print(f"  RESULT: PyTorch is {-diff:.2f} ms faster ({ (onnx_avg/pt_avg - 1)*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
