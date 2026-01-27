import torch
import time
from ultralytics import YOLO

def test_torch_computation():
    print("--- Testing PyTorch Computational Speed ---")
    size = 5000
    iterations = 5
    
    # Warm up (CUDA initialization takes time)
    if torch.cuda.is_available():
        dummy = torch.randn(100, 100).cuda()
        torch.mm(dummy, dummy)
        torch.cuda.synchronize()

    # CPU Test
    print(f"Running {iterations} iterations on CPU (Matrix {size}x{size})...")
    start_cpu = time.time()
    for _ in range(iterations):
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.mm(x_cpu, y_cpu)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"CPU total time: {cpu_time:.4f} seconds")

    # GPU Test
    if torch.cuda.is_available():
        print(f"Running {iterations} iterations on GPU (Matrix {size}x{size})...")
        # Move data to GPU once to measure pure computation speed
        x_gpu = torch.randn(size, size).cuda()
        y_gpu = torch.randn(size, size).cuda()
        
        # Start timing only the computation
        comp_start = time.time()
        for _ in range(iterations):
            z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        comp_end = time.time()
        
        gpu_comp_time = comp_end - comp_start
        print(f"GPU Pure Computation time: {gpu_comp_time:.4f} seconds")
        print(f"Speedup (Pure Computation): {cpu_time / gpu_comp_time:.2f}x")
    else:
        print("GPU not available for computation test.")

def test_yolo_gpu():
    print("\n--- Testing YOLO with GPU ---")
    try:
        # Load a tiny model to avoid large downloads
        model = YOLO("yolo11n.pt") 
        
        # Check if we can move model to device 0 (GPU)
        model.to('cuda')
        print(f"YOLO model successfully moved to: {model.device}")
        
        # Test a dummy inference (on a blank image)
        import numpy as np
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(blank_image, device=0, verbose=False)
        print("Inference on GPU successful!")
        
    except Exception as e:
        print(f"YOLO GPU Test Failed: {e}")

if __name__ == "__main__":
    test_torch_computation()
    test_yolo_gpu()
