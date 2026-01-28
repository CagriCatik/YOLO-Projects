import time
import torch
import os
from ultralytics import YOLO
from scripts.logger_utils import setup_production_logging

# Initialize Production Logger
logger = setup_production_logging("benchmark_comparison")

def run_benchmark():
    # 1. Configuration via Environment Variables
    pt_path = os.getenv("PT_MODEL_PATH", os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt"))
    onnx_path = os.getenv("ONNX_MODEL_PATH", os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.onnx"))
    
    # Use a default test image if none provided in env
    test_img = os.getenv("TEST_IMAGE", os.path.abspath("dataset/valid/images/Test_img-36-_jpg.rf.02ffc1aa413ca8edd972fb53a71a576a.jpg"))
    
    iterations = int(os.getenv("BENCHMARK_ITERATIONS", 100))
    warmup = int(os.getenv("BENCHMARK_WARMUP", 10))

    if not os.path.exists(pt_path) or not os.path.exists(onnx_path):
        logger.error("Models not found. Ensure both .pt and .onnx exist.")
        return

    logger.info(f"--- Starting Speed Benchmark ({iterations} iterations) ---")
    if torch.cuda.is_available():
        logger.info(f"Hardware: {torch.cuda.get_device_name(0)}")

    try:
        # 2. PyTorch Benchmark
        logger.info("Benchmarking PyTorch (.pt)...")
        model_pt = YOLO(pt_path)
        for _ in range(warmup):
            model_pt.predict(test_img, device=0, verbose=False)
        
        start_time = time.time()
        for _ in range(iterations):
            model_pt.predict(test_img, device=0, verbose=False)
        pt_avg = (time.time() - start_time) / iterations * 1000

        # 3. ONNX Benchmark
        logger.info("Benchmarking ONNX (.onnx)...")
        model_onnx = YOLO(onnx_path, task='detect')
        for _ in range(warmup):
            model_onnx.predict(test_img, verbose=False)
        
        start_time = time.time()
        for _ in range(iterations):
            model_onnx.predict(test_img, verbose=False)
        onnx_avg = (time.time() - start_time) / iterations * 1000

        # 4. Results
        logger.info("="*50)
        logger.info("      INFERENCE SPEED COMPARISON (AVG)")
        logger.info("="*50)
        logger.info(f"  PyTorch (.pt):  {pt_avg:.2f} ms")
        logger.info(f"  ONNX (.onnx):   {onnx_avg:.2f} ms")
        logger.info("-" * 50)
        
        diff = pt_avg - onnx_avg
        if diff > 0:
            logger.info(f"  RESULT: ONNX is {diff:.2f} ms faster ({ (pt_avg/onnx_avg - 1)*100:.1f}%)")
        else:
            logger.info(f"  RESULT: PyTorch is {-diff:.2f} ms faster ({ (onnx_avg/pt_avg - 1)*100:.1f}%)")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_benchmark()
