import os
import sys
from ultralytics import YOLO

def test_onnx_inference(source_path):
    # 1. Path to the exported ONNX model
    onnx_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.onnx")

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        print("Please run export_model.py first.")
        return

    print(f"--- Loading ONNX Model: {onnx_path} ---")
    # Ultralytics supports running .onnx files directly
    model = YOLO(onnx_path, task='detect')

    # 2. Run Inference
    print(f"Running inference on: {source_path}")
    results = model.predict(
        source=source_path,
        conf=0.25,
        save=True,
        project="inference_tests",
        name="onnx_results",
        exist_ok=True
    )

    print("\n" + "="*50)
    print("--- ONNX Inference Complete ---")
    print(f"  Results saved to: {results[0].save_dir}")
    print("="*50)

if __name__ == "__main__":
    # Default to the entire validation images folder for a better overview
    test_source = "dataset/valid/images/"
    
    if len(sys.argv) > 1:
        test_source = sys.argv[1]
        
    if os.path.exists(test_source):
        print(f"--- Processing Source: {test_source} ---")
        test_onnx_inference(test_source)
    else:
        print(f"Error: Target source {test_source} not found.")
