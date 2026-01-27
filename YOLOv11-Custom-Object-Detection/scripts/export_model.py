import os
from ultralytics import YOLO

def export_to_onnx():
    # 1. Path to the best trained weights
    weights_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt")

    if not os.path.exists(weights_path):
        print(f"Error: Trained weights not found at {weights_path}")
        print("Please wait for training to finish before exporting.")
        return

    print(f"--- Loading Model for Export: {weights_path} ---")
    model = YOLO(weights_path)

    # 2. Export the model
    # half=True exports to FP16 (Half Precision) - Much faster on RTX 3080
    print("--- Starting Export to ONNX (FP16 Optimized) ---")
    onnx_path = model.export(
        format='onnx', 
        simplify=True, 
        dynamic=False,
        half=True 
    )
    
    print("\n" + "="*50)
    print("--- Export Complete ---")
    print(f"  ONNX Model Saved: {onnx_path}")
    print("="*50)

if __name__ == "__main__":
    export_to_onnx()
