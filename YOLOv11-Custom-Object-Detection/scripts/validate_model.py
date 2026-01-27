import os
from ultralytics import YOLO

def validate_custom_model():
    # 1. Path to the best weights
    # Note: If you changed the project/name in train_pipeline.py, update this path.
    # By default, Ultralytics saves to runs/detect/project_name/version
    weights_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt")
    yaml_path = os.path.abspath("dataset/data.yaml")

    if not os.path.exists(weights_path):
        print(f"Error: Trained weights not found at {weights_path}")
        print("Please wait for the training to complete.")
        return

    print(f"--- Loading Model for Validation: {weights_path} ---")
    if not os.path.exists(yaml_path):
        print(f"  Error: YAML config not found at {yaml_path}")
        return
        
    model = YOLO(weights_path)

    # 2. Run Validation
    print(f"Running validation on '{yaml_path}' using device: 0 (RTX 3080)")
    metrics = model.val(
        data=yaml_path,
        split='val',
        batch=16,
        imgsz=640,
        device=0,      # Use RTX 3080
        plots=True
    )

    print("\n" + "="*50)
    print("--- Validation Metrics Summary ---")
    print(f"  Precision (P):     {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"  Recall (R):        {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"  mAP @50:           {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  mAP @50-95:        {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print("-" * 50)
    print(f"  Speed Score:       {metrics.speed['inference']:.2f}ms inference per image")
    print(f"  Validation Folder: {metrics.save_dir}")
    print("="*50)
    print("\nPRO TIP: Check the 'confusion_matrix.png' in the folder above for detailed error analysis.")

if __name__ == "__main__":
    validate_custom_model()
