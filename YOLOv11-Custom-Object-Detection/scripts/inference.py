import cv2
import torch
from ultralytics import YOLO
import sys
import os

def run_inference(source_path):
    # 1. Load the best trained model
    weights_path = "runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Warning: Custom weights not found at {weights_path}. Using base weights for demo.")
        weights_path = "yolo11n.pt"

    print(f"--- Loading Model: {weights_path} ---")
    model = YOLO(weights_path)

    # 2. Run Inference
    # stream=True is efficient for memory
    results = model.predict(
        source=source_path,
        conf=0.5,       # Confidence threshold
        iou=0.45,       # NMS IOU threshold
        device=0,       # Use RTX 3080
        show=False,     # Don't pop up windows if running in headless/script
        save=True       # Save results to runs/detect/predict
    )

    print(f"--- Inference Complete ---")
    print(f"Results saved to: {results[0].save_dir}")

    # 3. Simple Display (Optional)
    # for result in results:
    #     boxes = result.boxes
    #     print(f"Detected {len(boxes)} signs.")

if __name__ == "__main__":
    # Test on an image from the valid set if no path provided
    test_img = "dataset/valid/images/Test_img-36-_jpg.rf.025f18c66e2d1a0bd3d2a07c082ed3ec.jpg"
    
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        
    if os.path.exists(test_img):
        run_inference(test_img)
    else:
        print(f"Error: Target image {test_img} not found.")
