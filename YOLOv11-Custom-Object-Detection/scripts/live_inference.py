import cv2
import os
from ultralytics import YOLO
import torch

def start_live_inference():
    # 1. Configuration
    model_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.onnx")
    
    # Fallback to .pt if .onnx doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    print(f"--- Loading Model: {model_path} ---")
    # Using task='detect' for ONNX stability
    model = YOLO(model_path, task='detect')

    # 2. Initialize Camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set internal buffer to small value for real-time
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n" + "="*50)
    print("LIVE INFERENCE STARTED")
    print("  - Press 'Q' to quit")
    print("  - Results are displayed in real-time")
    print("="*50)

    while True:
        start_tick = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 3. Run Inference
        results = model.predict(source=frame, conf=0.5, verbose=False, stream=False)
        result = results[0]

        # 4. Detailed Logging
        if len(result.boxes) > 0:
            print(f"\n--- Frame Detections ---")
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = result.names[class_id]
                conf = float(box.conf[0])
                print(f"  Detected: {label: <10} | Confidence: {conf:.2f}")

        # 5. Calculate FPS
        end_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_tick - start_tick)

        # 6. Visualize Results
        annotated_frame = result.plot()
        
        # Add FPS text to the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLO11 Live Detection", annotated_frame)

        # 7. Exit Logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Live inference stopped.")

if __name__ == "__main__":
    start_live_inference()
