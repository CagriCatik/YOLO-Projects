import cv2
import os
import torch
from ultralytics import YOLO
from scripts.logger_utils import setup_production_logging

# Initialize Production Logger
logger = setup_production_logging("live_inference")

def start_live_inference():
    # 1. Configuration via Environment Variables
    model_path = os.getenv("MODEL_PATH", os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.onnx"))
    camera_id = int(os.getenv("CAMERA_ID", 0))
    min_confidence = float(os.getenv("MIN_CONFIDENCE", 0.25))  # Lowered for better distant detection
    img_size = int(os.getenv("IMG_SIZE", 640))
    
    # Fallback to .pt if .onnx doesn't exist
    if not os.path.exists(model_path):
        pt_fallback = model_path.replace(".onnx", ".pt")
        if os.path.exists(pt_fallback):
            model_path = pt_fallback
        else:
            # Try hardcoded local fallback if the ENV path completely failed
            model_path = os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt")

    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        return

    logger.info(f"--- Loading Model: {model_path} ---")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        logger.error(f"Failed to load engine: {str(e)}")
        return

    # 2. Initialize Camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error(f"Could not open webcam with ID: {camera_id}")
        return

    # Set internal buffer to small value for real-time
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info("LIVE INFERENCE STARTED")
    logger.info(f"  - Camera ID: {camera_id}")
    logger.info(f"  - Confidence Threshold: {min_confidence}")

    while True:
        start_tick = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Retrying...")
            continue

        # 3. Run Inference
        results = model.predict(
            source=frame, 
            conf=min_confidence, 
            imgsz=img_size,
            verbose=False, 
            stream=False
        )
        result = results[0]

        # 4. Detailed Logging (JSON-ready)
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = result.names[class_id]
                conf = float(box.conf[0])
                # Lower level log for high-frequency detections
                logger.debug(f"Detected: {label} ({conf:.2f})")

        # 5. Calculate FPS
        end_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_tick - start_tick)

        # 6. Visualize Results
        annotated_frame = result.plot()
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO11 Live Detection", annotated_frame)

        # 7. Exit Logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User requested exit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Live inference stopped.")

if __name__ == "__main__":
    start_live_inference()
