import os
from ultralytics import YOLO
from scripts.logger_utils import setup_production_logging

# Initialize Production Logger
logger = setup_production_logging("export_model")

def export_to_onnx():
    # 1. Path Configuration via Environment Variables
    weights_path = os.getenv("MODEL_PATH", os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt"))

    if not os.path.exists(weights_path):
        logger.error(f"Trained weights not found at {weights_path}")
        return

    logger.info(f"--- Loading Model for Export: {weights_path} ---")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # 2. Export the model
    # half=True exports to FP16 (Half Precision) - Much faster on RTX 3080
    logger.info("--- Starting Export to ONNX (FP16 Optimized) ---")
    try:
        onnx_path = model.export(
            format='onnx', 
            simplify=True, 
            dynamic=False,
            half=True 
        )
        
        logger.info("="*50)
        logger.info("--- Export Complete ---")
        logger.info(f"  ONNX Model Saved: {onnx_path}")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    export_to_onnx()
