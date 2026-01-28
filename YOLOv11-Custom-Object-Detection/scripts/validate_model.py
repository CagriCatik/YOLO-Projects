import os
from ultralytics import YOLO
from scripts.logger_utils import setup_production_logging

# Initialize Production Logger
logger = setup_production_logging("validate_model")

def validate_custom_model():
    # 1. Path Configuration via Environment Variables
    weights_path = os.getenv("MODEL_PATH", os.path.abspath("runs/detect/traffic_sign_detection/yolo11_custom/weights/best.pt"))
    yaml_path = os.getenv("DATASET_YAML", os.path.abspath("dataset/data.yaml"))

    if not os.path.exists(weights_path):
        logger.error(f"Trained weights not found at {weights_path}")
        return

    logger.info(f"--- Loading Model for Validation: {weights_path} ---")
    if not os.path.exists(yaml_path):
        logger.error(f"YAML config not found at {yaml_path}")
        return
        
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # 2. Run Validation
    logger.info(f"Running validation on '{yaml_path}'")
    try:
        metrics = model.val(
            data=yaml_path,
            split='val',
            batch=int(os.getenv("BATCH_SIZE", 16)),
            imgsz=int(os.getenv("IMG_SIZE", 640)),
            device=os.getenv("DEVICE", "0"),
            plots=True
        )

        logger.info("="*50)
        logger.info("--- Validation Metrics Summary ---")
        logger.info(f"  Precision (P):     {metrics.results_dict['metrics/precision(B)']:.4f}")
        logger.info(f"  Recall (R):        {metrics.results_dict['metrics/recall(B)']:.4f}")
        logger.info(f"  mAP @50:           {metrics.results_dict['metrics/mAP50(B)']:.4f}")
        logger.info(f"  mAP @50-95:        {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
        logger.info("-" * 50)
        logger.info(f"  Inference Speed:   {metrics.speed['inference']:.2f}ms/image")
        logger.info(f"  Results saved to:  {metrics.save_dir}")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    validate_custom_model()
