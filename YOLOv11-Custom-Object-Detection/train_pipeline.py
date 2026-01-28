import os
import json
import torch
from ultralytics import YOLO
from scripts.logger_utils import setup_production_logging

# Initialize Production Logger
logger = setup_production_logging("train_pipeline")

def train_custom_model():
    # 1. Path Configuration via Environment Variables (Production Ready)
    # Default values are provided for local development
    config_path = os.getenv("CONFIG_PATH", "config.json")
    dataset_yaml = os.getenv("DATASET_YAML", os.path.abspath("dataset/data.yaml"))
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found!")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Expert Configuration
    model_variant = os.getenv("MODEL_VARIANT", config.get("model_variant", "yolo11n.pt"))
    
    logger.info(f"--- Loading Expert Pipeline with {model_variant} ---")
    logger.info(f"Checking hardware availability...")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU Detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("  No GPU detected. Training will be extremely slow on CPU.")

    try:
        model = YOLO(model_variant)
    except Exception as e:
        logger.error(f"Failed to load model {model_variant}: {str(e)}")
        return

    # 3. Training Parameters
    training_args = config.get("training", {})
    training_args["data"] = dataset_yaml  # Inject production path
    
    logger.info(f"Configuring training for dataset: {dataset_yaml}")
    logger.info(f"Hyperparameters loaded from {config_path}")

    logger.info("--- Starting Training ---")
    try:
        results = model.train(**training_args)
        
        logger.info("="*50)
        logger.info("--- Training Complete ---")
        logger.info(f"  Best Fitness Score: {results.fitness:.4f}")
        logger.info(f"  Best Weights Path: {results.save_dir}/weights/best.pt")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    if not os.path.exists(os.getenv("DATASET_YAML", "dataset/data.yaml")):
        logger.error("dataset/data.yaml not found!")
    else:
        train_custom_model()
