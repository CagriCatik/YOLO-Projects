import os
import json
from ultralytics import YOLO
import torch

def train_custom_model():
    # 1. Load Configuration
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found!")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Dataset Path Configuration
    yaml_path = os.path.abspath("dataset/data.yaml")
    
    # 3. Expert Configuration
    model_variant = config.get("model_variant", "yolo11n.pt")
    
    print(f"--- Loading Expert Pipeline with {model_variant} ---")
    print(f"Checking hardware availability...")
    if torch.cuda.is_available():
        print(f"  GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  WARNING: No GPU detected. Training will be extremely slow on CPU.")

    model = YOLO(model_variant)

    # 4. Training Parameters
    training_args = config.get("training", {})
    training_args["data"] = yaml_path  # Ensure absolute path to data.yaml
    
    print(f"Configuring training for dataset: {yaml_path}")
    print(f"Hyperparameters loaded from {config_path}:")
    for key, value in training_args.items():
        if key != "data":
            print(f"  {key}: {value}")

    print("--- Starting Training ---")
    print(f"Logs and results will be saved to: {os.path.join(training_args.get('project', 'runs'), training_args.get('name', 'train'))}")
    
    results = model.train(**training_args)
    
    print("\n" + "="*50)
    print("--- Training Complete ---")
    # results.fitness is a scalar, we can just report that training is done.
    # The actual number of epochs is handled by YOLO's internal logger.
    print(f"  Best Fitness Score: {results.fitness:.4f}")
    print(f"  Best Weights Path: {results.save_dir}/weights/best.pt")
    print(f"  Last Weights Path: {results.save_dir}/weights/last.pt")
    print("="*50)

if __name__ == "__main__":
    # Ensure the dataset path is valid before starting
    if not os.path.exists("dataset/data.yaml"):
        print("Error: dataset/data.yaml not found!")
    else:
        train_custom_model()
