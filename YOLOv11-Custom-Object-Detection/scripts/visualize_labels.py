import cv2
import os
import yaml
import random
import numpy as np
from scripts.logger_utils import setup_production_logging

# Initialize Production Logger
logger = setup_production_logging("visualize_labels")

def visualize_samples(num_samples=5):
    """
    Randomly selects images from the dataset and visualizes the YOLO labels
    to ensure accuracy, normalization, and alignment.
    """
    # 1. Load data.yaml
    yaml_path = os.getenv("DATASET_YAML", os.path.abspath("dataset/data.yaml"))
    if not os.path.exists(yaml_path):
        logger.error(f"Dataset configuration not found at {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # Resolve training images path
    train_path = data_config.get('train', '')
    if not train_path:
        logger.error("No 'train' path found in data.yaml")
        return
    
    # Handle both absolute and relative paths in data.yaml
    if not os.path.isabs(train_path):
        train_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), train_path))

    if not os.path.exists(train_path):
        logger.error(f"Image directory not found: {train_path}")
        logger.info("Check if data.yaml paths correctly point to your image folders.")
        return

    class_names = data_config.get('names', {})
    
    logger.info(f"Checking labels in: {train_path}")
    
    # 2. Get image list
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(train_path) if f.lower().endswith(valid_extensions)]
    
    if not images:
        logger.error(f"No images found in {train_path}")
        return

    selected_images = random.sample(images, min(len(images), num_samples))
    
    logger.info(f"Visualizing {len(selected_images)} random samples...")

    for img_name in selected_images:
        img_path = os.path.join(train_path, img_name)
        # Find corresponding label file
        label_dir = train_path.replace('images', 'labels')
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            logger.warning(f"Label file missing for {img_name}")
            continue

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Could not read image {img_name}")
            continue
            
        h, w, _ = image.shape
        
        # Read Labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        logger.info(f"Processing {img_name} ({len(lines)} objects)")

        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # --- NORMALIZATION CHECK ---
            for val in coords:
                if not (0 <= val <= 1):
                    logger.error(f"CRITICAL: Found non-normalized value {val} in {label_name}!")
            
            # Convert YOLO (center x, center y, width, height) to pixels
            cx, cy, bw, bh = coords[0], coords[1], coords[2], coords[3]
            
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # Draw Box
            color = (0, 255, 0) # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            if isinstance(class_names, list):
                label_text = class_names[cls_id] if cls_id < len(class_names) else f"ID:{cls_id}"
            else:
                label_text = class_names.get(cls_id, f"ID:{cls_id}")
                
            cv2.putText(image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show Sample
        cv2.imshow("Label Verification (Press any key for next sample)", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    logger.info("Visualization complete.")

if __name__ == "__main__":
    visualize_samples()
