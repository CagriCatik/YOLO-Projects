# Traffic Sign Detection Dataset (YOLO11)

This dataset is configured for high-performance traffic sign detection using the YOLO11 framework.

## Dataset Overview
- **Total Images**: 2200+ (Source: Roboflow)
- **Format**: YOLOv11 Segment/Detect
- **Splits**:
  - `train/`: 1918 images
  - `valid/`: 188 images
  - `test/`: 188 images

## Classes (13 Total)
The dataset includes the following class labels:
`0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `object`

##  Processing & Augmentation
- **Input Size**: 640x640 (Stretched)
- **Augmentations**:
  - Auto-orientation
  - Random Gaussian Blur (0 to 2.5px)

## Training Pipeline
The model is being trained on an **NVIDIA RTX 3080** using the `train_pipeline.py` script.
- **Base Model**: `yolo11n.pt`
- **Epochs**: 100
- **Batch Size**: 16
- **Optimizer**: Auto (AdamW)

## Workspace Structure
- `dataset/`: This directory (Images & Labels)
- `docs/`: Training guides and implementation plans
- `scripts/`: GPU benchmarks and check tools
- `traffic_sign_detection/`: Training results and weights
