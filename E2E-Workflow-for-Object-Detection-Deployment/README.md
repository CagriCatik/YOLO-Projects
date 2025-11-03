# End-to-End YOLO Workflow for Object Detection and Model Deployment

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-brightgreen)](https://github.com/ultralytics/ultralytics)
[![Tests](https://img.shields.io/badge/tests-pytest-success)](#)

A streamlined YOLO (You Only Look Once) workflow for object detection.
Includes dataset preparation, training, validation, prediction, and artifact export.

---

## Overview

This archive contains the complete YOLO training and deployment pipeline.
It includes **ONNX export**, **unseen data evaluation**, **GPU verification**, and **visualization plots** for performance analysis.

The project automates dataset preparation (splitting and YAML generation), model training, inference, and artifact packaging using the Ultralytics framework.
It supports both modular CLI usage and full automation via configuration files.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
# or simply
pip install -e .
```

### Dataset Preparation

Prepare a ZIP archive with the following structure:

```bash
images/        # image files
labels/        # YOLO-format .txt labels
classes.txt    # one class per line
```

Then run:

```bash
yolo-proj prepare --zip /path/to/data.zip --out data --train-pct 0.9
```

This creates a split dataset with YAML metadata in `data/data.yaml`.

---

## Training

Train a YOLO model with Ultralytics:

```bash
yolo-proj train --data data/data.yaml --model yolo11s.pt --epochs 60 --imgsz 640 --project runs
```

You can specify any supported YOLO checkpoint (`yolo11n.pt`, `yolo11m.pt`, etc.).
All outputs, logs, and checkpoints will be stored under `runs/detect/train`.

---

## Prediction

Run inference on validation images:

```bash
yolo-proj predict --weights runs/detect/train/weights/best.pt --source data/validation/images --save True
```

Predicted images with bounding boxes will be saved in the output directory.

---

## Export Artifacts

Package the trained model and its results:

```bash
yolo-proj export --run-dir runs/detect/train --name my_model
```

This generates `artifacts/my_model.zip`, containing weights, metrics, ONNX exports, and training metadata.

---

## End-to-End Execution

Run the entire workflow from dataset to deployment:

```bash
python main.py
# or
python main.py path/to/config.yml
```

During execution, the runner will:

1. Detect available GPUs and select the optimal device [Verified].
2. Unzip and prepare the dataset if needed.
3. Train the YOLO model using Ultralytics with specified parameters.
4. Predict on unseen validation data to evaluate generalization.
5. Generate performance plots (mAP, precision, recall, loss curves, confusion matrix).
6. Export ONNX and PyTorch weights into a versioned archive under `artifacts/`.

---

## Advanced Notes

* **Mixed precision** is enabled by default on GPUs, boosting training throughput [Inference].
* For **small-object detection**, increase `imgsz` and use data augmentations such as mosaic or HSV jitter [Speculation].
* Analyze confusion matrices to detect class imbalance and label noise before adjusting confidence thresholds [Inference].
* **ONNX export** enables deployment across inference frameworks (e.g., TensorRT, OpenVINO, or ONNX Runtime).

---

## Command Reference

To train directly without config automation:

```bash
python -m yolo_project.cli train --data data/data.yaml --model yolo11s.pt --epochs 60 --imgsz 640 --project runs
```

To modify configuration for training-only runs:

```bash
python -c "import yaml; c=yaml.safe_load(open('config.yml')); c['data']['zip_path']=''; open('config.yml','w').write(yaml.safe_dump(c));"
```

---

## Directory Layout

```bash
.
├── yolo_project/        # Core module (CLI, utils, trainer)
├── main.py              # Runner for config-based execution
├── requirements.txt
├── pyproject.toml
├── data/                # Prepared dataset
├── runs/                # YOLO output (training, validation)
└── artifacts/           # Exported model archives (includes ONNX)
```

---

## References

* [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
* [COCO Dataset](https://cocodataset.org/#home)
* [YOLOv8 Paper (Arxiv)](https://arxiv.org/abs/2304.00504)
