# YOLOv8 OBB Bolts

[![Task](https://img.shields.io/badge/Task-Oriented_Object_Detection-blue)](https://docs.ultralytics.com/tasks/obb/)
[![Model](https://img.shields.io/badge/Model-YOLOv8_obb-0ea5e9)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics_Bookmark-orange)](https://github.com/ultralytics/ultralytics)
[![Made with](https://img.shields.io/badge/Made%20with-Python-3776AB)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/Uses-OpenCV-5C3EE8)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-yellow)](#)

This repository provides a small Python package for training and running inference with **Ultralytics YOLOv8 Oriented Bounding Box (OBB)** models on a bolts dataset. `yolov8-obb` is a YOLOv8 variant whose detection head and loss are configured to predict **rotated bounding boxes** instead of standard axis-aligned boxes.

---

## 1. Background: What is YOLOv8 OBB?

### Axis-aligned vs oriented boxes

Standard YOLOv8 models predict axis-aligned bounding boxes:

- Bounding box parameterization: center `(x, y)`, width `w`, height `h`
- Edges are parallel to the image axes

OBB models, such as `yolov8-obb`, predict rotated boxes:

- Bounding box parameterization (internally): `(x, y, w, h, r)`  
  where `r` is the rotation angle
- In label files, Ultralytics exposes this via the YOLO-OBB format:

```text
  class_id x1 y1 x2 y2 x3 y3 x4 y4
````

All coordinates are normalized to `[0, 1]`.

OBB models are particularly useful when:

- Objects can appear at arbitrary orientations
  (e.g. aerial/satellite imagery, ships, planes, cars in overhead views, solar panels, scene text, PCB components, bolts)
- Axis-aligned boxes would include large background areas or overlap multiple nearby instances

### In the Ultralytics ecosystem

- Pretrained OBB weights include:
  `yolov8n-obb.pt`, `yolov8s-obb.pt`, `yolov8m-obb.pt`, etc.
  These are trained on DOTA-style aerial datasets but can be fine-tuned on custom OBB datasets (such as the bolts dataset).
- Training and inference APIs are the same as standard YOLOv8, with important differences:

  - The data YAML must point to an **OBB dataset** in YOLO-OBB format.
  - Some features (for example certain tracking modes) are still limited or evolving for OBB.

In this project, when you use `yolov8n-obb.pt`, you are using a YOLOv8 network configured for **rotated boxes**, not standard horizontal boxes.

---

## 2. Repository structure

```bash
YOLOV-OBB-TRAINING/
  configs/
    train_info.yaml          # Ultralytics data config (copied from original project)

  src/
    config.py                # YAML helpers

    data/
      convert_labels.py      # DOTA -> YOLO OBB label converter

    experiments/
      train.py               # Training entrypoint
      infer_image.py         # Image / folder inference
      infer_webcam.py        # Webcam demo

  datasets/
    bolts_dataset/
      images/
        train/               # training images (not included)
        val/                 # validation images (not included)
      labels/
        train_original/      # original DOTA-style labels
        val_original/
        train/               # YOLO OBB-format labels (generated)
        val/

  data_for_test/             # test images (not included)
  models/                    # trained weights, e.g. best.pt (not included)

  requirements.txt
```

Dataset, test data, and trained model files are intentionally **not** included.
The directories exist as placeholders so that you can drop in your own `datasets` and `data_for_test` content.

---

## 3. Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` assumes a recent Ultralytics version that supports YOLOv8 OBB models.

---

## 4. Dataset layout and label format

Expected dataset layout:

```text
datasets/bolts_dataset/
  images/
    train/*.png|jpg
    val/*.png|jpg
  labels/
    train_original/*.txt
    val_original/*.txt
```

Each original label file (DOTA-style) should contain one object per line:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name
```

The label converter in this repo:

- Reads these DOTA-style labels from `train_original` and `val_original`
- Produces YOLO-OBB label files under `labels/train` and `labels/val`
- Maps the bolt class to index `0` (single-class dataset)

---

## 5. Label conversion (DOTA -> YOLO OBB)

Run the converter:

```bash
python -m src.data.convert_labels --dataset-root ./datasets/bolts_dataset
```

After this command:

- YOLO-OBB formatted labels will be written into:

  - `datasets/bolts_dataset/labels/train`
  - `datasets/bolts_dataset/labels/val`
- Each output file will follow the YOLO-OBB convention:

  ```text
  class_id x1 y1 x2 y2 x3 y3 x4 y4
  ```

  with coordinates normalized to `[0, 1]`.

---

## 6. Training

Typical training command:

```bash
python -m src.experiments.train \
  --model yolov8n-obb.pt \
  --data ./configs/train_info.yaml \
  --epochs 100 \
  --imgsz 640 \
  --project bolts_yolov8_obb \
  --name exp
```

Notes:

- `--model`
  Pretrained OBB checkpoint, for example `yolov8n-obb.pt`, `yolov8s-obb.pt`, etc.
- `--data`
  Ultralytics YAML pointing to the OBB-formatted dataset.
  The provided `configs/train_info.yaml` should already reference `datasets/bolts_dataset`.
- `--project` and `--name`
  Define the logging/output structure (e.g. `bolts_yolov8_obb/exp`).
- You can adjust epochs, image size, batch size, and other Ultralytics parameters as needed.

The best checkpoint is typically saved under:

```text
./bolts_yolov8_obb/exp/weights/best.pt
```

(adjust `exp` if you use a different `--name`).

---

## 7. Image inference

Run inference on a single image or directory:

```bash
python -m src.experiments.infer_image \
  --model ./bolts_yolov8_obb/exp/weights/best.pt \
  --source ./data_for_test \
  --save-dir ./runs/infer_images \
  --conf 0.25
```

Key arguments:

- `--model`
  Path to a trained YOLOv8 OBB checkpoint (`best.pt`).
- `--source`
  Single image path or a directory containing images.
- `--save-dir`
  Output directory where annotated results are stored.
- `--conf`
  Confidence threshold for visualized detections.

---

## 8. Webcam demo

Run real-time detection from a webcam:

```bash
python -m src.experiments.infer_webcam \
  --model ./bolts_yolov8_obb/exp/weights/best.pt \
  --device-index 0 \
  --conf 0.95
```

- `--device-index` selects the camera (0 is usually the default webcam).
- `--conf` sets a high confidence threshold for visualization.

Controls:

- Press `q` or `Esc` to close the webcam window.

---

## 9. Python dependencies

Minimal `requirements.txt`:

```text
ultralytics
opencv-python
pyyaml
```

Additional dependencies can be added as needed for logging, experiment tracking, or visualization.

---

## 10. Reference

- YOLOv8 OBB release discussion: [https://github.com/orgs/ultralytics/discussions/7472](https://github.com/orgs/ultralytics/discussions/7472)
- OBB dataset overview: [https://github.com/orgs/ultralytics/discussions/5378](https://github.com/orgs/ultralytics/discussions/5378)
- See Ultralytics documentation for details: [https://docs.ultralytics.com/tasks/obb/](https://docs.ultralytics.com/tasks/obb/)
- Robot Mania video (project context): [https://www.youtube.com/watch?v=7n6gCqC075g](https://www.youtube.com/watch?v=7n6gCqC075g)
