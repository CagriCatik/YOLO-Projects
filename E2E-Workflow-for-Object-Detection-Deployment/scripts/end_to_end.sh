#!/usr/bin/env bash
set -euo pipefail
# End-to-end example; adjust paths.
yolo-proj prepare --zip ./data.zip --out data --train-pct 0.9
yolo-proj train --data data/data.yaml --model yolo11s.pt --epochs 60 --imgsz 640 --project runs
yolo-proj predict --weights runs/detect/train/weights/best.pt --source data/validation/images --save True
yolo-proj export --run-dir runs/detect/train --name my_model
