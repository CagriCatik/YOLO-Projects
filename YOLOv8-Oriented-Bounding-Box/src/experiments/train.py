"""
Training script for YOLOv8 OBB on the bolts dataset.
"""

from pathlib import Path
import argparse

from ultralytics import YOLO


def train(
    model_name: str,
    data_cfg: str,
    epochs: int = 100,
    imgsz: int = 640,
    project: str = "bolts_yolov8_obb",
    name: str = "exp",
    device: str | int | None = None,
) -> None:
    model = YOLO(model_name)

    model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        device=device,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 OBB on bolts dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-obb.pt",
        help="Base model weights or model name.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./configs/train_info.yaml",
        help="Ultralytics data yaml file.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="bolts_yolov8_obb")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device index (e.g. '0') or 'cpu'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        model_name=args.model,
        data_cfg=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
