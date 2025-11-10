"""
Run inference for YOLOv8 OBB on one or more images.
"""

from pathlib import Path
import argparse

from ultralytics import YOLO


def infer(
    model_path: str,
    source: str,
    save_dir: str | None = None,
    conf: float = 0.25,
) -> None:
    model = YOLO(model_path)

    kwargs = {}
    if save_dir is not None:
        kwargs["project"] = save_dir
        kwargs["save"] = True
    else:
        kwargs["save"] = True

    model(source, conf=conf, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 OBB inference on images.")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/best.pt",
        help="Path to trained YOLOv8 OBB weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./data_for_test/test1.JPG",
        help="Path to image or directory of images.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./runs/infer_images",
        help="Directory for saving results (Ultralytics project path).",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    infer(
        model_path=args.model,
        source=args.source,
        save_dir=args.save_dir,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
