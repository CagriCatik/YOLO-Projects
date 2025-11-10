"""
Webcam demo for YOLOv8 OBB bolts detector.
"""

from __future__ import annotations

import argparse

import cv2
from ultralytics import YOLO


def run_webcam(
    model_path: str,
    device_index: int = 0,
    conf: float = 0.5,
    window_name: str = "YOLOv8 OBB Webcam",
) -> None:
    model = YOLO(model_path)
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {device_index}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=conf)
            annotated = results[0].plot()

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 OBB webcam inference.")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/best.pt",
        help="Path to trained YOLOv8 OBB weights.",
    )
    parser.add_argument("--device-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--conf", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    run_webcam(
        model_path=args.model,
        device_index=args.device_index,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
