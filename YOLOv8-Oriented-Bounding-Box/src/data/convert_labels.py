"""
DOTA-style -> YOLO OBB converter for the bolts dataset.

Assumes structure:

    datasets/bolts_dataset/
      images/
        train/
        val/
      labels/
        train_original/
        val_original/
        train/
        val/

The input label format is expected to be:

    x1 y1 x2 y2 x3 y3 x4 y4 class_name

Coordinates are in absolute pixels, class_name is a string like 'bolt'.
"""

from pathlib import Path
from typing import Dict, Iterable
import argparse

import cv2
from ultralytics.utils import TQDM


CLASS_MAPPING: Dict[str, int] = {
    "bolt": 0,
}


def normalize_obb(coords: list[float], w: int, h: int) -> list[float]:
    """Normalize 8 coordinates [x1, y1, ..., x4, y4] by image width/height."""
    if len(coords) != 8:
        raise ValueError(f"Expected 8 coordinates, got {len(coords)}")

    norm = []
    for i, c in enumerate(coords):
        if i % 2 == 0:
            norm.append(c / w)
        else:
            norm.append(c / h)
    return norm


def iter_images(image_dir: Path, exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".JPG")):
    """Yield image paths under image_dir with given extensions."""
    for p in sorted(image_dir.iterdir()):
        if p.suffix in exts:
            yield p


def convert_single_image(
    image_path: Path,
    orig_label_dir: Path,
    save_dir: Path,
    class_mapping: Dict[str, int],
) -> None:
    """Convert one image's annotation file."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    image_name = image_path.stem

    orig_label_path = orig_label_dir / f"{image_name}.txt"
    if not orig_label_path.is_file():
        # No annotation, silently skip
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{image_name}.txt"

    with orig_label_path.open("r") as f_in, save_path.open("w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 9:
                # malformed line
                continue

            *coord_parts, class_name = parts[:9]
            coords = [float(p) for p in coord_parts[:8]]

            if class_name not in class_mapping:
                # unknown class, skip
                continue

            class_idx = class_mapping[class_name]
            normalized = normalize_obb(coords, w, h)
            formatted_coords = [f"{c:.6g}" for c in normalized]
            f_out.write(f"{class_idx} {' '.join(formatted_coords)}\n")


def convert_dataset(dataset_root: str | Path) -> None:
    """Convert train and val splits under dataset_root."""
    root = Path(dataset_root)

    for split in ("train", "val"):
        image_dir = root / "images" / split
        orig_label_dir = root / "labels" / f"{split}_original"
        save_dir = root / "labels" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")
        if not orig_label_dir.is_dir():
            raise FileNotFoundError(f"Original label dir not found: {orig_label_dir}")

        image_paths = list(iter_images(image_dir))
        for img_path in TQDM(image_paths, desc=f"Converting {split}"):
            convert_single_image(img_path, orig_label_dir, save_dir, CLASS_MAPPING)


def main():
    parser = argparse.ArgumentParser(description="Convert DOTA annotations to YOLO OBB format.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./datasets/bolts_dataset",
        help="Root of bolts dataset (contains images/ and labels/).",
    )
    args = parser.parse_args()
    convert_dataset(args.dataset_root)


if __name__ == "__main__":
    main()
