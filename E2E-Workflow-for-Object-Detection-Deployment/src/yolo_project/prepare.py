import os
import random
import shutil
import zipfile
from pathlib import Path
import yaml
from typing import Tuple
from .log import log

def unzip_to(src_zip: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip, "r") as zf:
        zf.extractall(dst_dir)
    # Try to find the root containing images/ and labels/
    candidate = dst_dir
    subdirs = [p for p in dst_dir.rglob("*") if p.is_dir()]
    for d in [dst_dir] + subdirs:
        if (d / "images").exists() and (d / "labels").exists():
            return d
    return dst_dir

def split_dataset(src_root: Path, dst_root: Path, train_pct: float = 0.9) -> Tuple[Path, Path]:
    assert 0.0 < train_pct < 1.0, "train_pct must be in (0,1)"
    # Create target layout
    train_images = dst_root / "train" / "images"
    train_labels = dst_root / "train" / "labels"
    val_images = dst_root / "validation" / "images"
    val_labels = dst_root / "validation" / "labels"
    for p in [train_images, train_labels, val_images, val_labels]:
        p.mkdir(parents=True, exist_ok=True)

    # Enumerate images and matching labels
    image_dir = src_root / "images"
    label_dir = src_root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Expected 'images' and 'labels' subdirs under the dataset root")

    images = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not images:
        raise FileNotFoundError("No images found under images/")

    random.shuffle(images)
    split_idx = int(len(images) * train_pct)
    train_set, val_set = images[:split_idx], images[split_idx:]

    def move_pair(img_path: Path, dest_img_dir: Path, dest_lbl_dir: Path):
        rel = img_path.relative_to(image_dir)
        lbl_rel = rel.with_suffix(".txt")
        lbl_src = label_dir / lbl_rel
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        # Copy images and labels, preserve relative structure
        dest_img_path = dest_img_dir / rel
        dest_lbl_path = dest_lbl_dir / lbl_rel
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_img_path)
        if lbl_src.exists():
            shutil.copy2(lbl_src, dest_lbl_path)
        else:
            log(f"[yellow]Warning:[/yellow] Missing label for {img_path}")

    for img in train_set:
        move_pair(img, train_images, train_labels)
    for img in val_set:
        move_pair(img, val_images, val_labels)

    return dst_root / "train", dst_root / "validation"

def ensure_classes_txt(src_root: Path) -> Path:
    classes = src_root / "classes.txt"
    if not classes.exists():
        raise FileNotFoundError("classes.txt not found in dataset root; create it with one class name per line.")
    return classes

def write_data_yaml(out_path: Path, dataset_root: Path, classes_file: Path):
    with classes_file.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    data = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "validation/images",
        "nc": len(names),
        "names": names,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def prepare_from_zip(zip_path: str, out_dir: str, train_pct: float = 0.9) -> str:
    zip_path = Path(zip_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    work_dir = out_dir
    tmp_dir = out_dir / "_unzipped"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log(f"Unzipping {zip_path} -> {tmp_dir}")
    src_root = unzip_to(zip_path, tmp_dir)
    log(f"Detected dataset root: {src_root}")

    classes_file = ensure_classes_txt(src_root)
    log(f"Found classes file: {classes_file}")

    log(f"Splitting dataset with train_pct={train_pct}")
    split_dataset(src_root, work_dir, train_pct=train_pct)

    data_yaml = work_dir / "data.yaml"
    write_data_yaml(data_yaml, work_dir, classes_file)
    log(f"Wrote {data_yaml}")
    # Cleanup temp unzip
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(data_yaml)
