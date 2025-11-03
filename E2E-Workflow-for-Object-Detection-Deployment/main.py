#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO end-to-end runner driven by config.yml.
Requires a local PC with an NVIDIA GPU and CUDA-enabled PyTorch.
"""
import sys
from pathlib import Path
import yaml
import torch
from yolo_project.prepare import prepare_from_zip
from yolo_project.train import train as train_fn
from yolo_project.predict import predict as predict_fn
from yolo_project.export_artifacts import zip_run
from yolo_project.plots import generate_training_plots, run_validation_and_confusion

def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Install a CUDA-enabled PyTorch build and ensure your GPU drivers are installed."
        )
    torch.cuda.set_device(0)
    return torch.cuda.get_device_name(0)

def load_config(cfg_path: str):
    cfg_path = Path(cfg_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path: str = "config.yml"):
    cfg = load_config(cfg_path)

    # 1) Require GPU
    gpu_name = require_gpu()
    print(f"[INFO] Using GPU: {gpu_name}")

    # 2) Optional dataset preparation
    if cfg.get("data", {}).get("zip_path"):
        print("[INFO] Preparing dataset from zip...")
        data_yaml = prepare_from_zip(
            zip_path=cfg["data"]["zip_path"],
            out_dir=cfg["data"]["out_dir"],
            train_pct=float(cfg["data"].get("train_pct", 0.9)),
        )
        cfg["train"]["data_yaml"] = data_yaml

    # 3) Train
    print("[INFO] Starting training...")
    train_fn(
        data_yaml=cfg["train"]["data_yaml"],
        model=cfg["train"].get("model", "yolo11s.pt"),
        epochs=int(cfg["train"].get("epochs", 60)),
        imgsz=int(cfg["train"].get("imgsz", 640)),
        project=cfg["train"].get("project", "runs"),
    )
    run_dir = Path(cfg["train"].get("project", "runs")) / "detect" / "train"

    # 4) Predict on validation images
    if cfg.get("predict", {}).get("enabled", True):
        print("[INFO] Running prediction on validation images...")
        predict_fn(
            weights=str(run_dir / "weights" / "best.pt"),
            source=str(Path(cfg["data"]["out_dir"]) / "validation" / "images"),
            save=bool(cfg["predict"].get("save", True)),
            imgsz=int(cfg["predict"].get("imgsz", cfg["train"].get("imgsz", 640))),
            project=cfg["predict"].get("project", "runs"),
            name=cfg["predict"].get("name", "predict"),
        )

    # 5) Plots: metrics and confusion matrix
    print("[INFO] Generating plots...")
    from pathlib import Path as _P
    plots_dir = _P(cfg.get("plots", {}).get("out_dir", "artifacts/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    generate_training_plots(run_dir=str(run_dir), out_dir=str(plots_dir))
    run_validation_and_confusion(
        weights=str(run_dir / "weights" / "best.pt"),
        data_yaml=cfg["train"]["data_yaml"],
        project=str(run_dir.parent),
        name="val_plots"
    )

    # 6) Export artifacts
    if cfg.get("export", {}).get("enabled", True):
        print("[INFO] Exporting artifacts...")
        zip_path = zip_run(
            run_dir=str(run_dir),
            name=cfg["export"].get("name", "my_model"),
            out_dir=cfg["export"].get("out_dir", "artifacts"),
        )
        print(f"[INFO] Wrote artifacts to: {zip_path}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    main(cfg)
