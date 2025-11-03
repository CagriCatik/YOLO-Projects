
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO
from .log import log

def _read_results_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields if possible
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except Exception:
                    pass
            rows.append(r)
    return rows

def _plot_simple(x, y, xlabel, ylabel, title, out_file):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def generate_training_plots(run_dir: str, out_dir: str):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        log(f"[yellow]Warning:[/yellow] {csv_path} not found; skipping training plots.")
        return
    rows = _read_results_csv(csv_path)
    if not rows:
        log(f"[yellow]Warning:[/yellow] {csv_path} empty; skipping.")
        return
    epochs = [int(r.get("epoch", i)) for i, r in enumerate(rows)]

    def get(col):
        vals = []
        for r in rows:
            vals.append(r.get(col))
        return vals

    # mAP, precision, recall vs epoch
    for col, label in [
        ("metrics/mAP50", "mAP50"),
        ("metrics/mAP50-95", "mAP50-95"),
        ("metrics/precision", "Precision"),
        ("metrics/recall", "Recall"),
    ]:
        y = get(col)
        if all(isinstance(v, (int, float)) for v in y):
            _plot_simple(epochs, y, "Epoch", label, f"{label} vs Epoch", out_dir / f"{label.replace('/', '_')}_vs_epoch.png")

    # Loss curves
    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        y = get(col)
        if all(isinstance(v, (int, float)) for v in y):
            _plot_simple(epochs, y, "Epoch", col, f"{col} vs Epoch", out_dir / f"{col.replace('/', '_')}_vs_epoch.png")

def run_validation_and_confusion(weights: str, data_yaml: str, project: str, name: str = "val_plots"):
    model = YOLO(weights)
    log("Running validation with plots=True to generate confusion matrix and curves.")
    model.val(data=data_yaml, plots=True, project=project, name=name)
