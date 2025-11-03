import typer
from typing import Optional
from .prepare import prepare_from_zip
from .train import train as train_fn
from .predict import predict as predict_fn
from .export_artifacts import zip_run

app = typer.Typer(add_completion=False, help="CLI for preparing data, training YOLO, predicting, and exporting artifacts.")

@app.command()
def prepare(
    zip: str = typer.Option(..., "--zip", help="Path to data.zip that contains images/, labels/, classes.txt"),
    out: str = typer.Option("data", "--out", help="Output dataset directory"),
    train_pct: float = typer.Option(0.9, "--train-pct", help="Fraction of images for train split"),
):
    data_yaml = prepare_from_zip(zip, out, train_pct)
    typer.echo(data_yaml)

@app.command()
def train(
    data: str = typer.Option(..., "--data", help="Path to data.yaml"),
    model: str = typer.Option("yolo11s.pt", "--model", help="Ultralytics checkpoint to start from"),
    epochs: int = typer.Option(60, "--epochs", help="Number of epochs"),
    imgsz: int = typer.Option(640, "--imgsz", help="Training resolution"),
    project: str = typer.Option("runs", "--project", help="Ultralytics runs dir"),
):
    train_fn(data_yaml=data, model=model, epochs=epochs, imgsz=imgsz, project=project)

@app.command()
def predict(
    weights: str = typer.Option(..., "--weights", help="Path to best.pt (or other .pt)"),
    source: str = typer.Option(..., "--source", help="File, dir, URL, or camera stream"),
    save: bool = typer.Option(True, "--save", help="Save annotated outputs"),
    imgsz: int = typer.Option(640, "--imgsz", help="Prediction resolution"),
    project: str = typer.Option("runs", "--project", help="Ultralytics runs dir"),
    name: str = typer.Option("predict", "--name", help="Subdir name under runs"),
):
    predict_fn(weights=weights, source=source, save=save, imgsz=imgsz, project=project, name=name)

@app.command()
def export(
    run_dir: str = typer.Option(..., "--run-dir", help="Ultralytics run dir, e.g., runs/detect/train"),
    name: str = typer.Option("my_model", "--name", help="Artifact base name"),
    out_dir: str = typer.Option("artifacts", "--out-dir", help="Output directory for zip"),
):
    zip_run(run_dir=run_dir, name=name, out_dir=out_dir)

if __name__ == "__main__":
    app()
