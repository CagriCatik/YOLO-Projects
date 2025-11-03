from pathlib import Path
from ultralytics import YOLO
from .log import log

def train(data_yaml: str, model: str = "yolo11s.pt", epochs: int = 60, imgsz: int = 640, project: str = "runs"):
    data_yaml = Path(data_yaml).expanduser().resolve()
    project = Path(project).expanduser().resolve()

    log(f"Loading model: {model}")
    model_obj = YOLO(model)
    log(f"Starting training for {epochs} epochs at {imgsz}px")
    results = model_obj.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project),
        task="detect",
    )
    log("Training complete")
    return results
