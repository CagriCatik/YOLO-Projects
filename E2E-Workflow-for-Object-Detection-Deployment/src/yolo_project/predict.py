from pathlib import Path
from ultralytics import YOLO
from .log import log

def predict(weights: str, source: str, save: bool = True, imgsz: int = 640, project: str = "runs", name: str = "predict"):
    weights = Path(weights).expanduser().resolve()
    project = Path(project).expanduser().resolve()
    model = YOLO(str(weights))
    log(f"Running prediction on {source}")
    r = model.predict(source=source, save=save, imgsz=imgsz, project=str(project), name=name, task="detect")
    log("Prediction complete")
    return r
