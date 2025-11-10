from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)
