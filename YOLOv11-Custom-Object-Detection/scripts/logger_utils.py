import logging
import os
import sys
import json
from datetime import datetime

# Production-Grade: Auto-add project root to sys.path to prevent ModuleNotFoundErrors
# when running scripts directly from the scripts/ directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format for ELK/DataDog ingestion."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_production_logging(name="yolo_pipeline"):
    """
    Sets up a dual-logging system:
    1. Standard console output for developers.
    2. JSON-formatted file output for production monitoring.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # 1. Console Handler (Human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (JSON-formatted for production tools)
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}_{date_str}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger
