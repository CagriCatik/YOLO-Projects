FROM python:3.12-slim

# System deps for OpenCV and Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git unzip wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt README.md /app/
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY src /app/src
COPY scripts /app/scripts

# Default command shows help
CMD ["yolo-proj", "--help"]
