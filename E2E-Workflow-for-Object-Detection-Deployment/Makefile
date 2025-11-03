.PHONY: venv install train predict export clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -e . -r requirements.txt

train:
	yolo-proj train --data data/data.yaml --model yolo11s.pt --epochs 60 --imgsz 640 --project runs

predict:
	yolo-proj predict --weights runs/detect/train/weights/best.pt --source data/validation/images --save True

export:
	yolo-proj export --run-dir runs/detect/train --name my_model

clean:
	rm -rf runs artifacts
