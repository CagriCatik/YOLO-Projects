# YOLO Training Guide

This guide explains how to use the provided pipeline to train your custom object detection model on your RTX 3080.

## 1. Initial Setup
Ensure your virtual environment is active and your GPU is detected:
```powershell
.\venv\Scripts\Activate.ps1
python scripts/check_gpu.py
```
*You should see "CUDA available: True"*

## 2. Configuration (`train_pipeline.py`)
The pipeline is already pre-configured for your dataset. Key parameters to know:
- **`epochs`**: Set to `100`. If the model stops improving earlier, YOLO's "Early Stopping" will automatically finish.
- **`imgsz`**: Set to `640`. Larger values (e.g., 1280) catch smaller objects but require much more VRAM.
- **`batch`**: Set to `16`. Your RTX 3080 (10GB/12GB) can handle this easily.

## 3. Starting the Training
Run the master pipeline script:
```powershell
.\venv\Scripts\python.exe train_pipeline.py
```

### 📈 What happens during training?
1. **Validation**: The script checks your `dataset/data.yaml` to locate images.
2. **Download**: If `yolo11n.pt` is missing, it will download the pre-trained base.
3. **Epoch Logs**: You will see a progress bar for every epoch with:
   - **Box Loss**: How well the model finds the objects.
   - **Class Loss**: How well the model identifies what the object is.
   - **mAP50**: Mean Average Precision (The higher/closer to 1.0, the better).

## 4. Monitoring Progress
YOLO automatically creates a folder named `traffic_sign_detection/yolo11_custom/`.

### Visual Logs
Inside that folder, you can view:
- `results.png`: Graphs showing the loss curves over time.
- `train_batch0.jpg`: Visual proof of how the model sees the data during training.
- `confusion_matrix.png`: To see which classes are being mixed up.

### Real-time Monitoring (TensorBoard)
If you want interactive graphs, run this in a second terminal:
```powershell
.\venv\Scripts\tensorboard --logdir traffic_sign_detection/
```
Then open `http://localhost:6006` in your browser.

## 5. Output Weights
When training finishes, your weights are saved in:
`traffic_sign_detection/yolo11_custom/weights/`

- **`best.pt`**: Use this for production/inference. It has the highest accuracy.
- **`last.pt`**: Use this if you want to resume training later.

## 6. How to Resume
If training is interrupted (e.g., power outage), change your `train_pipeline.py` model line to:
```python
model = YOLO("traffic_sign_detection/yolo11_custom/weights/last.pt")
model.train(resume=True)
```
