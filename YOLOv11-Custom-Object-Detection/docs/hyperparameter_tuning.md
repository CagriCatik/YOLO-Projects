# Hyperparameter Tuning Guide for YOLOv11

Hyperparameters are the "knobs" and "dials" you adjust before training begins to control how the model learns. Unlike **parameters** (the internal weights learned by the model during training), **hyperparameters** are set by the user and remain constant (or follow a predefined schedule) during the training process.

---

## 1. Responsibilities of Hyperparameters

Each hyperparameter has a specific role in steering the model towards convergence and high accuracy:

| Hyperparameter | Responsibility | Impact if too high | Impact if too low |
| :--- | :--- | :--- | :--- |
| **Learning Rate (`lr0`)** | Speed of weight updates. | Model diverges (explodes). | extremely slow or no learning. |
| **Momentum** | Smooths updates by using previous gradients. | Overshoots local minima. | Training becomes jittery/noisy. |
| **Weight Decay** | Prevents overfitting by penalizing large weights. | Model underfits (too simple). | Model overfits to training data. |
| **Box Loss Gain** | Importance of bounding box accuracy. | Model focuses only on boxes, ignores classes. | Poor localization. |
| **Cls Loss Gain** | Importance of classification accuracy. | Ignores box accuracy. | High localization, wrong labels. |
| **Mosaic** | Combines 4 images into one to help find small objects. | Too much noise/clutter. | Poor performance on small objects. |

---

## 2. Key YOLOv11 Hyperparameters

### Optimizer Settings
- **`lr0`**: Initial learning rate (e.g., `0.01`).
- **`lrf`**: Final learning rate as a fraction of `lr0` (e.g., `0.01` means final LR is `lr0 * 0.01`).
- **`momentum`**: Typically set to `0.937` for SGD or `0.9` for Adam.
- **`weight_decay`**: Helps with L2 regularization (prevents weights from becoming too large).

### Loss Balancing
- **`box`**: Weight for the bounding box loss.
- **`cls`**: Weight for the classification loss.
- **`dfl`**: Weight for the Distribution Focal Loss (helps refine box boundaries).

### Data Augmentation (The "Secrets" to Generalization)
- **`hsv_h`, `hsv_s`, `hsv_v`**: Adjusts hue, saturation, and value randomly.
- **`degrees`**: Random rotations.
- **`translate`**: Randomly shifting the image.
- **`scale`**: Randomly zooming in/out.
- **`flipud` / `fliplr`**: Vertical and horizontal flips.
- **`mosaic`**: Probability of using mosaic augmentation (extremely powerful for YOLO).

---

## 3. How to Tune Hyperparameters

### A. Manual Tuning (Intuition-Based)
If you notice specific issues in your `results.png` or `confusion_matrix.png`:
- **Overfitting?** Increase `weight_decay` or increase augmentation (e.g., higher `scale` or `mosaic`).
- **Poor Small Object Detection?** Increase `mosaic` or use a higher `imgsz`.
- **Loss not decreasing?** Adjust `lr0` or change the `optimizer` (e.g., from `SGD` to `AdamW`).

### B. Automated Tuning (Evolution)
YOLOv11 supports **Hyperparameter Evolution**. This uses a Genetic Algorithm to run hundreds of small "mutations" on hyperparameters to find the optimal combination.

To run evolution (Warning: This takes a long time):
```python
# Example of running evolution for 300 generations
model.tune(data='dataset/data.yaml', epochs=30, iterations=300, optimizer='AdamW')
```
*The results will be saved in `runs/detect/tune/best_hyperparameters.yaml`.*

---

## 4. Best Practices for This Project

For traffic sign detection, we care deeply about:
1. **Small Objects**: Traffic signs are often small. High `mosaic` and `scale` are crucial.
2. **Color Integrity**: Color defines many signs (Red for Stop, Blue for Info). Avoid extreme `hsv_h` (hue) shifts that could turn a red sign green.
3. **Real-world Variation**: Use `blur`, `noise`, and `weather` augmentations if the signs will be detected in varying conditions.

---

## 5. Summary Table of Defaults
YOLOv11 comes with robust defaults in `default.yaml`. Unless you have a very specific dataset, sticking close to these is usually best:

- **Optimizer**: `auto` (usually picks SGD or AdamW)
- **Warmup Epochs**: `3.0` (helps stabilize early training)
- **Close Mosaic**: `10` (turns off mosaic for the last 10 epochs to refine accuracy)
