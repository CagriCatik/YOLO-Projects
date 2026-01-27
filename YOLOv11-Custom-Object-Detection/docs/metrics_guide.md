# 📊 Understanding Object Detection Metrics

This document explains the key evaluation criteria used in the YOLO pipeline to measure the performance of your traffic sign detection model.

---

## 1. Intersection over Union (IoU)
Before understanding the other metrics, we must define **IoU**.
IoU measures the overlap between the **Ground Truth** (the box you labeled) and the **Prediction** (the box the model drew).

*   **Calculation**: `Area of Overlap / Area of Union`
*   **Significance**: An IoU of 1.0 is a perfect match. In YOLO, a prediction is usually considered "correct" if its IoU is above **0.5 (50%)**.

---

## 2. Precision & Recall (The Fundamentals)

### **Precision (Accuracy of Positives)**
*   **Definition**: Out of all the signs the model *claimed* to find, how many were actually signs?
*   **High Precision**: Means the model has very few "False Positives" (it doesn't hallucinate signs on trees or walls).
*   **Formula**: `TP / (TP + FP)`

### **Recall (Completeness of Search)**
*   **Definition**: Out of all the signs that *actually exist* in the image, how many did the model find?
*   **High Recall**: Means the model has very few "False Negatives" (it doesn't miss signs).
*   **Formula**: `TP / (TP + FN)`

> **The Trade-off**: Usually, as you increase Recall (make the model more sensitive), Precision drops (it starts making more mistakes).

---

## 3. Mean Average Precision (mAP)
This is the most important metric in object detection. It is the average of the "Average Precision" (AP) across all classes.

### **mAP@50 (or mAP.5)**
*   This measures the precision across different recall levels when the **IoU threshold is set to 0.5**.
*   It is a "relaxed" metric. It checks if the model found the sign and roughly placed a box around it.

### **mAP@50-95 (The Gold Standard)**
*   This calculates the mAP at multiple IoU thresholds (from 0.5 to 0.95 in steps of 0.05) and then averages them.
*   It rewards the model for being **precise with its box placement**. If your sign is detected but the box is slightly shifted, this score will be lower.
*   **Target**: For traffic signs, a score above **0.6-0.7** is good; above **0.8** is excellent.

---

## 4. Loss Functions
During training, you will see three types of "Loss". Unlike mAP, **lower is better** for loss.

*   **Box Loss**: Measures how accurately the model's box matches the ground truth edges.
*   **Class Loss**: Measures how accurately the model identifies the *type* of sign (e.g., distinguishing a Stop sign from a Speed Limit sign).
*   **DFL (Distribution Focal Loss)**: Helps the model predict the boundaries of boxes more precisely, especially when edges are blurry.

---

## 5. Visual Tools

### **Confusion Matrix**
A table showing exactly which classes are being mixed up. 
*   *Example*: If the model frequently labels "Class 0" as "Class 1", you will see a high number in those intersecting cells.

### **F1 Curve**
The F1 score is the "Harmonic Mean" of Precision and Recall. The curve shows the balance between the two at different confidence thresholds. The peak of this curve tells you the **optimal confidence threshold** to use for your application.
