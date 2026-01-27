# YOLO Labeling Guideline

To train a YOLO model, your data must follow a specific format. This guide ensures your dataset is "YOLO-conform."

## 1. File Structure
For EVERY image file (e.g., `img_001.jpg`), there must be a corresponding text file with the EXACT same name (e.g., `img_001.txt`) in the labels directory.

- **Image Path**: `train/images/img_001.jpg`
- **Label Path**: `train/labels/img_001.txt`

## 2. Label File Format
Each `.txt` file contains one line for each object detected in the image.
The format for each line is:
`<class_id> <x_center> <y_center> <width> <height>`

### Definitions:
- **`class_id`**: Integer index of the class (starts from 0).
- **`x_center`**: X coordinate of the bounding box center (normalized 0 to 1).
- **`y_center`**: Y coordinate of the bounding box center (normalized 0 to 1).
- **`width`**: Width of the bounding box (normalized 0 to 1).
- **`height`**: Height of the bounding box (normalized 0 to 1).

### Normalized Coordinates:
To normalize coordinates:
- `x_center = box_x_center / image_width`
- `y_center = box_y_center / image_height`
- `width = box_width / image_width`
- `height = box_height / image_height`

## 3. Best Practices for Labeling
- **Tight Boxes**: Draw bounding boxes as tightly as possible around the object.
- **Occlusions**: If an object is partially hidden, label the visible part (or the estimated full extent depending on the use case).
- **Consistency**: Ensure all instances of a class are labeled across all images.
- **Background Images**: Pictures with NO objects should have an EMPTY `.txt` file (or no file at all, depending on the implementation, but an empty `.txt` is safer).

## 4. Recommended Tools
- **CVAT (Highly Recommended)**: Professional, web-based, cloud or local. Follow our [CVAT Setup Guide](cvat_setup_guide.md) to run it locally with Docker.
- **Roboflow**: Excellent for managing datasets and exporting in correct YOLO formats.
- **Label Studio**: Highly flexible multi-purpose labeling tool.

## 5. Verification
Before training, always visualize some labels to ensure they are correct:
- Are the boxes in the right place?
- Are the class IDs correct?
- Is the normalization working? (Values should be between 0 and 1).
