# Utils - Visualization Tools for YOLO Object Detection

This folder contains utility scripts to **visualize object detection results** for YOLO-format datasets.  
It includes helper functions for drawing bounding boxes and a CLI tool for quickly viewing labeled images from the terminal.

---

## 📂 Contents

### 1. `draw.py`
Provides reusable functions for working with bounding boxes and YOLO label files:

- **`get_color(cls_id)`** – Returns a consistent color for each class ID.
- **`draw_boxes(image, bboxes)`** – Draws bounding boxes (with class labels) on an image.
- **`show_before_after(original, augmented, original_boxes, augmented_boxes)`** – Side-by-side comparison of original and augmented images with labels.
- **`yolo_to_xyxy(label_path, img_width, img_height)`** – Converts YOLO normalized labels (`class x_center y_center width height`) to pixel coordinates.

---

### 2. `visualize_yolo.py` (CLI Tool)
A command-line script to load an image and its YOLO label file, draw bounding boxes, and display the result in a Matplotlib window.

#### **Usage**
From the terminal:
```bash
python visualize_yolo.py --image path/to/image.jpg --label path/to/label.txt
