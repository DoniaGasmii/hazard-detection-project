# 📦 Format Conversion Utilities
This folder contains utility scripts to convert between popular object detection annotation formats. These tools are essential for preparing datasets for training or evaluation across different models and frameworks.

-------------------------------------------------------------------------------------------

## 📄 YOLO to COCO Conversion Script
 
 `yolo_to_coco.py`: Converts YOLO-format annotations into a single COCO-format JSON file.

📂 **Expected Directory Structure:** Your YOLO dataset should follow this format:
```
dataset/
├── images/
│   └── train/
│       ├── image1.jpg
│       └── ...
│   └── test/
│       ├── image2.jpg
│       └── ...
│   └── valid/
│       ├── image3.jpg
│       └── ...
├── yolo_labels/
│       ├── image1.txt
│       └── ...
│   └── test/
│       ├── image2.txt
│       └── ...
│   └── valid/
│       ├── image3.txt
│       └── ...
└── classes.txt
```
Each `.txt` file contains bounding boxes in YOLO format:
`<class_id> <x_center> <y_center> <width> <height>`
All values must be normalized to [0, 1].

To test the script you can use the datasample in utils/test_image!

▶️**Usage:**
```
python utils/data_format_conversions/yolo_to_coco.py \
  --yolo_dir dataset/yolo_labels/train \
  --image_dir dataset/images/train \
  --output_json dataset/images/**coco_annotations/instances_train.json** \
  --class_list dataset/classes.txt
```
| Argument        | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `--yolo_dir`    | Path to directory containing YOLO `.txt` label files.                         |
| `--image_dir`   | Path to directory containing image files (.jpg, .png, etc).                   |
| `--output_json` | Path to save the resulting COCO `.json` file.                                 |
| `--class_list`  | Path to a text file with one class name per line (order must match YOLO IDs). |

✅ **Output:**
A COCO-compliant annotations.json file. It should be saved in dataset/images because the script doesn't copy the images we use the same folder as the yolo one to avoid unncessary complexity (efficient conversion time and storage usage).

-------------------------------------------------------------------------------------------

## 📄 COCO to YOLO Conversion Script

-------------------------------------------------------------------------------------------

## 📄 YOLO to VOC Conversion Script

-------------------------------------------------------------------------------------------

## 📄 VOC to YOLO Conversion Script

-------------------------------------------------------------------------------------------
