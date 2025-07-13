#!/usr/bin/env python3

"""
Convert YOLO annotations to COCO format.

Example Usage (bash):
    python3 utils/data_format_conversions/yolo_to_coco.py \
        --yolo_dir path/to/yolo_labels/ \
        --image_dir path/to/images/ \
        --output_json path/to/output/instances.json \
        --class_list path/to/classes.txt
"""

import os
import json
import argparse

def load_classes(class_list_path):
    with open(class_list_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def convert_yolo_to_coco(yolo_dir, image_dir, output_json, class_list):
    categories = [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_list)]
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for label_file in sorted(os.listdir(yolo_dir)):
        if not label_file.endswith(".txt"):
            continue

        # Assume matching image name
        image_filename = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Skipping {image_filename} (not found in {image_dir})")
            continue

        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size

        images.append({
            "id": img_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        label_path = os.path.join(yolo_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x, y, w, h = map(float, parts)
                class_id = int(class_id)

                # Convert YOLO to COCO format
                x_min = (x - w / 2) * width
                y_min = (y - h / 2) * height
                box_w = w * width
                box_h = h * height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO annotations saved to: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO labels to COCO format.")
    parser.add_argument("--yolo_dir", required=True, help="Directory with YOLO .txt files")
    parser.add_argument("--image_dir", required=True, help="Directory with matching images")
    parser.add_argument("--output_json", required=True, help="Path to output instances.json")
    parser.add_argument("--class_list", required=True, help="Path to class names (one per line)")
    args = parser.parse_args()

    class_list = load_classes(args.class_list)
    convert_yolo_to_coco(args.yolo_dir, args.image_dir, args.output_json, class_list)

if __name__ == "__main__":
    main()
