#!/usr/bin/env python3

"""
Convert COCO annotations to YOLO format.

Example Usage (bash):
    python3 utils/data_format_conversions/coco_to_yolo.py \
        --coco_path path/to/instances.json \
        --yolo_path path/to/output_labels/
"""


import os
import json
import argparse
from collections import defaultdict

def convert_coco_to_yolo(coco_path, yolo_path):
    # Load COCO JSON
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Map category_id to continuous YOLO id (0, 1, 2, ...)
    category_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Make sure output directory exists
    os.makedirs(yolo_path, exist_ok=True)

    for image_id, anns in annotations_by_image.items():
        img = images[image_id]
        img_w, img_h = img['width'], img['height']
        file_name = os.path.splitext(img['file_name'])[0] + '.txt'
        out_path = os.path.join(yolo_path, file_name)

        with open(out_path, 'w') as f:
            for ann in anns:
                bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h

                yolo_cat_id = category_id_to_yolo[ann['category_id']]
                f.write(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Conversion complete! YOLO labels saved to: {yolo_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format.")
    parser.add_argument("--coco_path", required=True, help="Path to COCO instances JSON file.")
    parser.add_argument("--yolo_path", required=True, help="Directory to save YOLO annotation txt files.")
    args = parser.parse_args()

    convert_coco_to_yolo(args.coco_path, args.yolo_path)

if __name__ == "__main__":
    main()



