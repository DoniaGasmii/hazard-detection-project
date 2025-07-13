#!/usr/bin/env python3

"""
Convert Pascal VOC XML annotations to YOLO format with proper directory structure.

Example Usage:
    python3 utils/data_format_conversions/voc_to_yolo.py \
        --voc_root path/to/VOC2007 \
        --output_dir path/to/output/yolo_dataset \
        --class_list path/to/classes.txt
"""

import os
import argparse
import shutil
import xml.etree.ElementTree as ET

def load_classes(class_list_path):
    with open(class_list_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh
    return x_center, y_center, width, height

def convert_annotation(xml_path, out_txt_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(out_txt_path, "w") as f:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            xml_box = obj.find("bndbox")
            bbox = (
                int(xml_box.find("xmin").text),
                int(xml_box.find("ymin").text),
                int(xml_box.find("xmax").text),
                int(xml_box.find("ymax").text)
            )
            yolo_box = convert_bbox((w, h), bbox)
            f.write(f"{cls_id} " + " ".join(f"{a:.6f}" for a in yolo_box) + "\n")

def convert_all(voc_root, output_root, classes):
    image_sets_dir = os.path.join(voc_root, "ImageSets", "Main")
    annotations_dir = os.path.join(voc_root, "Annotations")
    images_dir = os.path.join(voc_root, "JPEGImages")

    for split in ["train", "val", "test"]:
        split_file = os.path.join(image_sets_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            continue

        # Create split folders
        image_out_dir = os.path.join(output_root, "images", split)
        label_out_dir = os.path.join(output_root, "labels", split)
        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        with open(split_file) as f:
            ids = [line.strip() for line in f.readlines()]

        for image_id in ids:
            img_src = os.path.join(images_dir, f"{image_id}.jpg")
            img_dst = os.path.join(image_out_dir, f"{image_id}.jpg")
            xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
            txt_path = os.path.join(label_out_dir, f"{image_id}.txt")

            if not os.path.exists(img_src):
                print(f"Image not found: {img_src}")
                continue

            shutil.copy2(img_src, img_dst)
            convert_annotation(xml_path, txt_path, classes)

    print(f"VOC to YOLO conversion complete! Dataset saved to: {output_root}")

def main():
    parser = argparse.ArgumentParser(description="Convert VOC annotations to YOLO format with correct folder structure.")
    parser.add_argument("--voc_root", required=True, help="Path to VOC2007 root folder")
    parser.add_argument("--output_dir", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--class_list", required=True, help="Path to class list (one class per line)")
    args = parser.parse_args()

    classes = load_classes(args.class_list)
    convert_all(args.voc_root, args.output_dir, classes)

if __name__ == "__main__":
    main()
