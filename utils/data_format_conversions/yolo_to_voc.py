#!/usr/bin/env python3

"""
Convert YOLO-format annotations to Pascal VOC XML format.

Example usage:
    python3 utils/data_format_conversions/yolo_to_voc.py \
        --yolo_root path/to/yolo_dataset \
        --output_dir path/to/output/voc_dataset \
        --class_list path/to/classes.txt
"""

import os
import argparse
import shutil
from xml.etree.ElementTree import Element, SubElement, ElementTree


def load_classes(class_list_path):
    with open(class_list_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def create_voc_annotation(filename, image_shape, boxes, classes, output_path):
    annotation = Element('annotation')

    SubElement(annotation, 'folder').text = 'VOC2007'
    SubElement(annotation, 'filename').text = filename
    SubElement(annotation, 'path').text = filename

    source = SubElement(annotation, 'source')
    SubElement(source, 'database').text = 'Unknown'

    size = SubElement(annotation, 'size')
    SubElement(size, 'width').text = str(image_shape[1])
    SubElement(size, 'height').text = str(image_shape[0])
    SubElement(size, 'depth').text = str(image_shape[2])

    SubElement(annotation, 'segmented').text = '0'

    for cls_id, bbox in boxes:
        obj = SubElement(annotation, 'object')
        SubElement(obj, 'name').text = classes[cls_id]
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bndbox = SubElement(obj, 'bndbox')
        x_min, y_min, x_max, y_max = bbox
        SubElement(bndbox, 'xmin').text = str(int(x_min))
        SubElement(bndbox, 'ymin').text = str(int(y_min))
        SubElement(bndbox, 'xmax').text = str(int(x_max))
        SubElement(bndbox, 'ymax').text = str(int(y_max))

    tree = ElementTree(annotation)
    tree.write(output_path)


def convert_yolo_to_voc(yolo_root, output_root, classes):
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(yolo_root, 'images', split)
        lbl_dir = os.path.join(yolo_root, 'labels', split)
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        voc_img_dir = os.path.join(output_root, 'JPEGImages')
        voc_anno_dir = os.path.join(output_root, 'Annotations')
        voc_split_dir = os.path.join(output_root, 'ImageSets', 'Main')

        os.makedirs(voc_img_dir, exist_ok=True)
        os.makedirs(voc_anno_dir, exist_ok=True)
        os.makedirs(voc_split_dir, exist_ok=True)

        image_ids = []

        for file in os.listdir(img_dir):
            if not file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(img_dir, file)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(file)[0] + '.txt')
            image_id = os.path.splitext(file)[0]
            image_ids.append(image_id)

            # Copy image
            shutil.copy2(img_path, os.path.join(voc_img_dir, f"{image_id}.jpg"))

            # Load image to get shape
            import cv2
            image = cv2.imread(img_path)
            h, w, c = image.shape

            # Convert labels
            boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())
                        x_min = (x_c - bw / 2) * w
                        y_min = (y_c - bh / 2) * h
                        x_max = (x_c + bw / 2) * w
                        y_max = (y_c + bh / 2) * h
                        boxes.append((int(cls_id), [x_min, y_min, x_max, y_max]))

            # Write XML
            xml_out = os.path.join(voc_anno_dir, f"{image_id}.xml")
            create_voc_annotation(f"{image_id}.jpg", image.shape, boxes, classes, xml_out)

        # Write split file
        split_file = os.path.join(voc_split_dir, f"{split}.txt")
        with open(split_file, 'w') as f:
            f.write('\n'.join(image_ids))

    print(f"YOLO to VOC conversion complete! Dataset saved to: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO-format dataset to Pascal VOC format.")
    parser.add_argument('--yolo_root', required=True, help='Path to YOLO-format dataset root')
    parser.add_argument('--output_dir', required=True, help='Output directory for Pascal VOC dataset')
    parser.add_argument('--class_list', required=True, help='Path to class list file')
    args = parser.parse_args()

    classes = load_classes(args.class_list)
    convert_yolo_to_voc(args.yolo_root, args.output_dir, classes)


if __name__ == "__main__":
    main()
