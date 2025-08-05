import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from draw import *
from typing import List

def plot_confidence_histogram(pred_labels_dir: str):
    """
    Plot a histogram of detection confidence scores from YOLO prediction .txt files.
    """
    confidences = []
    for label_file in glob.glob(os.path.join(pred_labels_dir, "*.txt")):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    confidences.append(float(parts[5]))

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Detection Confidence Histogram")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def visualize_gt_vs_predictions(image_path: str, gt_labels_path: str, pred_labels_path: str, class_names: List[str]):
    """
    Side-by-side ground truth vs prediction visual comparison.
    """
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Load boxes
    gt_boxes = yolo_to_xyxy(gt_labels_path, w, h)
    pred_boxes = []
    with open(pred_labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                cls, x_c, y_c, bw, bh, _ = map(float, parts[:6])
            else:
                cls, x_c, y_c, bw, bh = map(float, parts[:5])
            x_min = (x_c - bw / 2) * w
            y_min = (y_c - bh / 2) * h
            x_max = (x_c + bw / 2) * w
            y_max = (y_c + bh / 2) * h
            pred_boxes.append([x_min, y_min, x_max, y_max, int(cls)])

    # Draw
    gt_img = draw_boxes(img, gt_boxes)
    pred_img = draw_boxes(img, pred_boxes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(gt_img)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred_img)
    axes[1].set_title("Predictions")
    axes[1].axis("off")
    plt.show()


def visualize_predictions_grid(images_dir: str, labels_dir: str, class_names: List[str], max_images: int = 9):
    """
    Display multiple inference samples in a grid using Paired palette.
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))[:max_images]
    cols = int(np.ceil(np.sqrt(len(image_files))))
    rows = int(np.ceil(len(image_files) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for idx, img_path in enumerate(image_files):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        label_path = os.path.join(labels_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            boxes = yolo_to_xyxy(label_path, w, h)
            img = draw_boxes(img, boxes)
        axes[idx].imshow(img)
        axes[idx].axis("off")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def filter_and_plot_by_confidence(labels_dir: str, min_conf: float = 0.5, target_class: str = None, class_names: List[str] = None):
    """
    Filter detections by confidence and/or class, then plot histogram.
    """
    filtered_confs = []
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls_id = int(parts[0])
                    conf = float(parts[5])
                    if conf >= min_conf and (target_class is None or class_names[cls_id] == target_class):
                        filtered_confs.append(conf)

    plt.figure(figsize=(8, 5))
    plt.hist(filtered_confs, bins=15, color="orange", edgecolor="black")
    plt.title(f"Filtered Detections (min_conf={min_conf}, class={target_class or 'All'})")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
