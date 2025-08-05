import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from draw import *
from typing import List
import pandas as pd
import seaborn as sns
import json
from pathlib import Path

def compare_inference_results(*result_paths, output_dir=None):
    """
    Compare object detection benchmark results for multiple models.

    Parameters:
    - result_paths: paths to result files (CSV or JSON). Each file should contain
      metrics such as 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1', 'inference_time'.
    - output_dir: directory to save plots (optional).
    """
    sns.set_theme(style="whitegrid", palette="Paired")

    # Load all results
    all_results = []
    for path in result_paths:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported file format: {path}")

        # Add model name from file name
        df["model"] = path.stem
        all_results.append(df)

    # Merge into one DataFrame
    results_df = pd.concat(all_results, ignore_index=True)

    # Ensure numeric columns
    numeric_cols = ["mAP50", "mAP50-95", "precision", "recall", "f1", "inference_time"]
    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    print("\n=== Benchmark Comparison Table ===\n")
    print(results_df[["model"] + numeric_cols])

    # --- Visualization 1: Bar charts for each metric ---
    for metric in numeric_cols:
        if metric in results_df.columns:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=results_df, x="model", y=metric)
            plt.title(f"Model Comparison - {metric}")
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.tight_layout()
            if output_dir:
                plt.savefig(Path(output_dir) / f"{metric}_comparison.png", dpi=300)
            plt.show()

    # --- Visualization 2: Precision vs Recall Scatter ---
    if {"precision", "recall"}.issubset(results_df.columns):
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=results_df,
            x="precision",
            y="recall",
            hue="model",
            size="mAP50" if "mAP50" in results_df.columns else None,
            sizes=(50, 300),
            palette="Paired",
            legend=True
        )
        plt.title("Precision vs Recall")
        plt.tight_layout()
        if output_dir:
            plt.savefig(Path(output_dir) / "precision_vs_recall.png", dpi=300)
        plt.show()

    # --- Visualization 3: Radar chart for all metrics ---
    radar_metrics = [m for m in numeric_cols if m in results_df.columns]
    if radar_metrics:
        from math import pi

        # Normalize data for radar
        df_norm = results_df.copy()
        for m in radar_metrics:
            max_val = df_norm[m].max()
            if max_val > 0:
                df_norm[m] = df_norm[m] / max_val

        labels = radar_metrics
        num_vars = len(labels)

        for _, row in df_norm.iterrows():
            values = row[labels].tolist()
            values += values[:1]  # close the loop
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]

            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], labels)
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["model"])
            ax.fill(angles, values, alpha=0.25)
            plt.title(f"Radar Chart - {row['model']}")
            plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
            if output_dir:
                plt.savefig(Path(output_dir) / f"{row['model']}_radar.png", dpi=300)
            plt.show()

    return results_df




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
