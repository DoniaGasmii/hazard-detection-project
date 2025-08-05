# DISAL Computer Vision Internship – Experiments & Utilities

This repository contains experiments and utility scripts developed during my internship at the **Distributed Intelligent Systems and Algorithms Laboratory (DISAL)**, focusing on **computer vision** tasks.

## 📂 Structure

- **notebooks/**
  - Jupyter notebooks for various experiments and training runs:
    - `gso_yolov8_experiment.ipynb` – YOLOv8 with GSO modifications.
    - `training_yolov8_MOCS.ipynb` – Model training on MOCS dataset.
    - `yolo_finetune_retention.ipynb` – Fine-tuning experiments with retention strategies.

- **utils/**
  - **data_augmentation/** – Image & label augmentation tools.
  - **data_format_conversions/** – Scripts for converting between formats (YOLO ↔ COCO, etc.).
  - **data_labeling/** – Utilities for annotation and label management.
  - **test_image/** – Sample images and labels for testing.
  - **visualizations/** – Tools for visualizing YOLO annotations and inference results.

- `requirements.txt` – Python dependencies.
- `.gitignore` – Git ignore rules.
- `README.md` – Project documentation (you are here).

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
