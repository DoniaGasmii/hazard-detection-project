# utils/data_annotation/annotate_with_detector.py
from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2

def annotate_images_with_yolo(model_path, image_dir, output_dir, conf_threshold=0.4):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for img_path in tqdm(image_paths, desc="Annotating"):
        results = model(img_path, conf=conf_threshold)[0]
        basename = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(output_dir, f"{basename}.txt")

        with open(label_path, 'w') as f:
            for box in results.boxes:
                cls = int(box.cls[0])
                x_center, y_center, width, height = box.xywhn[0].tolist()
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
