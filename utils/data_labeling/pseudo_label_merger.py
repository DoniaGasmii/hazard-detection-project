import os
import json
from roboflow import Roboflow

# ==========================
# USER CONFIGURATION
# ==========================

API_KEY = "YOUR_ROBOFLOW_API_KEY"

# Final class mapping for YOLO (no "tools")
CLASS_MAP = {
    "worker": 0,
    "helmet": 1,
    "harness": 2,
    "rope": 3,
    "scaffolding": 4,
    "barricade": 5,
    "guardrail": 6,
    "edge": 7,
    "ladder": 8
}

# Format: { "model_name": {"project": "PROJECT_NAME", "version": VERSION_NUMBER, "class_name": "worker"} }
MODELS = {
    "worker_model": {"project": "worker-project", "version": 1, "class_name": "worker"},
    "helmet_model": {"project": "helmet-project", "version": 1, "class_name": "helmet"},
    "harness_model": {"project": "harness-project", "version": 1, "class_name": "harness"},
    "rope_model": {"project": "rope-project", "version": 1, "class_name": "rope"},
    "scaffolding_model": {"project": "scaffolding-project", "version": 1, "class_name": "scaffolding"},
    "barricade_model": {"project": "barricade-project", "version": 1, "class_name": "barricade"},
    "guardrail_model": {"project": "guardrail-project", "version": 1, "class_name": "guardrail"},
    "edge_model": {"project": "edge-project", "version": 1, "class_name": "edge"},
    "ladder_model": {"project": "ladder-project", "version": 1, "class_name": "ladder"}
}


IMAGES_DIR = "images_input"  # Folder containing your unlabeled images
OUTPUT_DIR = "dataset"
CONF_THRESHOLD = 0.3  # Global confidence threshold

# ==========================
# SCRIPT LOGIC
# ==========================

def xyxy_to_yolo(x, y, w, h, img_w, img_h):
    """Convert pixel bbox to YOLO normalized format."""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h

def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "all"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "all"), exist_ok=True)

def main():
    rf = Roboflow(api_key=API_KEY)
    ensure_dirs()

    # Copy images to dataset/images/all
    for img_file in os.listdir(IMAGES_DIR):
        os.system(f'cp "{os.path.join(IMAGES_DIR, img_file)}" "{os.path.join(OUTPUT_DIR, "images", "all", img_file)}"')

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_file in image_files:
        all_detections = []
        img_path = os.path.join(IMAGES_DIR, img_file)

        # Get image dimensions
        import cv2
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        for model_name, info in MODELS.items():
            project = rf.workspace().project(info["project"])
            model = project.version(info["version"]).model

            preds = model.predict(img_path, confidence=CONF_THRESHOLD).json()

            for det in preds["predictions"]:
                class_id = CLASS_MAP[info["class_name"]]
                x_center, y_center, w_norm, h_norm = xyxy_to_yolo(det["x"] - det["width"] / 2,
                                                                   det["y"] - det["height"] / 2,
                                                                   det["width"],
                                                                   det["height"],
                                                                   img_w, img_h)
                all_detections.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save merged detections to YOLO label file
        label_path = os.path.join(OUTPUT_DIR, "labels", "all", os.path.splitext(img_file)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(all_detections))

    print(f"✅ Done! Pseudo-labels saved in {OUTPUT_DIR}/labels/all/ and images in {OUTPUT_DIR}/images/all/")

if __name__ == "__main__":
    main()
