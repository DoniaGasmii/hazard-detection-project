import os
from roboflow import Roboflow

# ==========================
# USER CONFIGURATION
# ==========================

API_KEY = "wK2zDNvQOIzgQKj8nGMs"

CLASS_MAP = {
    "worker": 0,
    "helmet": 1,
    "harness": 2,
    "rope": 3,
    "scaffolding": 4,
    "barricade": 5,
    "guardrail": 6,
    "edge": 7,
    "Ladder": 8
}

MODELS = {
    # "worker_model": {
    #     "project": "worker-yssgk",
    #     "version": 1,
    #     "class_names": ["worker"]
    # },
    "helmet_model": {
        "project": "helmet-3sbkd",
        "version": 1,
        "class_names": ["helmet"]
    },
    # "harness_model": {
    #     "project": "harness-xmpgb",
    #     "version": 2,
    #     "class_names": ["harness"]
    # },
    "rope_model": {
        "project": "rope-ngeva",
        "version": 2,
        "class_names": ["rope"]
    },
    "scaffolding_model": {
        "project": "scaffolding-gckjd",
        "version": 2,
        "class_names": ["scaffolding"]
    },
    "barricade_model": {
        "project": "barricade-3hoeg",
        "version": 1,
        "class_names": ["barricade"]
    },
    "guardrail_edge_model": {  # Multi-class model
        "project": "guardrail-open-edge-cvtxk",
        "version": 1,
        "class_names": ["guardrail", "edge"]
    },
    "ladder_model": {
        "project": "ladder-mxjbs-hm8us",
        "version": 1,
        "class_names": ["ladder"]
    }
}



IMAGES_DIR = "data_sample/raw_images"
OUTPUT_DIR = "data_sample/labeled_yolo_dataset"
CONF_THRESHOLD = 0.3

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
    import cv2
    rf = Roboflow(api_key=API_KEY)
    ensure_dirs()

    # Copy images into dataset/images/all
    for img_file in os.listdir(IMAGES_DIR):
        os.system(f'cp "{os.path.join(IMAGES_DIR, img_file)}" "{os.path.join(OUTPUT_DIR, "images", "all", img_file)}"')

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_file in image_files:
        all_detections = []
        img_path = os.path.join(IMAGES_DIR, img_file)

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # Run each model and collect predictions
        for model_name, info in MODELS.items():
            project = rf.workspace().project(info["project"])
            model = project.version(info["version"]).model

            preds = model.predict(img_path, confidence=CONF_THRESHOLD).json()

            for det in preds["predictions"]:
                if "class" not in det:
                    print(f"Skipping non-object-detection prediction: {det.get('prediction_type', 'unknown')}")
                    continue

                detected_class = det["class"]
                if detected_class not in CLASS_MAP:
                    print(f"Warning: Detected class '{detected_class}' not in CLASS_MAP. Skipping.")
                    continue

                class_id = CLASS_MAP[detected_class]
                x_center, y_center, w_norm, h_norm = xyxy_to_yolo(
                    det["x"] - det["width"] / 2,
                    det["y"] - det["height"] / 2,
                    det["width"],
                    det["height"],
                    img_w, img_h
                )
                all_detections.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

        # Save merged detections for this image
        label_path = os.path.join(OUTPUT_DIR, "labels", "all", os.path.splitext(img_file)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(all_detections))

    print(f"Done! Pseudo-labels saved in {OUTPUT_DIR}/labels/all/ and images in {OUTPUT_DIR}/images/all/")

if __name__ == "__main__":
    main()
