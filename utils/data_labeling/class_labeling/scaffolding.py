# utils/data_labeling/class_labeling/scaffolding.py
from pathlib import Path
import argparse
from .base_class_labeler import (
    predict_roboflow_det_xyxy,
    merge_with_existing,
    finalize_and_write,
    DetXYXY,
)

# Adjust to YOUR global class mapping (canonical name -> id)
CLASS_TO_ID = {
    "scaffolding": 4,   # <- must match your global YAML/classes.names index
    "scaffold": 4,
    "staging": 4,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with images")
    ap.add_argument("--labels-in", required=True, help="Existing YOLO labels to preserve/merge")
    ap.add_argument("--labels-out", required=True, help="Output YOLO labels folder")
    ap.add_argument("--project", required=True, help="Roboflow project slug, e.g., workspace/scaffolding-gckjd")
    ap.add_argument("--version", type=int, required=True, help="Roboflow model version (int)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--dedupe-iou", type=float, default=0.6, help="IoU to treat specialist vs existing as duplicates")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="Final per-class NMS when writing")
    args = ap.parse_args()

    img_dir = Path(args.images)
    labels_in = Path(args.labels_in)
    labels_out = Path(args.labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    images = sorted([p for p in img_dir.rglob("*") if p.suffix in exts])

    total_written = 0
    for i, img_path in enumerate(images, 1):
        # 1) run specialist model
        sp_dets = predict_roboflow_det_xyxy(
            img_path=img_path,
            project_slug=args.project,
            version=args.version,
            class_to_id=CLASS_TO_ID,
            conf=args.conf,
            iou=args.iou,
            api_key=args.api_key,
            provenance="roboflow_scaffolding",
        )

        # 2) merge with existing (keep existing; add new non-duplicates)
        merged = merge_with_existing(img_path, labels_in, sp_dets, dedupe_iou=args.dedupe_iou)

        # 3) final safety NMS + write
        n = finalize_and_write(img_path, merged, labels_out, nms_iou=args.nms_iou)
        total_written += n
        print(f"[scaffolding] {i:04d}/{len(images)} {img_path.name}: sp={len(sp_dets)} -> wrote={n}")

    print(f"[scaffolding] done. wrote {total_written} boxes total to {labels_out}")

if __name__ == "__main__":
    main()
