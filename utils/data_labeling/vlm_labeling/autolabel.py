import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from groundingdino.util.inference import load_image, annotate
from .io_yolo import write_yolo_labels
from .postprocess import classwise_nms  # alias filter + canonicalize + NMS

from .backends.yoloworld_backend import YOLOWorldBackend
from .backends.groundingdino_backend import GroundingDINOBackend

BACKENDS = {
    "groundingdino": GroundingDINOBackend,
    "yoloworld": YOLOWorldBackend,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML with backend + classes")
    parser.add_argument("--source", required=True, help="Folder of images (searched recursively)")
    parser.add_argument("--out", default="runs/autolabel", help="Output folder")
    parser.add_argument("--backend", default="groundingdino", choices=BACKENDS.keys())
    parser.add_argument("--save", action="store_true", help="Save visualizations under out/viz")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="IoU threshold for class-wise NMS")
    parser.add_argument("--min-rel-area", type=float, default=0.0,
                        help="Drop boxes smaller than this fraction of image area (e.g., 0.002)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Init backend (it should handle CUDA->CPU fallback internally)
    backend_cls = BACKENDS[args.backend]
    backend = backend_cls(**cfg["backend"])

    class_names = cfg["classes"]["names"]
    prompts = cfg["classes"]["aliases"]          # list of alias phrases to prompt
    alias_map = cfg["classes"]["alias_map"]      # phrase -> class_id

    # Discover images
    src_path = Path(args.source).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    images = sorted([p for p in src_path.rglob("*") if p.suffix in exts])

    print(f"[autolabel] scanning: {src_path}")
    print(f"[autolabel] found {len(images)} images in {src_path}")

    if not images:
        print("[autolabel] no images found. Exiting.")
        return

    # Output dirs
    out_dir = Path(args.out)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    if args.save:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Canonical alias map for the writer (name -> id)
    canon_alias_map = {class_names[i]: i for i in range(len(class_names))}

    # Timing stats
    t_total = 0.0
    total_kept = 0

    t_all_start = time.time()
    for idx, img in enumerate(images, 1):
        # Load image (GroundingDINO helper returns (path_str, np.ndarray RGB))
        _, image_rgb = load_image(str(img))

        # Inference timing (per image)
        dets = backend.predict(image_rgb, prompts)  # raw phrases

        # Post-process: filter to our classes, canonical names, class-wise NMS
        W, H = Image.open(img).size
        kept = classwise_nms(
            dets, alias_map, class_names,
            iou_thr=args.nms_iou,
            min_rel_area=args.min_rel_area,
            img_wh=(W, H),
        )
        total_kept += len(kept)

        print(f"[autolabel] {idx:04d}/{len(images)} {img.name}: raw={len(dets)} kept={len(kept)}")

        # Build a lightweight view for writer (supports canonical labels)
        class Simple:
            def __init__(self, k):
                self.label = k.label          # canonical name
                self.bbox_xyxy = k.bbox_xyxy
                self.score = k.score
                self.cls_id = k.cls_id

        kept_view = [Simple(k) for k in kept]
        write_yolo_labels(img, kept_view, canon_alias_map, labels_dir.as_posix(), nms_iou=args.nms_iou  )

        # Optional visualization
        if args.save and kept:
            try:
                boxes_np = np.array([k.bbox_xyxy for k in kept], dtype=np.float32)
                logits_np = np.array([k.score for k in kept], dtype=np.float32)
                boxes_t = torch.from_numpy(boxes_np) if len(boxes_np) else torch.empty((0, 4), dtype=torch.float32)
                logits_t = torch.from_numpy(logits_np) if len(logits_np) else torch.empty((0,), dtype=torch.float32)
                phrases = [f"{k.label}" for k in kept]  # canonical label for viz

                img_rgb = np.array(Image.open(img).convert("RGB"))
                annotated = annotate(image_source=img_rgb, boxes=boxes_t, logits=logits_t, phrases=phrases)
                (viz_dir / img.name).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(annotated).save(viz_dir / img.name)
            except Exception as e:
                print(f"[autolabel][viz] failed for {img.name}: {e}")

    t_all = time.time() - t_all_start
    print(f"Done. Labels saved in {labels_dir}")
    print(f"[autolabel] processed {len(images)} images | "
          f"total time={t_all:.2f}s | avg end-to-end={t_all/len(images):.3f}s/img | "
          f"avg inference={t_total/len(images):.3f}s/img | total kept boxes={total_kept}")


if __name__ == "__main__":
    main()
