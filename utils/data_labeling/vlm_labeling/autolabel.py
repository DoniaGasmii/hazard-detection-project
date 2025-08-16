import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from groundingdino.util.inference import load_image, annotate
from .backends.groundingdino_backend import GroundingDINOBackend
from .io_yolo import write_yolo_labels
from .postprocess import classwise_nms  # NEW
import time

BACKENDS = {"groundingdino": GroundingDINOBackend}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="runs/autolabel")
    parser.add_argument("--backend", default="groundingdino", choices=BACKENDS.keys())
    parser.add_argument("--save", action="store_true", help="save visualization under out/viz")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="IoU threshold for class-wise NMS")
    parser.add_argument("--min-rel-area", type=float, default=0.0, help="drop boxes smaller than this fraction of image area (e.g., 0.002)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backend_cls = BACKENDS[args.backend]
    backend = backend_cls(**cfg["backend"])

    class_names = cfg["classes"]["names"]
    prompts = cfg["classes"]["aliases"]          # list of alias phrases to prompt
    alias_map = cfg["classes"]["alias_map"]      # phrase -> class_id

    # discover images
    src_path = Path(args.source).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    images = sorted([p for p in src_path.rglob("*") if p.suffix in exts])
    print(f"[autolabel] scanning: {src_path}")
    print(f"[autolabel] found {len(images)} images in {src_path}")

    out_dir = Path(args.out)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    if args.save:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    for img in images:
        _, image_rgb = load_image(str(img))

        # raw detections from backend (still using phrases)
        dets = backend.predict(image_rgb, prompts)

        # post-process: alias filter + collapse to canonical + class-wise NMS
        W, H = Image.open(img).size
        kept = classwise_nms(
            dets, alias_map, class_names,
            iou_thr=args.nms_iou,
            min_rel_area=args.min_rel_area,
            img_wh=(W, H),
        )

        # debug
        print(f"[autolabel] {img.name}: raw={len(dets)} kept={len(kept)}")

        # write YOLO labels with canonical class IDs
        # build a lightweight view compatible with your writer
        class Simple:
            def __init__(self, k):  # k is PPDet
                self.label = k.label
                self.bbox_xyxy = k.bbox_xyxy
                self.score = k.score
                self.cls_id = k.cls_id
        kept_view = [Simple(k) for k in kept]

        # Slight tweak: your writer expects alias_map lookups by phrase.
        # We can create a tiny alias_map for canonical names:
        canon_alias_map = {class_names[i]: i for i in range(len(class_names))}
        write_yolo_labels(img, kept_view, canon_alias_map, labels_dir.as_posix())

        # visualization (canonical names only)
        if args.save and kept:
            try:
                boxes_np = np.array([k.bbox_xyxy for k in kept], dtype=np.float32)
                logits_np = np.array([k.score for k in kept], dtype=np.float32)
                boxes_t = torch.from_numpy(boxes_np)
                logits_t = torch.from_numpy(logits_np)
                phrases = [f"{k.label} {k.score:.2f}" for k in kept]  # canonical name
                img_rgb = np.array(Image.open(img).convert("RGB"))
                annotated = annotate(image_source=img_rgb, boxes=boxes_t, logits=logits_t, phrases=phrases)
                Image.fromarray(annotated).save(viz_dir / img.name)
            except Exception as e:
                print(f"[autolabel][viz] failed for {img.name}: {e}")
    
    print(f"Done. Labels saved in {labels_dir}")
    elapsed = time.time() - start_time
    print(f"[autolabel] processed {len(images)} images in {elapsed:.2f}s "
      f"({elapsed/len(images):.3f}s per image on average)")
