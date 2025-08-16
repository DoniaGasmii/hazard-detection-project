import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from groundingdino.util.inference import load_image, annotate
from .backends.groundingdino_backend import GroundingDINOBackend
from .io_yolo import write_yolo_labels

BACKENDS = {"groundingdino": GroundingDINOBackend}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config with backend + classes/aliases")
    parser.add_argument("--source", required=True, help="Folder with images (searched recursively)")
    parser.add_argument("--out", default="runs/autolabel", help="Output run directory")
    parser.add_argument("--backend", default="groundingdino", choices=BACKENDS.keys())
    parser.add_argument("--save", action="store_true", help="Save visualization images under out/viz")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Init backend (handles CPU fallback internally if you added it there)
    backend_cls = BACKENDS[args.backend]
    backend = backend_cls(**cfg["backend"])

    # Prompts / alias map
    prompts = cfg["classes"]["aliases"]
    alias_map = cfg["classes"]["alias_map"]

    # Discover images
    src_path = Path(args.source).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    images = sorted([p for p in src_path.rglob("*") if p.suffix in exts])

    print(f"[autolabel] scanning: {src_path}")
    print(f"[autolabel] found {len(images)} images in {src_path}")

    # Output dirs
    out_dir = Path(args.out)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = out_dir / "viz"
    if args.save:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        # Load image (GroundingDINO helper returns (path_str, np.ndarray RGB))
        _, image_rgb = load_image(str(img))

        # Predict
        dets = backend.predict(image_rgb, prompts)

        # Debug line: how many detections and what phrases
        raw_phrases = [d.label for d in dets]
        kept = [d for d in dets if d.label in alias_map]
        preview = raw_phrases[:6]
        suffix = "..." if len(raw_phrases) > 6 else ""
        print(f"[autolabel] {img.name}: {len(dets)} det(s), kept {len(kept)} | phrases={preview}{suffix}")

        # Write YOLO labels (your writer should always create the .txt, even if empty)
        write_yolo_labels(img, kept, alias_map, labels_dir.as_posix())

        # Optional visualization
        if args.save and dets:
            try:
                # GroundingDINO annotate() expects torch tensors for boxes/logits
                boxes_np = np.array([d.bbox_xyxy for d in dets], dtype=np.float32)
                logits_np = np.array([d.score for d in dets], dtype=np.float32)
                boxes_t = torch.from_numpy(boxes_np) if len(boxes_np) else torch.empty((0, 4), dtype=torch.float32)
                logits_t = torch.from_numpy(logits_np) if len(logits_np) else torch.empty((0,), dtype=torch.float32)

                img_rgb = np.array(Image.open(img).convert("RGB"))
                annotated = annotate(
                    image_source=img_rgb,
                    boxes=boxes_t,
                    logits=logits_t,
                    phrases=raw_phrases,  # list[str]
                )
                Image.fromarray(annotated).save(viz_dir / img.name)
            except Exception as e:
                print(f"[autolabel][viz] failed for {img.name}: {e}")

    print(f"Done. Labels saved in {labels_dir}")


if __name__ == "__main__":
    main()
