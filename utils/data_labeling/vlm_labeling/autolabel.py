import argparse, yaml
from pathlib import Path
from PIL import Image
from groundingdino.util.inference import load_image
from .backends.groundingdino_backend import GroundingDINOBackend
from .io_yolo import write_yolo_labels

BACKENDS = {"groundingdino": GroundingDINOBackend}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--source", required=True, help="Folder of images")
    p.add_argument("--out", default="runs/autolabel")
    p.add_argument("--backend", default="groundingdino")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    backend_cls = BACKENDS[args.backend]
    backend = backend_cls(**cfg["backend"])
    prompts = cfg["classes"]["aliases"]
    alias_map = cfg["classes"]["alias_map"]

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in Path(args.source).rglob("*") if p.suffix.lower() in exts]

    out_dir = Path(args.out) / "labels"
    for img in images:
        img_src, image = load_image(str(img))
        detections = backend.predict(image, prompts)
        write_yolo_labels(img, detections, alias_map, out_dir)

    print(f"Done. Labels saved in {out_dir}")

if __name__ == "__main__":
    main()
