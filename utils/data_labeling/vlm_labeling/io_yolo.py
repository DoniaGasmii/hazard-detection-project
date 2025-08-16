from pathlib import Path
from PIL import Image
from .schema import Detections

def to_yolo_line(cls_id, x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

def write_yolo_labels(img_path, detections: Detections, alias_map: dict, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    w, h = Image.open(img_path).size
    lines = []
    for det in detections:
        if det.label not in alias_map:
            continue
        cls_id = alias_map[det.label]
        x1, y1, x2, y2 = det.bbox_xyxy
        lines.append(to_yolo_line(cls_id, x1, y1, x2, y2, w, h))
    (out / (Path(img_path).stem + ".txt")).write_text("".join(lines))
