# utils/data_labeling/class_labeling/base_class_labeler.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os, sys, json
import numpy as np
from PIL import Image

# --- Optional: use torch/torchvision.ops.nms if available (faster), else fallback to numpy ---
try:
    import torch
    from torchvision.ops import nms as tv_nms
except Exception:
    torch = None
    tv_nms = None

# --- Roboflow (installed via `pip install roboflow`) ---
try:
    from roboflow import Roboflow
except Exception as e:
    Roboflow = None
    print("[class_labeling] Roboflow not installed. Run: pip install roboflow", file=sys.stderr)

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class DetXYXY:
    cls_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    source: str  # e.g., "roboflow_scaffolding" (provenance)

@dataclass
class YoloBox:
    cls_id: int
    bbox_xyxy: Tuple[float, float, float, float]

# ----------------------------
# Geometry helpers
# ----------------------------
def _clip_xyxy(x1, y1, x2, y2, W, H):
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    xa = max(0.0, min(xa, W))
    xb = max(0.0, min(xb, W))
    ya = max(0.0, min(ya, H))
    yb = max(0.0, min(yb, H))
    return xa, ya, xb, yb

def _xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h

def _yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return x1, y1, x2, y2

def _xyxy_to_yolo(x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh

def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)

# ----------------------------
# NMS per-class
# ----------------------------
def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
        area_o = (boxes[order[1:], 2]-boxes[order[1:], 0]) * (boxes[order[1:], 3]-boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep

def classwise_nms_for_list(dets: List[DetXYXY], iou_thr: float, W: int, H: int) -> List[DetXYXY]:
    """Run per-class NMS on an already merged list of DetXYXY (from multiple models)."""
    by_cls: Dict[int, List[DetXYXY]] = {}
    for d in dets:
        by_cls.setdefault(d.cls_id, []).append(d)

    kept_all: List[DetXYXY] = []
    for cls_id, items in by_cls.items():
        if not items: 
            continue
        boxes = []
        scores = []
        for d in items:
            x1, y1, x2, y2 = _clip_xyxy(*d.bbox_xyxy, W, H)
            boxes.append([x1, y1, x2, y2])
            scores.append(d.score)
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if len(boxes) == 0:
            continue
        if tv_nms is not None and torch is not None:
            keep_idx = tv_nms(torch.tensor(boxes), torch.tensor(scores), iou_thr).tolist()
        else:
            keep_idx = _nms_numpy(boxes, scores, iou_thr)
        for i in keep_idx:
            kept_all.append(items[i])
    return kept_all

# ----------------------------
# YOLO IO helpers
# ----------------------------
def load_existing_yolo(txt_path: Path, img_path: Path) -> List[YoloBox]:
    if not txt_path.exists():
        return []
    W, H = Image.open(img_path).size
    out: List[YoloBox] = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, bw, bh, W, H)
        x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
        out.append(YoloBox(cls_id=cls, bbox_xyxy=(x1, y1, x2, y2)))
    return out

def write_yolo_labels_xyxy(
    img_path: Path, 
    dets_xyxy: List[DetXYXY], 
    out_dir: Path, 
    always_create: bool = True,
    min_wh: float = 1e-6
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = Image.open(img_path).size
    stem = img_path.stem
    lines: List[str] = []
    for d in dets_xyxy:
        x1, y1, x2, y2 = _clip_xyxy(*d.bbox_xyxy, W, H)
        if (x2 - x1) <= min_wh or (y2 - y1) <= min_wh:
            continue
        cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, W, H)
        # clip to [0,1]
        cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw)); bh = max(0.0, min(1.0, bh))
        if bw <= 0.0 or bh <= 0.0:
            continue
        lines.append(f"{d.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    label_path = out_dir / f"{stem}.txt"
    if lines or always_create:
        label_path.write_text("".join(lines))
    return len(lines)

# ----------------------------
# Roboflow helpers
# ----------------------------
def _get_roboflow(api_key: Optional[str] = None):
    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Set ROBOFLOW_API_KEY env var or pass api_key=...")
    if Roboflow is None:
        raise RuntimeError("Roboflow package not installed. Run: pip install roboflow")
    return Roboflow(api_key=key)

def _resolve_project(rf: "Roboflow", slug: str):
    """
    slug may be 'workspace/project' or just 'project' (uses default workspace).
    """
    if "/" in slug:
        ws, proj = slug.split("/", 1)
        return rf.workspace(ws).project(proj)
    return rf.workspace().project(slug)

def predict_roboflow_det_xyxy(
    img_path: Path,
    project_slug: str,
    version: int,
    class_to_id: Dict[str, int],
    conf: float = 0.5,
    iou: float = 0.5,
    api_key: Optional[str] = None,
    provenance: str = "roboflow",
) -> List[DetXYXY]:
    """
    Runs a Roboflow OD model and returns a list of DetXYXY in ABSOLUTE XYXY pixel coords.
    class_to_id maps model class names (lowercased) to your global class ids.
    """
    rf = _get_roboflow(api_key)
    project = _resolve_project(rf, project_slug)
    model = project.version(version).model

    preds = model.predict(img_path.as_posix(), confidence=conf, overlap=iou).json()
    img = Image.open(img_path); W, H = img.size

    out: List[DetXYXY] = []
    for det in preds.get("predictions", []):
        label = str(det.get("class", "")).lower().strip()
        if label not in class_to_id:
            continue
        cls_id = class_to_id[label]
        # Roboflow returns center x,y and width/height in pixels
        x = float(det["x"]) - float(det["width"]) / 2.0
        y = float(det["y"]) - float(det["height"]) / 2.0
        w = float(det["width"]); h = float(det["height"])
        x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
        x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
        score = float(det.get("confidence", det.get("confidence_score", 1.0)))
        out.append(DetXYXY(cls_id=cls_id, bbox_xyxy=(x1, y1, x2, y2), score=score, source=provenance))
    return out

# ----------------------------
# Merge logic
# ----------------------------
def merge_with_existing(
    img_path: Path,
    existing_labels_dir: Path,
    new_dets: List[DetXYXY],
    dedupe_iou: float = 0.6,
) -> List[DetXYXY]:
    """
    Keep all existing boxes, add new ones that don't overlap (same class, IoU>=thr) with existing.
    """
    W, H = Image.open(img_path).size
    merged: List[DetXYXY] = []
    # 1) existing
    existing = load_existing_yolo(existing_labels_dir / f"{img_path.stem}.txt", img_path)
    for e in existing:
        merged.append(DetXYXY(cls_id=e.cls_id, bbox_xyxy=e.bbox_xyxy, score=1.0, source="existing"))
    # 2) add new that are not duplicates of existing for same class
    for d in new_dets:
        dup = False
        for e in existing:
            if e.cls_id == d.cls_id and iou_xyxy(e.bbox_xyxy, d.bbox_xyxy) >= dedupe_iou:
                dup = True; break
        if not dup:
            merged.append(d)
    return merged

def finalize_and_write(
    img_path: Path,
    dets: List[DetXYXY],
    out_labels_dir: Path,
    nms_iou: float = 0.5,
) -> int:
    W, H = Image.open(img_path).size
    # Final safety NMS per class (across sources)
    dets_final = classwise_nms_for_list(dets, iou_thr=nms_iou, W=W, H=H)
    # Write YOLO file
    return write_yolo_labels_xyxy(img_path, dets_final, out_labels_dir)
