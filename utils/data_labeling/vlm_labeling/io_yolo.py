# utils/data_labeling/vlm_labeling/io_yolo.py
from pathlib import Path
from typing import Iterable, Dict, Tuple, List, Optional
from PIL import Image
import numpy as np

try:
    import torch
    from torchvision.ops import nms as tv_nms
except Exception:
    torch = None
    tv_nms = None


def _clip_xyxy(x1, y1, x2, y2, w, h) -> Tuple[float, float, float, float]:
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    xa = max(0.0, min(xa, w))
    xb = max(0.0, min(xb, w))
    ya = max(0.0, min(ya, h))
    yb = max(0.0, min(yb, h))
    return xa, ya, xb, yb


def _xyxy_to_yolo(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def _classwise_nms_for_writer(
    dets: List, nms_iou: float, w: int, h: int
) -> List:
    """Dedupe per class inside the writer as a final safety net."""
    by_cls: Dict[int, List] = {}
    for d in dets:
        cls_id = getattr(d, "cls_id", None)
        if cls_id is None:
            continue
        by_cls.setdefault(cls_id, []).append(d)

    kept_all: List = []
    for cls_id, items in by_cls.items():
        if not items:
            continue
        boxes = []
        scores = []
        for d in items:
            x1, y1, x2, y2 = d.bbox_xyxy
            x1, y1, x2, y2 = _clip_xyxy(float(x1), float(y1), float(x2), float(y2), float(w), float(h))
            boxes.append([x1, y1, x2, y2])
            scores.append(float(getattr(d, "score", 1.0)))
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if len(boxes) == 0:
            continue
        if tv_nms is not None and torch is not None:
            keep_idx = tv_nms(torch.tensor(boxes), torch.tensor(scores), nms_iou).tolist()
        else:
            keep_idx = _nms_numpy(boxes, scores, nms_iou)
        for i in keep_idx:
            kept_all.append(items[i])
    return kept_all


def write_yolo_labels(
    img_path,
    detections: Iterable,
    alias_map: Dict[str, int],           # canonical name -> id
    out_dir: str,
    min_wh: float = 1e-6,
    always_create: bool = True,
    nms_iou: float = 0.5,                 # extra safety NMS in writer
) -> int:
    """
    Writes <stem>.txt in YOLO format from *postprocessed* detections,
    but also runs a final class-wise NMS to guarantee no overlaps in txt.

    Returns: number of lines written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    w, h = Image.open(img_path).size
    stem = Path(img_path).stem

    # Resolve cls_id for each detection and run class-wise NMS again
    dets_with_id = []
    for d in detections:
        cls_id = getattr(d, "cls_id", None)
        if cls_id is None:
            label = getattr(d, "label", None)
            if label not in alias_map:
                continue
            cls_id = alias_map[label]
        # store a small tuple we can use again
        dets_with_id.append(type("D", (), {
            "cls_id": cls_id,
            "bbox_xyxy": d.bbox_xyxy,
            "score": float(getattr(d, "score", 1.0)),
        }))

    dets_final = _classwise_nms_for_writer(dets_with_id, nms_iou=nms_iou, w=w, h=h)

    lines: List[str] = []
    for d in dets_final:
        x1, y1, x2, y2 = d.bbox_xyxy
        x1, y1, x2, y2 = _clip_xyxy(float(x1), float(y1), float(x2), float(y2), float(w), float(h))
        if (x2 - x1) <= min_wh or (y2 - y1) <= min_wh:
            continue
        cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, w, h)
        # clip to [0,1]
        cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw)); bh = max(0.0, min(1.0, bh))
        if bw <= 0.0 or bh <= 0.0:
            continue
        lines.append(f"{d.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    label_path = out / f"{stem}.txt"
    if lines or always_create:
        label_path.write_text("".join(lines))
    return len(lines)
