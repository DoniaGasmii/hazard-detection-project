from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional
import torch
import numpy as np

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None  # we'll fall back to a numpy NMS if TV isn't available


@dataclass
class PPDet:
    cls_id: int
    label: str          # canonical name for viz
    bbox_xyxy: Tuple[float, float, float, float]
    score: float


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.size == 1:
            break
        ious = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in idxs[1:]])
        idxs = idxs[1:][ious <= iou_thr]
    return keep


def classwise_nms(
    dets: Iterable,               # list of your Detection objects from backend
    alias_map: Dict[str, int],    # phrase -> class_id
    class_names: List[str],       # canonical names; index = class_id
    iou_thr: float = 0.5,
    min_rel_area: float = 0.0,
    img_wh: Optional[Tuple[int, int]] = None,  # (W,H) if using min_rel_area
) -> List[PPDet]:
    """
    - keep only labels present in alias_map (drops stray classes like 'driver')
    - map aliases -> class IDs
    - optional min area filter (relative to image area)
    - class-wise NMS (keeps highest score per overlap)
    - return canonical labels for viz
    """
    filtered: Dict[int, List[PPDet]] = {}

    img_area = None
    if min_rel_area and img_wh:
        img_area = img_wh[0] * img_wh[1]

    # 1) filter & map aliases
    for d in dets:
        if d.label not in alias_map:
            continue
        cls_id = alias_map[d.label]
        x1, y1, x2, y2 = d.bbox_xyxy
        if img_area:
            area = max(0.0, x2-x1) * max(0.0, y2-y1)
            if area < min_rel_area * img_area:
                continue
        filtered.setdefault(cls_id, []).append(
            PPDet(cls_id=cls_id, label=class_names[cls_id],
                  bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                  score=float(d.score))
        )

    # 2) NMS per class
    final: List[PPDet] = []
    for cls_id, items in filtered.items():
        if not items:
            continue
        boxes = np.array([p.bbox_xyxy for p in items], dtype=np.float32)
        scores = np.array([p.score for p in items], dtype=np.float32)
        if tv_nms is not None:
            keep = tv_nms(torch.tensor(boxes), torch.tensor(scores), iou_thr).tolist()
        else:
            keep = _nms_numpy(boxes, scores, iou_thr)
        for k in keep:
            final.append(items[k])

    return final
