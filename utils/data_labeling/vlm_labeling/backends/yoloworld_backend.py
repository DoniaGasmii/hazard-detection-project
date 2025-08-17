# utils/data_labeling/vlm_labeling/backends/yoloworld_backend.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch

@dataclass
class Detection:
    label: str
    bbox_xyxy: Tuple[float, float, float, float]
    score: float

class YOLOWorldBackend:
    def __init__(self, weights_path: str, config_path: str = None,
                 box_thr: float = 0.3, text_thr: float = 0.25, device: str = "cpu", **kwargs):
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        # load your model here
        # self.model = yolo_world.load(weights_path, config_path).to(self.device).eval()

    def predict(self, image_rgb: np.ndarray, prompts: List[str]) -> List[Detection]:
        H, W = image_rgb.shape[:2]
        # preprocess -> tensor on self.device
        # run model with text prompts
        # boxes_xyxy, scores, phrases = <inference(...)>
        # make sure boxes are in original image coordinates
        dets: List[Detection] = []
        for (x1, y1, x2, y2), s, phrase in zip(boxes_xyxy, scores, phrases):
            if s < self.box_thr:
                continue
            dets.append(Detection(label=normalize(phrase),
                                  bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                                  score=float(s)))
        return dets

def normalize(p: str) -> str:
    p = p.lower().strip()
    # optional: strip punctuation, fix common aliases
    # p = re.sub(r"[^a-z0-9\s]", "", p)
    p = p.replace("hardhat", "hard hat")
    p = p.replace("guard rail", "guardrail")
    return p
