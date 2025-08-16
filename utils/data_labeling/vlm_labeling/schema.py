from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Detection:
    label: str
    bbox_xyxy: Tuple[float, float, float, float]
    score: float

Detections = List[Detection]
