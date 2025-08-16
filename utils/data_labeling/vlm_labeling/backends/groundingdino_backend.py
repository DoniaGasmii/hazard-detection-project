import urllib.request
from .._paths import resolve_path, ensure_parent
from groundingdino.util.inference import load_model, predict
from ..schema import Detection, Detections
import re
import torch

_SWINT_PTH_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
_SWINT_CFG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

def _maybe_download(url: str, dst_str: str):
    dst = ensure_parent(resolve_path(dst_str))
    if not dst.exists():
        print(f"[autolabel] downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst.as_posix())
    return dst

def _normalize_phrase(p: str) -> str:
    p = p.lower().strip()
    p = re.sub(r"[^a-z0-9\s]", "", p)   # drop punctuation
    p = re.sub(r"\s+", " ", p)          # collapse spaces
    # common equivalences
    p = p.replace("hardhat", "hard hat")
    p = p.replace("guard rail", "guardrail")
    p = p.replace("hand rail", "handrail")
    p = p.replace("safety line", "harness rope")
    return p

class GroundingDINOBackend:
    def __init__(self, weights_path, config_path, box_thr=0.3, text_thr=0.25, device="cuda", auto_download=True):
        # resolve paths relative to repo root / expand ~ and env vars
        weights_path = resolve_path(weights_path).as_posix()
        config_path  = resolve_path(config_path).as_posix()

        # optional auto-download if files are missing
        if auto_download:
            if "groundingdino_swint_ogc.pth" in weights_path:
                _maybe_download(_SWINT_PTH_URL, weights_path)
            if "GroundingDINO_SwinT_OGC.py" in config_path:
                _maybe_download(_SWINT_CFG_URL, config_path)
        if device == "cuda" and not torch.cuda.is_available():
            print("[autolabel] CUDA not available, falling back to CPU.")
            device = "cpu"
        self.device = device
        self.model = load_model(config_path, weights_path, device=self.device)
        self.box_thr = box_thr
        self.text_thr = text_thr

    def predict(self, image, prompts: list[str]) -> Detections:
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            device=self.device,
            caption=", ".join(prompts),
            box_threshold=self.box_thr,
            text_threshold=self.text_thr,
        )
        dets = []
        for (x1, y1, x2, y2), score, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
            dets.append(Detection(label=_normalize_phrase(phrase),
                                  bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                                  score=float(score)))
        return dets
