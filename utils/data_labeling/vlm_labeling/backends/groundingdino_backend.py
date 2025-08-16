import urllib.request
from .._paths import resolve_path, ensure_parent
from groundingdino.util.inference import load_model, predict
from ..schema import Detection, Detections

_SWINT_PTH_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth"
_SWINT_CFG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

def _maybe_download(url: str, dst_str: str):
    dst = ensure_parent(resolve_path(dst_str))
    if not dst.exists():
        print(f"[autolabel] downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst.as_posix())
    return dst

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

        self.model = load_model(config_path, weights_path, device=device)
        self.box_thr = box_thr
        self.text_thr = text_thr

    def predict(self, image, prompts: list[str]) -> Detections:
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=", ".join(prompts),
            box_threshold=self.box_thr,
            text_threshold=self.text_thr,
        )
        return [
            Detection(label=p.lower().strip(),
                      bbox_xyxy=tuple(map(float, b.tolist())),
                      score=float(s))
            for b, s, p in zip(boxes, logits, phrases)
        ]
