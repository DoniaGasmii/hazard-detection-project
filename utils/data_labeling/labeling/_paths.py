from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/utils/labeling -> up 2

def resolve_path(p: str) -> Path:
    """
    Expand ~ and env vars, and make relative paths resolve from the repo root.
    """
    p = os.path.expandvars(os.path.expanduser(p))
    path = Path(p)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
