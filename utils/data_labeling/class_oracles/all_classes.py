# utils/data_labeling/class_labeling/all_classes.py
"""
Run multiple class-specific labelers in sequence.
Each stage reads from previous labels (labels_in) and writes merged labels to labels_out.
You can keep images constant and only change the labels folder per stage.
"""
from pathlib import Path
import subprocess
import sys
import argparse

def run(cmd: list[str]):
    print("[all_classes] running:", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        sys.exit(ret.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with images")
    ap.add_argument("--labels-dino", required=True, help="Labels produced by DINO stage (input to first specialist)")
    ap.add_argument("--out-root", required=True, help="Root folder to create per-stage merged labels")
    ap.add_argument("--rf-api-key", default=None, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    # Scaffolding
    ap.add_argument("--scaf-project", required=True)
    ap.add_argument("--scaf-version", type=int, required=True)
    # Harness (example second specialist)
    ap.add_argument("--harness-project", default=None)
    ap.add_argument("--harness-version", type=int, default=None)
    args = ap.parse_args()

    images = Path(args.images)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Stage 1: scaffolding
    stage1 = out_root / "labels_after_scaffolding"
    run([
        sys.executable, "-m", "utils.data_labeling.class_labeling.scaffolding",
        "--images", images.as_posix(),
        "--labels-in", args.labels_dino,
        "--labels-out", stage1.as_posix(),
        "--project", args.scaf-project if hasattr(args, "scaf-project") else args.scaf_project,  # handle dash/underscore
        "--version", str(args.scaf_version),
        "--api-key", args.rf_api_key or ""
    ])

    # Stage 2: harness (optional)
    labels_in_next = stage1
    if args.harness_project and args.harness_version:
        stage2 = out_root / "labels_after_harness"
        run([
            sys.executable, "-m", "utils.data_labeling.class_labeling.harness",
            "--images", images.as_posix(),
            "--labels-in", labels_in_next.as_posix(),
            "--labels-out", stage2.as_posix(),
            "--project", args.harness_project,
            "--version", str(args.harness_version),
            "--api-key", args.rf_api_key or ""
        ])
        labels_in_next = stage2

    print(f"[all_classes] done. final labels in: {labels_in_next}")

if __name__ == "__main__":
    main()
