# utils/data_cleaning/rename_images.py

import os
import argparse
import random
import uuid
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def safe_two_phase_rename(pairs: List[Tuple[str, str]]) -> None:
    """Rename many files safely without collisions (two-phase approach)."""
    tmp_map = []
    # Phase 1: move to temporary unique names
    for src, dst in pairs:
        if src == dst or not os.path.exists(src):
            continue
        tmp = f"{src}.__tmp_{uuid.uuid4().hex}"
        os.replace(src, tmp)
        tmp_map.append((tmp, dst))
    # Phase 2: move to final destination
    for tmp, dst in tmp_map:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(tmp, dst)


def rename_images_and_labels(images_dir: str, labels_dir: str = None,
                             start: int = 0, shuffle: bool = False, seed: int = None) -> None:
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images folder not found: {images_dir}")

    files = sorted(os.listdir(images_dir))
    img_files = [f for f in files if is_image(f) and os.path.isfile(os.path.join(images_dir, f))]

    if not img_files:
        print("No image files found to rename.")
        return

    # Shuffle if requested
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(img_files)

    planned_img_moves: List[Tuple[str, str]] = []
    planned_lbl_moves: List[Tuple[str, str]] = []

    for idx, fname in enumerate(img_files, start=start):
        src_img = os.path.join(images_dir, fname)
        _, ext = os.path.splitext(fname)
        new_img_name = f"{idx}{ext.lower()}"
        dst_img = os.path.join(images_dir, new_img_name)
        planned_img_moves.append((src_img, dst_img))

        if labels_dir:
            stem = os.path.splitext(fname)[0]
            src_lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(src_lbl):
                dst_lbl = os.path.join(labels_dir, f"{idx}.txt")
                planned_lbl_moves.append((src_lbl, dst_lbl))

    # Perform renaming safely
    safe_two_phase_rename(planned_img_moves + planned_lbl_moves)

    mode = "shuffled and renamed" if shuffle else "renamed"
    print(f"{mode.capitalize()} {len(planned_img_moves)} images in '{images_dir}' (starting at {start}).")
    if labels_dir:
        print(f"{mode.capitalize()} {len(planned_lbl_moves)} labels in '{labels_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename images sequentially (0,1,2,...) with optional shuffling, and rename YOLO labels consistently."
    )
    parser.add_argument("images", type=str, help="Path to the folder containing images.")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to YOLO labels folder (with .txt). If provided, matching labels are renamed too.")
    parser.add_argument("--start", type=int, default=0, help="Starting index for renaming (default: 0).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle images before renaming (default: False).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (used only if --shuffle).")

    args = parser.parse_args()

    if args.labels and not os.path.isdir(args.labels):
        raise ValueError(f"Labels folder not found: {args.labels}")

    rename_images_and_labels(args.images, args.labels, args.start, args.shuffle, args.seed)


if __name__ == "__main__":
    main()
