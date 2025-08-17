# utils/data_cleaning/rename_images.py
import os
import argparse
import uuid
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def safe_two_phase_rename(pairs: List[Tuple[str, str]]) -> None:
    """
    Rename many files without collisions using a two-phase approach:
      1) src -> src.__tmp_<uuid>
      2) src.__tmp_<uuid> -> dst
    `pairs` is a list of (src, dst) absolute paths.
    """
    tmp_map = []
    # phase 1
    for src, dst in pairs:
        if src == dst:
            continue
        if not os.path.exists(src):
            continue
        tmp = f"{src}.__tmp_{uuid.uuid4().hex}"
        os.replace(src, tmp)
        tmp_map.append((tmp, dst))
    # phase 2
    for tmp, dst in tmp_map:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(tmp, dst)

def make_sequential_names(n: int, start: int, ext: str) -> str:
    return f"{start + n}{ext}"

def rename_images_and_labels(images_dir: str, labels_dir: str = None, start: int = 0) -> None:
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images folder not found: {images_dir}")

    files = sorted(os.listdir(images_dir))
    img_files = [f for f in files if is_image(f) and os.path.isfile(os.path.join(images_dir, f))]

    if not img_files:
        print("No image files found to rename.")
        return

    # Build planned renames for images
    planned_img_moves: List[Tuple[str, str]] = []
    planned_lbl_moves: List[Tuple[str, str]] = []

    for idx, fname in enumerate(img_files):
        src_img = os.path.join(images_dir, fname)
        _, ext = os.path.splitext(fname)
        new_img_name = make_sequential_names(idx, start, ext.lower())
        dst_img = os.path.join(images_dir, new_img_name)
        planned_img_moves.append((src_img, dst_img))

        # Handle labels if provided
        if labels_dir:
            stem = os.path.splitext(fname)[0]
            src_lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(src_lbl):
                new_lbl_name = os.path.splitext(new_img_name)[0] + ".txt"
                dst_lbl = os.path.join(labels_dir, new_lbl_name)
                planned_lbl_moves.append((src_lbl, dst_lbl))

    # Warnings for missing labels (non-fatal)
    if labels_dir:
        missing = []
        for fname in img_files:
            stem = os.path.splitext(fname)[0]
            if not os.path.isfile(os.path.join(labels_dir, stem + ".txt")):
                missing.append(fname)
        if missing:
            print(f"Warning: {len(missing)} images have no matching .txt in labels folder (they will still be renamed):")
            for m in missing[:10]:
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

    # Execute two-phase rename to avoid collisions
    safe_two_phase_rename(planned_img_moves + planned_lbl_moves)

    print(f"Renamed {len(planned_img_moves)} images in '{images_dir}' (starting at {start}).")
    if labels_dir:
        print(f"Renamed {len(planned_lbl_moves)} label files in '{labels_dir}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Rename images sequentially (0,1,2,...) and optionally rename matching YOLO .txt labels."
    )
    parser.add_argument("images", type=str, help="Path to the folder containing images.")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to YOLO labels folder (with .txt). If provided, matching labels are renamed too.")
    parser.add_argument("--start", type=int, default=0, help="Starting index for renaming (default: 0).")

    args = parser.parse_args()

    labels_dir = args.labels
    if labels_dir and not os.path.isdir(labels_dir):
        raise ValueError(f"Labels folder not found: {labels_dir}")

    rename_images_and_labels(args.images, labels_dir, args.start)

if __name__ == "__main__":
    main()
