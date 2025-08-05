import argparse
import cv2

import sys
import os
# Add the parent directory to sys.path so we can import visualizations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualizations.draw import *

from augmentation_functions import (
    apply_rotation, apply_brightness_noise, apply_occlusion_dropout, apply_motion_blur,
    apply_color_jitter, apply_distortion_with_boxes, apply_zoom_out,
    apply_fog_effect, apply_rain_effect
)

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO augmentations before/after.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--label", type=str, required=True, help="Path to YOLO label file.")
    parser.add_argument("--augmentation", type=str, required=True,
                        choices=["rotation", "brightness_noise", "occlusion", "motion_blur", 
                                 "color_jitter", "distortion", "zoom_out", "fog", "rain"],
                        help="Augmentation type to apply.")
    args = parser.parse_args()

    # Load image and labels
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    bboxes_full = yolo_to_xyxy(args.label, w, h)
    class_labels = [b[4] for b in bboxes_full]
    bboxes = [b[:4] for b in bboxes_full]

    # Select augmentation
    if args.augmentation == "rotation":
        aug_img, aug_boxes = apply_rotation(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "brightness_noise":
        aug_img, aug_boxes = apply_brightness_noise(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "occlusion":
        aug_img, aug_boxes = apply_occlusion_dropout(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "motion_blur":
        aug_img, aug_boxes = apply_motion_blur(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "color_jitter":
        aug_img, aug_boxes = apply_color_jitter(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "distortion":
        aug_img, aug_boxes = apply_distortion_with_boxes(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "zoom_out":
        aug_img, aug_boxes = apply_zoom_out(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "fog":
        aug_img, aug_boxes = apply_fog_effect(img, bboxes, class_labels, verbose=False)
    elif args.augmentation == "rain":
        aug_img, aug_boxes = apply_rain_effect(img, bboxes, class_labels, verbose=False)
    else:
        raise ValueError("Invalid augmentation type.")

    # Show before/after visualization
    show_before_after(img, aug_img,
                      [[*box, cls] for box, cls in zip(bboxes, class_labels)],
                      aug_boxes)

if __name__ == "__main__":
    main()
