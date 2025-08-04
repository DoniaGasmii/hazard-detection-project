# Rewriting the script without markdown-style triple backticks to fix syntax error.

# Full script with all augmentation functions implemented so far.

import os
import cv2
import numpy as np
import random
import albumentations as A
from visualizations.draw import show_before_after

# ------------------------------ Albumentations Wrapper ------------------------------

def apply_albumentation(image, bboxes, class_labels, transform, verbose=False):
    aug = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_img = aug['image']
    aug_boxes = [[*box, cls] for box, cls in zip(aug['bboxes'], class_labels)]

    if verbose:
        original_boxes = [[*box, cls] for box, cls in zip(bboxes, class_labels)]
        show_before_after(image, aug_img, original_boxes, aug_boxes)

    return aug_img, aug_boxes

# ------------------------------ Albumentation-based Transforms ------------------------------

def apply_rotation(image, bboxes, class_labels, angle_limit=30, verbose=False):
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=angle_limit, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return apply_albumentation(image, bboxes, class_labels, transform, verbose)

def apply_brightness_noise(image, bboxes, class_labels, brightness_limit=0.3,
                           contrast_limit=0.1, noise_var=(0.05, 5.0), verbose=False):
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0),
        A.GaussNoise(var_limit=noise_var, p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return apply_albumentation(image, bboxes, class_labels, transform, verbose)

def apply_occlusion_dropout(image, bboxes, class_labels, max_holes=8, max_height=80,
                            max_width=80, fill_value=0, verbose=False):
    transform = A.Compose([
        A.CoarseDropout(max_holes=max_holes, max_height=max_height,
                        max_width=max_width, fill_value=fill_value, p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return apply_albumentation(image, bboxes, class_labels, transform, verbose)

def apply_motion_blur(image, bboxes, class_labels, blur_limit=15, verbose=False):
    transform = A.Compose([
        A.MotionBlur(blur_limit=blur_limit, p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return apply_albumentation(image, bboxes, class_labels, transform, verbose)

def apply_color_jitter(image, bboxes, class_labels, brightness=0.3,
                       contrast=0.3, saturation=0.2, hue=0.2, verbose=False):
    transform = A.Compose([
        A.ColorJitter(brightness=brightness, contrast=contrast,
                      saturation=saturation, hue=hue, p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return apply_albumentation(image, bboxes, class_labels, transform, verbose)

# ------------------------------ Custom Transforms ------------------------------

def apply_distortion_with_boxes(image, bboxes, class_labels, k1=0.3, k2=0.1,
                                 p1=0.01, p2=0.01, k3=0.0, verbose=False):
    h, w = image.shape[:2]
    K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (w, h), cv2.CV_32FC1)
    distorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    all_pts = []
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        all_pts.extend([[x_min, y_min], [x_max, y_max]])
    all_pts = np.array(all_pts, dtype=np.float32).reshape(-1, 1, 2)
    distorted_pts = cv2.undistortPoints(all_pts, K, dist_coeffs, P=K).reshape(-1, 2)

    new_boxes = []
    for i in range(0, len(distorted_pts), 2):
        pt1, pt2 = distorted_pts[i], distorted_pts[i + 1]
        new_boxes.append([min(pt1[0], pt2[0]), min(pt1[1], pt2[1]),
                          max(pt1[0], pt2[0]), max(pt1[1], pt2[1])])
    final_boxes = [box + [cls] for box, cls in zip(new_boxes, class_labels)]

    if verbose:
        original_boxes = [box + [cls] for box, cls in zip(bboxes, class_labels)]
        show_before_after(image, distorted_img, original_boxes, final_boxes)

    return distorted_img, final_boxes

def apply_zoom_out(image, bboxes, class_labels, zoom_out_factor=0.5,
                   pad_value=114, verbose=False):
    h, w = image.shape[:2]
    new_w = int(w / zoom_out_factor)
    new_h = int(h / zoom_out_factor)
    pad_x = (new_w - w) // 2
    pad_y = (new_h - h) // 2

    padded = np.full((new_h, new_w, 3), pad_value, dtype=np.uint8)
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = image

    zoomed_bboxes = []
    for box, cls in zip(bboxes, class_labels):
        x_min, y_min, x_max, y_max = box
        new_box = [x_min + pad_x, y_min + pad_y, x_max + pad_x, y_max + pad_y, cls]
        zoomed_bboxes.append(new_box)

    if verbose:
        original_boxes = [box + [cls] for box, cls in zip(bboxes, class_labels)]
        show_before_after(image, padded, original_boxes, zoomed_bboxes)

    return padded, zoomed_bboxes

def apply_fog_effect(image, bboxes, class_labels, fog_intensity=0.5, verbose=False):
    overlay = np.full_like(image, 255, dtype=np.uint8)
    foggy = cv2.addWeighted(image, 1 - fog_intensity, overlay, fog_intensity, 0)
    final_boxes = [[*box, cls] for box, cls in zip(bboxes, class_labels)]

    if verbose:
        original_boxes = [box + [cls] for box, cls in zip(bboxes, class_labels)]
        show_before_after(image, foggy, original_boxes, final_boxes)

    return foggy, final_boxes

def apply_rain_effect(image, bboxes, class_labels, drop_length=20,
                      drop_thickness=1, rain_density=0.01, verbose=False):
    rain_img = image.copy()
    h, w = rain_img.shape[:2]
    num_drops = int(h * w * rain_density)

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        x_end = x + np.random.randint(-5, 5)
        y_end = y + drop_length
        cv2.line(rain_img, (x, y), (x_end, y_end), (200, 200, 200), drop_thickness)

    rain_img = cv2.blur(rain_img, (3, 3))
    final_boxes = [[*box, cls] for box, cls in zip(bboxes, class_labels)]

    if verbose:
        original_boxes = [box + [cls] for box, cls in zip(bboxes, class_labels)]
        show_before_after(image, rain_img, original_boxes, final_boxes)

    return rain_img, final_boxes

def apply_mosaic(img_dir, lbl_dir, load_fn, output_size=640, verbose=False):
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    selected_files = random.sample(img_files, 4)
    final_img = np.full((output_size * 2, output_size * 2, 3), 114, dtype=np.uint8)
    final_bboxes = []

    for i, file in enumerate(selected_files):
        image, bboxes, class_labels, _ = load_fn(img_dir, lbl_dir)
        h, w = image.shape[:2]

        scale = output_size / max(h, w)
        resized_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        resized_h, resized_w = resized_img.shape[:2]

        if i == 0: x1a, y1a = 0, 0
        elif i == 1: x1a, y1a = output_size, 0
        elif i == 2: x1a, y1a = 0, output_size
        else: x1a, y1a = output_size, output_size
        x2a, y2a = x1a + resized_w, y1a + resized_h
        final_img[y1a:y2a, x1a:x2a] = resized_img

        for box, cls in zip(bboxes, class_labels):
            x_min = box[0] * scale + x1a
            y_min = box[1] * scale + y1a
            x_max = box[2] * scale + x1a
            y_max = box[3] * scale + y1a
            final_bboxes.append([x_min, y_min, x_max, y_max, cls])

    if verbose:
        show_before_after(final_img, final_img, [], final_bboxes)

    return final_img, final_bboxes
