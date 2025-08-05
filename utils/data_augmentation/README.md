# Augmentation CLI – Visualize YOLO Image Augmentations

A command-line tool to **visualize before/after** image augmentations for YOLO-format datasets.  
It uses functions from `augmentation_functions.py` and helpers from `visualizations/draw.py`.  
No files are saved — results are shown side-by-side in a Matplotlib window.

---

## 📂 Available Augmentations

- `rotation` – Random rotation
- `brightness_noise` – Adjust brightness/contrast + Gaussian noise
- `occlusion` – Coarse dropout (simulate occlusion)
- `motion_blur` – Motion blur effect
- `color_jitter` – Change brightness, contrast, saturation, hue
- `distortion` – Lens distortion
- `zoom_out` – Zoom out with padding
- `fog` – Fog overlay
- `rain` – Synthetic rain streaks

---

## ⚙️ Usage

```bash
python augment_cli.py \
  --image path/to/image.jpg \
  --label path/to/label.txt \
  --augmentation <augmentation_name>
```
#### Example:
```bash
python augment_cli.py \
  --image ../test_image/images/train/test_image.jpg \
  --label ../test_image/yolo_labels/train/test_label.txt \
  --augmentation fog
```
This will show a side-by-side before/after visualization of the fog effect on the test image.
