# Data Cleaning

This folder contains preprocessing and cleaning scripts for preparing datasets.  
More scripts will be added here later as the pipeline grows.

---

##  Current Scripts

### `rename_images.py`
Renames images in a folder sequentially (`0.jpg, 1.jpg, 2.jpg...`).  
If a YOLO labels folder is provided, it also renames the corresponding `.txt` files to keep image–label pairs aligned.

#### Usage
- Rename images only:
```bash
python utils/data_cleaning/rename_images.py ./data/images
```
- Rename images and YOLO labels:
```bash
python utils/data_cleaning/rename_images.py ./data/images --labels ./data/labels
```
- Start numbering at `5`:
```bash
python utils/data_cleaning/rename_images.py ./data/images --labels ./data/labels --start 5
```

- **Shuffle first, then rename:**
```bash
python utils/data_cleaning/rename_images.py ./data/images --labels ./data/labels --shuffle
```
- **Shuffle reproducibly:**
```bash
python utils/data_cleaning/rename_images.py ./data/images --labels ./data/labels --shuffle --seed 42
```
