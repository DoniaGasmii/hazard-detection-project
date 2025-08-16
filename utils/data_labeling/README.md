# Data Labeling Utilities

This folder contains helper scripts and resources for generating and managing annotations for the **hazard detection** project.  

We currently support **two complementary strategies** for labeling:

1. **Class-specific detectors + merger**  
   Train/fine-tune small YOLO/Roboflow models per hazard type, run them on raw data, then **merge predictions** into unified YOLO labels.  
   → Good for *closed-set classes* where you already have strong models.

2. **Open-vocabulary VLM backend (GroundingDINO)**  
   Use a **vision-language model** to detect objects via prompts/aliases (e.g., “hard hat”, “safety helmet”) and automatically map them to our fixed YOLO class list.  
   → Good for *bootstrapping new classes* or rapidly labeling raw/unseen images.

Both methods are combined into an **active learning loop**, where models pre-label data, humans review/correct, and retraining improves performance iteratively.

---

## Labeling strategy

<img width="578" height="425" alt="image" src="https://github.com/user-attachments/assets/3b4d5c5c-bd69-415c-bbdb-32fc0d47b22c" />

### A. Class-specific oracle (previous approach)
- **Oracle** = multiple class-specific detection models (YOLOs or Roboflow exports).
- **Steps:**
  1. Start with a pool of unlabeled images.
  2. Run each class-specific model (helmet, harness, ladder, …).
  3. Merge their predictions into a **unified YOLO TXT dataset**.
  4. Manually review & fix incorrect labels.
  5. Retrain unified model → repeat.

This reduces human effort by letting multiple small models act as the oracle.

---

### B. Open-vocabulary oracle (new approach)
- **Oracle** = GroundingDINO (open-vocab detector).
- **Steps:**
  1. Prompt with class aliases (e.g., `"helmet"`, `"hard hat"`, `"safety helmet"`).
  2. VLM outputs bounding boxes + phrases.
  3. Map aliases → unified YOLO classes (`alias_map`).
  4. Save YOLO `.txt` labels automatically.
  5. Human review as usual.

This allows bootstrapping labels for *all objects at once*.

---

## Current scripts

We provide two complementary labeling oracles:

---

### 1. Class-specific Oracle → `pseudo_label_merger.py`

This script automates **pseudo-labeled datasets** using multiple class-specific models (YOLO/Roboflow):

1. Runs inference on unlabeled images with each hazard-specific model (e.g., helmet, harness, ladder).  
2. Collects predictions from all models.  
3. Merges them into **unified YOLO TXT annotations** following a shared class map.  
4. Saves results into a YOLO-style dataset structure.

**Purpose:**  
- Bootstrap a labeled dataset when you already have specialized detectors.  
- Acts as the **oracle** in the active learning loop.

**Quick run:**
```bash
python pseudo_label_merger.py
```
**Before running, edit the script to set:**

- **`API_KEY`** → your Roboflow private key  
- **`CLASS_MAP`** → dictionary of your unified class IDs  
- **`MODELS`** → list of Roboflow project slugs, version numbers, and classes  
- **`IMAGES_DIR`** → path to unlabeled images  
- **`OUTPUT_DIR`** → where the labeled YOLO dataset will be written  
- **`CONF_THRESHOLD`** → *(optional)* minimum confidence for keeping detections  
---

**Output structure**

```bash
OUTPUT_DIR/
├── images/   # copies of your input images
├── labels/   # YOLO .txt labels (merged)
└── data.yaml # dataset stub for YOLO training (you should set it up)
```
### 2. Open-vocabulary Oracle → `vlm_labeling/autolabel.py`

This script uses an **open-vocabulary detector** (GroundingDINO backend by default) to auto-label images based on text prompts and class aliases.

1. Prompts with a list of **aliases** (e.g., `helmet`, `hard hat`, `safety helmet`).  
2. The VLM outputs bounding boxes + phrases.  
3. Aliases are mapped to a **fixed YOLO class list** via `alias_map`.  
4. Writes YOLO `.txt` labels for each image.  

---

**Quick run:**

**Before starting make sure to install:**
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install opencv-python pillow pyyaml shapely yacs addict timm==0.9.12 transformers==4.40.0 scikit-image matplotlib
pip install pycocotools-windows
```

```bash
python -m utils.data_labeling.vlm_labeling.autolabel --config configs/labeling/fall_hazard_objects.yaml --source utils/data_labeling/datasample/raw_images --out utils/data_labeling/datasample/dino_labels --save-viz

```
**Config file: `configs/labeling/fall_hazard_objects.yaml`**

- Defines the **9 hazard classes** and their aliases.  
- Sets thresholds (`box_thr`, `text_thr`) and backend model paths (`weights_path`, `config_path`).  
- Aliases collapse into one class ID via `alias_map`.  

---

**Output structure**

```bash
datasample/autolabel/dino/
├── labels/
│   ├── img1.txt
│   ├── img2.txt
│   └── ...
└── data.yaml   # dataset stub for YOLO training
```















