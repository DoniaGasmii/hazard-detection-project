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

We provide **two families of labeling oracles**:

---

### 1. Class-specific Oracle

#### a) Per-class Roboflow scripts → `class_oracles/`
Each hazard type (helmet, harness, ladder, …) has its **own script** that:
1. Loads the corresponding Roboflow model.  
2. Runs inference on raw images.  
3. Writes YOLO `.txt` labels for just that class.  

**Usage example:**
```bash
python class_oracles/helmet_oracle.py
python class_oracles/harness_oracle.py
```
This makes debugging easier (e.g., checking why *scaffolding* is often missed) and allows selective re-runs when only one class needs updating.

**Output**
Labels are saved into per-class folders under `outputs/`, ready for review or merging.

### b) Multi-class merger → `pseudo_label_merger.py`
This script merges the outputs from all class-specific scripts (or directly queries Roboflow projects, depending on config).

1. Runs inference on unlabeled images with each hazard-specific model.  
2. Collects predictions from all models.  
3. Merges them into **unified YOLO TXT annotations** following a shared class map.  
4. Saves results into a YOLO-style dataset structure.  

### Quick run
```bash
python pseudo_label_merger.py
```
**Before running, edit the script to set:**
- `API_KEY` → your Roboflow private key  
- `CLASS_MAP` → dictionary of unified class IDs  
- `MODELS` → list of Roboflow project slugs, version numbers, and classes  
- `IMAGES_DIR` / `OUTPUT_DIR`  

**Output structure:**
```bash
OUTPUT_DIR/
├── images/
├── labels/
└── data.yaml
---



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
python -m utils.data_labeling.vlm_labeling.autolabel --config configs/labeling/fall_hazard_objects.yaml --source utils/data_labeling/data_sample/raw_images --out utils/data_labeling/data_sample/dino_labels --save
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




















