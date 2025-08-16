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

This allows bootstrapping labels for *all nine hazard classes at once*.

---

## Current scripts

### `pseudo_label_merger.py`
Automates **pseudo-labeled datasets** using class-specific models:
1. Runs inference on unlabeled images with multiple Roboflow models.
2. Collects predictions and merges them into **unified YOLO TXT annotations**.
3. Saves results into a YOLO-style dataset.

**Purpose:**  
- Bootstrap labeled data from multiple weak or specialized models.  
- Acts as **Step 2 & 3** in the active learning loop.

**Quick run:**
```bash
python pseudo_label_merger.py
