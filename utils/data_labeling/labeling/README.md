# Data Labeling Utilities

This folder contains helper scripts and resources for generating and managing annotations for the **hazard detection** project.  
The overall labeling strategy follows an **active learning process**, where the model selectively queries the most informative samples, and an *oracle* provides the labels.

<img width="578" height="425" alt="image" src="https://github.com/user-attachments/assets/3b4d5c5c-bd69-415c-bbdb-32fc0d47b22c" />

## Labeling strategy

In our setup:
- **The labeling oracle** is composed of:
  - Multiple **class‑specific object detection models** trained either locally or on Roboflow (one per hazard type, or multi‑class models where applicable).
  - A **human review step** to validate and correct model predictions before they are added to the training set. Correcting labels is still manual labor but faster, easier and less expensive than annotating the whole dataset.
- The process:
  1. Start with a pool of unlabeled images.
  2. Run each class‑specific model to generate predictions.
  3. Merge predictions into a **unified multi‑class YOLO dataset**.
  4. Manually review and fix incorrect labels.
  5. Retrain the unified model and repeat the loop, **querying only the most valuable samples (the best next samples needed to improve the model will be the new pool)**.

This approach speeds up labeling, reduces human effort, and ensures higher quality data by combining **automated pseudo‑labeling** and **human‑in‑the‑loop correction**.

---

## Current scripts

### `pseudo_label_merger.py`
This script automates the creation of **pseudo‑labeled datasets** by:
1. Running inference on a set of unlabeled images using multiple **class‑specific Roboflow models** (e.g., helmet, harness, ladder).
2. Collecting predictions from each model and merging them into **unified YOLO TXT annotations** according to a shared class mapping.
3. Saving results into a YOLO‑style directory structure:
   
**Purpose:**  
- Bootstrap a labeled dataset using pre‑trained class‑specific models as the oracle.
- Serve as **Step 2 & 3** in the active learning loop, followed by human label review.

**Quick run (default settings):**
```bash
python pseudo_label_merger.py
```
**Customize for your own setup:**

Before running, open the script and update:

- **`API_KEY`** → your Roboflow **Private API key**.  
- **`CLASS_MAP`** → your unified class list with numeric IDs.  
- **`MODELS`** → Roboflow project slugs, version numbers, and associated classes.  
- **`IMAGES_DIR`** → path to your folder of raw unlabeled images.  
- **`OUTPUT_DIR`** → where labeled YOLO data will be saved.  
- **`CONF_THRESHOLD`** → *(optional)* minimum confidence score for keeping predictions.

Once configured, place your raw images in `IMAGES_DIR` and run the script.


## Data sample

The `data_sample` folder contains a small subset of images (`raw_images/`) for testing and debugging.  




