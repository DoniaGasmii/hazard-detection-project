import os
import io
import zipfile
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
import torch
import cv2
from ultralytics import YOLO

# ---- Optional: Paired palette via matplotlib
import matplotlib.pyplot as plt
paired = [plt.cm.Paired(i) for i in range(12)]
paired_hex = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r,g,b,_ in paired]

# =========================
# App header + sidebar I/O
# =========================
st.set_page_config(page_title="Benchmark Dashboard", layout="wide")
st.title("Benchmark Dashboard")
st.caption("Upload training logs, test eval artifacts, or comparison CSVs. Explore metrics, losses, and errors interactively.")

# ---- Sidebar: section + uploaders + palette
with st.sidebar:
    section = st.radio("Section", ["Training", "Test", "Compare", "Inference"], index=1)

    if section == "Training":
        train_csv = st.file_uploader("Training logs CSV", type=["csv"], key="train_csv")

    elif section == "Test":
        st.markdown("**Ultralytics eval outputs**")
        eval_zip = st.file_uploader("Eval folder (ZIP)", type=["zip"], key="eval_zip")
        metrics_csv = st.file_uploader("Metrics CSV (*_metrics_val.csv)", type=["csv"], key="metrics_csv")
        per_class_csv = st.file_uploader("Per-class mAP CSV (*_per_class_map_val.csv)", type=["csv"], key="perclass_csv")
        cm_csv = st.file_uploader("Confusion Matrix CSV (*_confusion_matrix_val.csv)", type=["csv"], key="cm_csv")

    elif section == "Compare":
        comp_csvs = st.file_uploader(
            "Upload CSVs to compare (must include `model`)", type=["csv"], accept_multiple_files=True, key="compare_csvs"
        )

    elif section == "Inference":
        inf_image = st.file_uploader("Image", type=["png","jpg","jpeg"], key="inf_img")
        weight_files = st.file_uploader("YOLOv8 weights (one or more .pt files)", type=["pt"], accept_multiple_files=True, key="inf_wts")
        conf_th = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
        iou_th  = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
        imgsz   = st.number_input("Image size (0 = model default)", min_value=0, max_value=2048, value=0, step=32)


    st.markdown("---")
    st.header("Display options")
    palette_name = st.selectbox(
        "Color palette",
        options=["Default", "Paired", "Plotly", "D3", "G10", "T10"],
        index=1
    )
    palette_map = {
        "Default": None,
        "Paired": paired_hex,
        "Plotly": px.colors.qualitative.Plotly,
        "D3": px.colors.qualitative.D3,
        "G10": px.colors.qualitative.G10,
        "T10": px.colors.qualitative.T10,
    }
    palette = palette_map[palette_name]
    st.caption("Tip: upload only what you need per section. The controls update by section.")

# =========================
# Helpers
# =========================
TRAIN_ALIASES = {
    "epoch": "epoch",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "metrics/precision(B)": "precision_B",
    "metrics/recall(B)": "recall_B",
    "metrics/mAP50(B)": "mAP50_B",
    "metrics/mAP50-95(B)": "mAP50_95_B",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr_pg0",
    "lr/pg1": "lr_pg1",
    "lr/pg2": "lr_pg2",
    "time": "time",
}

PRETTY = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
}

EXPLAIN = {
    "BoxPR_curve": "Precision–Recall curve; area relates to AP.",
    "BoxP_curve":  "Precision vs confidence threshold; helps pick a threshold.",
    "BoxR_curve":  "Recall vs confidence threshold.",
    "BoxF1_curve": "F1 vs confidence; peak ≈ a good operating point.",
    "confusion_matrix": "Confusion matrix (counts). Rows=true class, cols=pred.",
    "confusion_matrix_normalized": "Confusion matrix normalized per true class.",
    "val_batch": "Sample batch: *_labels are GT, *_pred are predictions.",
}

def coerce_numeric_inplace(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def explain_from_name(name: str):
    base = os.path.splitext(os.path.basename(name))[0]
    for k, v in EXPLAIN.items():
        if k in base:
            return base, v
    return base, "Evaluation artifact produced by Ultralytics."

def load_eval_zip(up_zip):
    """Return a dict with images list and optional CSV dataframes found inside the ZIP."""
    out = {"images": [], "metrics": None, "per_class": None, "cm": None}
    try:
        z = zipfile.ZipFile(up_zip)
        for zi in z.infolist():
            lname = zi.filename.lower()
            if lname.endswith((".png", ".jpg", ".jpeg")):
                with z.open(zi) as f:
                    b = f.read()
                label, desc = explain_from_name(zi.filename)
                out["images"].append((zi.filename, b, desc))
            elif lname.endswith("_metrics_val.csv"):
                out["metrics"] = pd.read_csv(z.open(zi))
            elif lname.endswith("_per_class_map_val.csv"):
                out["per_class"] = pd.read_csv(z.open(zi))
            elif lname.endswith("_confusion_matrix_val.csv"):
                out["cm"] = pd.read_csv(z.open(zi), index_col=0)
    except Exception as e:
        st.error(f"Could not read ZIP: {e}")
    return out

def normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={k: v for k, v in PRETTY.items() if k in df.columns})
    if "model" not in df.columns:
        df["model"] = df.get("run", "model")
    return df

# ---------- Drawing helpers ----------
def draw_boxes_on_image(img_bgr, boxes, labels, color=(0, 255, 0), thickness=2):
    out = img_bgr.copy()
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if lab:
            cv2.putText(out, str(lab), (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def to_bgr(np_img):
    # Streamlit returns RGB; OpenCV uses BGR
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img

def to_rgb(np_img):
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        return cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    return np_img

# ---------- Feature map capture ----------
class FeatureHook:
    def __init__(self, module):
        self.feat = None
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, inp, out):
        # out: [B, C, H, W]
        with torch.no_grad():
            fmap = out.detach().float()
            fmap = fmap.mean(dim=1, keepdim=True)  # average over channels
            self.feat = fmap  # store
    def close(self):
        self.hook.remove()

def list_named_layers(model: YOLO):
    # Return list of (name, module) for the underlying nn.Module
    layers = []
    for name, m in model.model.named_modules():
        # skip the root '' module
        if name != "" and hasattr(m, "forward"):
            layers.append((name, m))
    return layers

def compute_featuremap_overlay(model: YOLO, img_bgr, layer_name: str, conf=0.25, iou=0.45, imgsz=0):
    """
    Runs a forward pass with a hook on `layer_name`, returns (overlay_rgb, raw_heatmap_gray_norm)
    """
    model.model.eval()
    named = dict(list_named_layers(model))
    if layer_name not in named:
        raise ValueError(f"Layer '{layer_name}' not found. Pick one of: {list(named.keys())[:8]} ...")

    # register hook
    fh = FeatureHook(named[layer_name])

    # Ultralytics prediction (no saving to disk); single image forward
    # Note: stream=False returns a list of Results
    res = model.predict(source=[img_bgr[..., ::-1]], conf=conf, iou=iou, imgsz=imgsz or None, verbose=False)
    # grab the feature map captured
    fmap = fh.feat  # [1,1,h,w]
    fh.close()
    if fmap is None:
        raise RuntimeError("Hook captured no feature map (unexpected).")

    fmap = fmap[0, 0].cpu().numpy()
    # normalize 0..1
    fm_min, fm_max = float(fmap.min()), float(fmap.max()) if float(fmap.max()) != 0 else 1.0
    fmap_norm = (fmap - fm_min) / (fm_max - fm_min + 1e-8)

    # resize to image size
    H, W = img_bgr.shape[:2]
    fmap_up = cv2.resize(fmap_norm, (W, H), interpolation=cv2.INTER_CUBIC)

    # colorize heatmap
    heat_u8 = np.uint8(255 * fmap_up)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR heat
    overlay = cv2.addWeighted(img_bgr, 0.55, heat_color, 0.45, 0)

    # also return predictions (boxes + labels) for convenience
    return overlay, fmap_up, res


# =========================
# TRAINING
# =========================
if section == "Training":
    st.header("📚 Training")
    if "train_csv" not in locals() or train_csv is None:
        st.info("Upload a training logs CSV in the sidebar.")
    else:
        train_df = pd.read_csv(train_csv)
        rename_map = {k: v for k, v in TRAIN_ALIASES.items() if k in train_df.columns}
        train_df = train_df.rename(columns=rename_map)
        coerce_numeric_inplace(train_df)

        st.caption("Sample rows")
        st.dataframe(train_df.head(200), use_container_width=True)

        epoch_col = "epoch" if "epoch" in train_df.columns else None

        st.markdown("#### Losses over epochs")
        if not epoch_col:
            st.warning("No 'epoch' column found.")
        else:
            loss_cols = [c for c in ["train_box_loss","train_cls_loss","train_dfl_loss","val_box_loss","val_cls_loss","val_dfl_loss"] if c in train_df.columns]
            chosen_losses = st.multiselect("Loss curves", loss_cols, default=[c for c in loss_cols if "box" in c or "cls" in c] or loss_cols[:2])
            if chosen_losses:
                df_long = train_df[[epoch_col] + chosen_losses].melt(id_vars=epoch_col, var_name="series", value_name="value").sort_values(epoch_col)
                fig_loss = px.line(df_long, x=epoch_col, y="value", color="series", markers=True, color_discrete_sequence=palette)
                fig_loss.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info("Select at least one loss curve to display.")

        st.markdown("#### Metrics over epochs")
        metric_cols = [c for c in ["precision_B","recall_B","mAP50_B","mAP50_95_B"] if c in train_df.columns]
        chosen_metrics = st.multiselect("Metric curves", metric_cols, default=metric_cols)
        if epoch_col and chosen_metrics:
            df_long_m = train_df[[epoch_col] + chosen_metrics].melt(id_vars=epoch_col, var_name="metric", value_name="value").sort_values(epoch_col)
            fig_m = px.line(df_long_m, x=epoch_col, y="value", color="metric", markers=True, color_discrete_sequence=palette)
            fig_m.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("#### Learning rates")
        lr_cols = [c for c in ["lr_pg0","lr_pg1","lr_pg2"] if c in train_df.columns]
        if epoch_col and lr_cols:
            df_lr = train_df[[epoch_col] + lr_cols].melt(id_vars=epoch_col, var_name="lr_group", value_name="lr").sort_values(epoch_col)
            fig_lr = px.line(df_lr, x=epoch_col, y="lr", color="lr_group", markers=True, color_discrete_sequence=palette)
            fig_lr.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.caption("No LR columns detected (lr_pg0/pg1/pg2).")

# =========================
# TEST (images + CSVs)
# =========================
elif section == "Test":
    st.header("🧪 Test")

    # A) Load from ZIP if provided (images + any CSVs inside)
    zip_payload = load_eval_zip(eval_zip) if "eval_zip" in locals() and eval_zip is not None else None

    # ---------- Images ----------
    st.subheader("Eval images")
    imgs = zip_payload["images"] if (zip_payload and zip_payload["images"]) else []
    if not imgs:
        st.caption("Upload your eval folder as a ZIP in the sidebar to preview PR/F1/CM/batch images.")
    else:
        imgs.sort(key=lambda x: x[0])
        names = [n for (n, _, _) in imgs]
        default = [n for n in names if any(k in n for k in ["BoxPR_curve","BoxF1_curve","confusion_matrix"])]
        chosen = st.multiselect("Choose images to render", options=names, default=default or names)
        for name, b, desc in imgs:
            if name in chosen:
                st.markdown(f"**{os.path.basename(name)}** · _{desc}_")
                st.image(b, use_container_width =True)

    # ---------- Headline metrics (bar) ----------
    st.subheader("Headline metrics")
    df_metrics = None
    if zip_payload and zip_payload["metrics"] is not None:
        df_metrics = zip_payload["metrics"]
    elif "metrics_csv" in locals() and metrics_csv is not None:
        df_metrics = pd.read_csv(metrics_csv)

    if df_metrics is None:
        st.caption("Upload *_metrics_val.csv in the sidebar to plot precision/recall/mAP.")
    else:
        df_metrics = normalize_metrics_df(df_metrics)
        st.dataframe(df_metrics, use_container_width=True)
        metric_cols = [c for c in ["precision","recall","mAP50","mAP50_95"] if c in df_metrics.columns]
        if metric_cols:
            long = df_metrics.melt(
                id_vars=[c for c in df_metrics.columns if c not in metric_cols],
                value_vars=metric_cols, var_name="metric", value_name="value"
            )
            fig = px.bar(long, x="metric", y="value", color="model", barmode="group", color_discrete_sequence=palette)
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**What this shows:** the main performance numbers for this model run.")
        else:
            st.warning("No standard metric columns found. Expected one of: precision, recall, mAP50, mAP50_95.")

    # ---------- Per-class mAP ----------
    st.subheader("Per-class mAP@0.5:0.95")
    df_pc = None
    if zip_payload and zip_payload["per_class"] is not None:
        df_pc = zip_payload["per_class"]
    elif "per_class_csv" in locals() and per_class_csv is not None:
        df_pc = pd.read_csv(per_class_csv)

    if df_pc is None:
        st.caption("Upload *_per_class_map_val.csv to see class-wise bars.")
    else:
        st.dataframe(df_pc, use_container_width=True)
        if {"class_name","mAP50_95"}.issubset(df_pc.columns):
            fig = px.bar(df_pc.sort_values("mAP50_95", ascending=False), x="class_name", y="mAP50_95", color_discrete_sequence=palette)
            fig.update_layout(height=380, xaxis_tickangle=-30, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**What this shows:** class-wise quality; look for very low bars to find weak classes.")

    # ---------- Confusion matrix ----------
    st.subheader("Confusion matrix (counts)")
    df_cm = None
    if zip_payload and zip_payload["cm"] is not None:
        df_cm = zip_payload["cm"]
    elif "cm_csv" in locals() and cm_csv is not None:
        df_cm = pd.read_csv(cm_csv, index_col=0)

    if df_cm is None:
        st.caption("Upload *_confusion_matrix_val.csv to render the heatmap.")
    else:
        fig_cm = px.imshow(df_cm.values, x=list(df_cm.columns), y=list(df_cm.index), text_auto=True,
                           labels=dict(x="pred", y="true", color="count"))
        fig_cm.update_layout(height=500, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("**How to read:** bright diagonal = correct; bright off‑diagonal cells = common confusions.")

# =========================
# COMPARE (multi‑CSV)
# =========================
elif section == "Compare":
    st.header("📈 Compare models")
    if "comp_csvs" not in locals() or not comp_csvs:
        st.info("Upload one or more CSVs with a `model` column and metrics (e.g., precision/recall/mAP).")
    else:
        frames = []
        for f in comp_csvs:
            try:
                d = pd.read_csv(f)
                d["__source__"] = f.name
                frames.append(d)
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")
        comp = pd.concat(frames, ignore_index=True)
        comp = normalize_metrics_df(comp)
        coerce_numeric_inplace(comp)

        if "model" not in comp.columns:
            st.error("No `model` column found. Please add it to your CSVs.")
        else:
            st.dataframe(comp, use_container_width=True)

            num_cols = comp.select_dtypes(include=[np.number]).columns.tolist()
            default_metrics = [m for m in ["precision","recall","mAP50","mAP50_95"] if m in num_cols]
            metrics = st.multiselect("Pick metrics to compare", options=num_cols, default=default_metrics or num_cols[:3])

            hue = st.selectbox("Color by (optional)", options=["(none)","__source__"], index=0)
            hue = None if hue == "(none)" else hue

            for m in metrics:
                st.markdown(f"#### {m} — comparison")
                cols = st.columns(2)

                with cols[0]:
                    st.caption("Bar (mean±std if repeats)")
                    agg = comp.groupby(["model"] + ([hue] if hue else []), dropna=False)[m].agg(["mean","std","count"]).reset_index()
                    xlab = "label"
                    if hue:
                        agg[xlab] = agg["model"].astype(str) + " | " + agg[hue].astype(str)
                        color = hue
                    else:
                        agg[xlab] = agg["model"].astype(str)
                        color = "model"
                    fig = px.bar(agg, x=xlab, y="mean", error_y="std", color=color, color_discrete_sequence=palette)
                    fig.update_layout(xaxis_tickangle=-25, height=360, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Shows relative performance per model (error bars = std across runs/seeds).")

                with cols[1]:
                    st.caption("Box (variability)")
                    fig2 = px.box(comp, x="model", y=m, color=hue if hue else None, points="all", color_discrete_sequence=palette)
                    fig2.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("Shows spread across runs/folds. Wide boxes or many outliers = unstable.")

# =========================
# COMPARE (multi‑CSV)
# =========================

elif section == "Inference":
    st.header("🔍 Inference (boxes + feature map)")
    if inf_image is None or not weight_files:
        st.info("Upload an image and at least one YOLOv8 weights file in the sidebar.")
    else:
        # Load image
        pil = Image.open(inf_image).convert("RGB")
        base_rgb = np.array(pil)
        base_bgr = to_bgr(base_rgb)

        # Let user pick which model to visualize a feature map from
        # Load all models first (cached per filename)
        @st.cache_resource
        def load_model_from_bytes(name, bytes_obj):
            tmp_path = os.path.join(st.session_state.get("_tmp_dir_", "/tmp"), name)
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(bytes_obj.getbuffer())
            return YOLO(tmp_path)

        models = []
        for wf in weight_files:
            try:
                mdl = load_model_from_bytes(wf.name, wf)
                models.append((wf.name, mdl))
            except Exception as e:
                st.error(f"Failed to load {wf.name}: {e}")

        if not models:
            st.stop()

        # Choose model for featuremap visualization (predictions will be shown for all)
        model_names = [n for (n, _) in models]
        fm_model_name = st.selectbox("Choose model for feature‑map", options=model_names, index=0)

        # List layers
        fm_model = dict(models)[fm_model_name]
        layer_list = [n for (n, m) in list_named_layers(fm_model)]
        # Heuristic: pick a mid‑neck layer by default
        default_idx = 0
        for i, n in enumerate(layer_list):
            if any(k in n.lower() for k in ["neck", "c2f", "spp", "sppe", "stage"]):
                default_idx = i
                break
        layer_name = st.selectbox("Layer for feature‑map", options=layer_list, index=default_idx)

        # Run buttons
        run = st.button("Run inference")

        if run:
            # 1) Feature‑map overlay for the selected model
            with st.spinner("Running inference + capturing feature map..."):
                try:
                    overlay_bgr, fmap_gray, res = compute_featuremap_overlay(
                        fm_model, base_bgr, layer_name, conf=conf_th, iou=iou_th, imgsz=imgsz
                    )
                except Exception as e:
                    st.error(f"Feature‑map failed: {e}")
                    overlay_bgr, fmap_gray, res = base_bgr, None, []

            # 2) Draw boxes for each model (separate panels)
            st.markdown("### Predictions")
            cols = st.columns(len(models))
            for i, (name, mdl) in enumerate(models):
                with cols[i]:
                    results = mdl.predict(source=[base_rgb], conf=conf_th, iou=iou_th, imgsz=imgsz or None, verbose=False)
                    r = results[0]
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.cls is not None else []
                    names = r.names if hasattr(r, "names") else {}
                    labels = [names.get(int(c), str(int(c))) for c in cls_ids] if isinstance(names, dict) else [str(int(c)) for c in cls_ids]
                    boxed = draw_boxes_on_image(base_bgr, boxes_xyxy, labels, color=(0, 255, 0))
                    st.image(to_rgb(boxed), caption=f"{name} — {len(boxes_xyxy)} boxes", use_container_width =True)

            # 3) Feature‑map panel
            st.markdown("### Feature‑map (avg across channels)")
            st.image(to_rgb(overlay_bgr), caption=f"{fm_model_name} · layer: {layer_name}", use_container_width =True)
            st.caption("Tip: try different layers (backbone vs neck vs head) to see how activation shifts.")


st.markdown("---")
st.caption(f"Palette set to **{palette_name}**. Change it in the sidebar.")


