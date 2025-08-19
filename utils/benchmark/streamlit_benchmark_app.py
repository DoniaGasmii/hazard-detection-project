import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

import matplotlib.pyplot as plt
# Convert matplotlib's "Paired" colormap to a list of hex colors
paired = [plt.cm.Paired(i) for i in range(12)]  # 12 distinct colors
paired_hex = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r,g,b,_ in paired]

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

st.set_page_config(page_title="Benchmark Dashboard", layout="wide")
st.title("🧪 Benchmark Dashboard")
st.caption("Upload training logs, test results, and evaluation CSVs. Explore metrics, losses, and errors interactively.")

# ---------------------------
# Sidebar: uploads + settings
# ---------------------------
with st.sidebar:
    st.header("Data sources")

    train_csv = st.file_uploader(
        "Training logs CSV (over epochs)",
        type=["csv"], key="train_csv"
    )
    test_csv = st.file_uploader(
        "Test results CSV (per run/model)",  # aggregate metrics on the test set
        type=["csv"], key="test_csv"
    )
    errors_csv = st.file_uploader(
        "Errors CSV (optional: model,true,pred[,confidence])",
        type=["csv"], key="errors_csv"
    )
    eval_csvs = st.file_uploader(
        "Extra eval CSV(s) (optional, e.g., validation or multiple runs)",
        type=["csv"], accept_multiple_files=True, key="eval_csvs"
    )

    st.markdown("---")
    st.header("Display options")

    palette_name = st.selectbox(
        "Color palette",
        options=["Default", "Paired", "Plotly", "D3", "G10", "T10"],
        index=1  # Paired by default since you requested it
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

    st.caption("Tip: you can upload any subset (only training logs, only eval, etc.).")


# -------------
# Load helpers
# -------------
def read_csv_safe(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception as e:
        st.warning(f"Could not read CSV {getattr(upload, 'name', '(unnamed)')}: {e}")
        return None

train_df = read_csv_safe(train_csv)
test_df  = read_csv_safe(test_csv)
errors_df = read_csv_safe(errors_csv)

eval_frames = []
if eval_csvs:
    for f in eval_csvs:
        df = read_csv_safe(f)
        if df is not None:
            df["__source__"] = f.name
            eval_frames.append(df)
eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else None

# Normalize columns in training logs (your schema example)
# epoch	time	train/box_loss	train/cls_loss	train/dfl_loss	metrics/precision(B)	metrics/recall(B)	metrics/mAP50(B)	metrics/mAP50-95(B)	val/box_loss	val/cls_loss	val/dfl_loss	lr/pg0	lr/pg1	lr/pg2
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

def coerce_numeric_inplace(df):
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

if train_df is not None:
    # Rename only matching columns (keep originals too if useful)
    rename_map = {k: v for k, v in TRAIN_ALIASES.items() if k in train_df.columns}
    train_df = train_df.rename(columns=rename_map)
    coerce_numeric_inplace(train_df)

if test_df is not None:
    coerce_numeric_inplace(test_df)
if eval_df is not None:
    coerce_numeric_inplace(eval_df)
if errors_df is not None:
    coerce_numeric_inplace(errors_df)

# -----------
# UI: 3 tabs
# -----------
tab_train, tab_test, tab_eval = st.tabs(["📚 Training", "🧪 Test", "📈 Eval / Compare"])

# ----------------
# TRAINING TAB
# ----------------
with tab_train:
    st.subheader("Training Logs")
    if train_df is None:
        st.info("Upload a training logs CSV in the sidebar to view this section.")
    else:
        # Show head
        st.caption("Sample rows")
        st.dataframe(train_df.head(200), use_container_width=True)

        # Select which curves to show
        # Losses
        st.markdown("#### Losses over epochs")
        epoch_col = "epoch" if "epoch" in train_df.columns else None
        if epoch_col is None:
            st.warning("No 'epoch' column found in the training logs.")
        else:
            loss_cols = [c for c in ["train_box_loss","train_cls_loss","train_dfl_loss","val_box_loss","val_cls_loss","val_dfl_loss"] if c in train_df.columns]
            chosen_losses = st.multiselect("Loss curves", loss_cols, default=[c for c in loss_cols if "box" in c or "cls" in c] or loss_cols[:2])
            if chosen_losses:
                df_long = train_df[[epoch_col] + chosen_losses].melt(id_vars=epoch_col, var_name="series", value_name="value").sort_values(epoch_col)
                fig_loss = px.line(
                    df_long, x=epoch_col, y="value", color="series",
                    markers=True,
                    color_discrete_sequence=palette
                )
                fig_loss.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info("Select at least one loss curve to display.")

        # Metrics (precision/recall/mAP)
        st.markdown("#### Metrics over epochs")
        metric_cols = [c for c in ["precision_B","recall_B","mAP50_B","mAP50_95_B"] if c in train_df.columns]
        chosen_metrics = st.multiselect("Metric curves", metric_cols, default=metric_cols)
        if epoch_col and chosen_metrics:
            df_long_m = train_df[[epoch_col] + chosen_metrics].melt(id_vars=epoch_col, var_name="metric", value_name="value").sort_values(epoch_col)
            fig_m = px.line(
                df_long_m, x=epoch_col, y="value", color="metric",
                markers=True,
                color_discrete_sequence=palette
            )
            fig_m.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_m, use_container_width=True)
        elif epoch_col and not chosen_metrics:
            st.info("Select at least one metric curve to display.")

        # Learning rates
        st.markdown("#### Learning rates")
        lr_cols = [c for c in ["lr_pg0","lr_pg1","lr_pg2"] if c in train_df.columns]
        if epoch_col and lr_cols:
            df_lr = train_df[[epoch_col] + lr_cols].melt(id_vars=epoch_col, var_name="lr_group", value_name="lr").sort_values(epoch_col)
            fig_lr = px.line(
                df_lr, x=epoch_col, y="lr", color="lr_group",
                markers=True,
                color_discrete_sequence=palette
            )
            fig_lr.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.caption("No LR columns detected (lr_pg0/pg1/pg2).")

# ----------------
# TEST TAB
# ----------------
with tab_test:
    st.subheader("Test Results & Errors")
    if test_df is None and errors_df is None:
        st.info("Upload a test metrics CSV and/or an errors CSV in the sidebar.")
    else:
        if test_df is not None:
            st.markdown("#### Test metrics table")
            # Try to detect common columns
            # Expect at least 'model' and some numeric metrics
            if "model" not in test_df.columns:
                # try to guess a model-like column
                maybe_model = [c for c in test_df.columns if c.lower() in {"model","model_name","run","run_id"}]
                if maybe_model:
                    test_df = test_df.rename(columns={maybe_model[0]: "model"})
            st.dataframe(test_df, use_container_width=True)

            # Show simple bars for top metrics
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            top_metric = st.selectbox("Metric to compare across models", options=numeric_cols or ["(none)"])
            if top_metric != "(none)" and "model" in test_df.columns:
                agg = test_df.groupby("model", dropna=False)[top_metric].agg(["mean","std","count"]).reset_index()
                fig_bar = px.bar(
                    agg, x="model", y="mean", error_y="std", title=None,
                    labels={"mean": top_metric, "model": "model"},
                    color="model",
                    color_discrete_sequence=palette
                )
                fig_bar.update_layout(xaxis_tickangle=-20, height=360, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        # Errors / confusion
        st.markdown("#### Errors analysis")
        if errors_df is None:
            st.caption("Upload an errors CSV to unlock confusion matrix and top confusions.")
        else:
            needed = {"model","true","pred"}
            if not needed.issubset(errors_df.columns):
                st.warning(f"Errors CSV must have columns: {needed}.")
            else:
                # Choose model or all
                model_choices = ["(all)"] + sorted(errors_df["model"].dropna().unique().tolist())
                choose_model = st.selectbox("Select model for confusion", options=model_choices, index=0, key="test_err_model")
                df_err = errors_df.copy()
                if choose_model != "(all)":
                    df_err = df_err[df_err["model"] == choose_model]

                labels = sorted(set(df_err["true"].unique()).union(set(df_err["pred"].unique())))
                if SKLEARN_OK and len(labels) > 0:
                    cm = confusion_matrix(df_err["true"], df_err["pred"], labels=labels)
                    fig_cm = px.imshow(
                        cm, x=labels, y=labels, text_auto=True,
                        labels=dict(x="pred", y="true", color="count"),
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(height=500, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.info("scikit-learn not available or no labels found; cannot build confusion matrix.")

                # Top confusions
                df_top = (
                    df_err[df_err["true"] != df_err["pred"]]
                    .groupby(["true","pred"]).size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                st.markdown("**Top confusions**")
                st.dataframe(df_top.head(50), use_container_width=True)

# ----------------
# EVAL TAB
# ----------------
with tab_eval:
    st.subheader("Evaluation / Comparison")
    if eval_df is None:
        st.info("Upload one or more eval CSVs (e.g., validation results, multiple runs).")
    else:
        st.caption("Combined view of all uploaded eval CSVs.")
        st.dataframe(eval_df.head(200), use_container_width=True)

        # Choose grouping axes
        all_cols = [c for c in eval_df.columns if c not in {"__source__"}]
        # Prefer 'model' if present
        group_by = st.selectbox("Group by", options=all_cols, index=(all_cols.index("model") if "model" in all_cols else 0))
        hue = st.selectbox("Color by (optional)", options=["(none)"] + [c for c in all_cols if c != group_by], index=0)
        hue = None if hue == "(none)" else hue

        # Pick metrics
        num_cols = eval_df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = st.multiselect("Metrics", options=num_cols, default=num_cols[: min(5, len(num_cols))])

        # Quick filter
        st.markdown("**Filter**")
        filt_col = st.selectbox("Column", options=all_cols, key="eval_filter_col")
        unique_vals = sorted(eval_df[filt_col].dropna().unique().tolist())
        selected_vals = st.multiselect("Keep values", options=unique_vals, default=unique_vals, key="eval_filter_vals")

        dff = eval_df.copy()
        if selected_vals and len(selected_vals) != len(unique_vals):
            dff = dff[dff[filt_col].isin(selected_vals)]

        # Summary
        def build_summary(df, metrics, group_by, hue):
            gcols = [group_by] + ([hue] if hue and hue in df.columns else [])
            agg = {m: ["mean","std","count"] for m in metrics if m in df.columns}
            if not agg:
                return pd.DataFrame()
            out = df.groupby(gcols, dropna=False).agg(agg)
            out.columns = ["_".join(col).strip() for col in out.columns.values]
            return out.reset_index()

        summary = build_summary(dff, metrics, group_by, hue)
        if not summary.empty:
            st.markdown("#### Summary by group")
            st.dataframe(summary, use_container_width=True)

        # Charts
        st.markdown("#### Visualizations")
        for m in metrics:
            cols = st.columns(2)

            with cols[0]:
                st.caption(f"Bar • {m}")
                gcols = [group_by] + ([hue] if hue and hue in dff.columns else [])
                agg = dff.groupby(gcols, dropna=False)[m].agg(["mean","std","count"]).reset_index()
                # Build a label column if hue present
                if hue and hue in agg.columns:
                    agg["label"] = agg[group_by].astype(str) + " | " + agg[hue].astype(str)
                    xcol = "label"
                else:
                    xcol = group_by
                fig = px.bar(
                    agg, x=xcol, y="mean", error_y="std",
                    labels={"mean": m, xcol: xcol},
                    color=hue if hue and hue in agg.columns else group_by,
                    color_discrete_sequence=palette
                )
                fig.update_layout(xaxis_tickangle=-25, height=360, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            with cols[1]:
                st.caption(f"Box • {m}")
                fig2 = px.box(
                    dff, x=group_by, y=m,
                    color=hue if hue and hue in dff.columns else None,
                    points="all",
                    color_discrete_sequence=palette
                )
                fig2.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Palette set to '{}' (Plotly qualitative). Adjust in the sidebar.".format(palette_name))
