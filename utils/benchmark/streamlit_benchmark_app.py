import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

st.set_page_config(page_title="Benchmark Dashboard", layout="wide")

st.title("🧪 Benchmark Dashboard")
st.caption("Load results CSVs, compare models, and explore errors interactively.")

with st.sidebar:
    st.header("Data sources")
    uploaded_results = st.file_uploader(
        "Upload one or more results CSVs",
        type=["csv"], accept_multiple_files=True, key="results_uploader"
    )
    uploaded_errors = st.file_uploader(
        "Upload an errors CSV (optional)", type=["csv"], key="errors_uploader"
    )
    st.markdown("---")
    st.caption("Tip: run `streamlit run` from your repo and use the uploader.")

# ---- Load results ----
frames = []
if uploaded_results:
    for f in uploaded_results:
        try:
            df = pd.read_csv(f)
            df["__source__"] = f.name
            frames.append(df)
        except Exception as e:
            st.warning(f"Failed to read {f.name}: {e}")

if frames:
    results = pd.concat(frames, ignore_index=True)
else:
    st.info("Upload at least one results CSV to begin.")
    st.stop()

if "model" not in results.columns:
    st.error("The results CSV must contain a 'model' column.")
    st.stop()

# Coerce numeric if possible
for c in results.columns:
    if results[c].dtype == "object":
        try:
            results[c] = pd.to_numeric(results[c], errors="ignore")
        except Exception:
            pass

num_cols = results.select_dtypes(include=[np.number]).columns.tolist()
metric_candidates = [c for c in num_cols if c.lower() not in {"epoch", "fold", "seed"}]

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")
    group_by = st.selectbox(
        "Group by",
        options=[c for c in results.columns if c != "__source__"],
        index=(results.columns.tolist().index("model") if "model" in results.columns else 0)
    )
    hue = st.selectbox(
        "Color by (optional)",
        options=["(none)"] + [c for c in results.columns if c not in {"__source__", group_by}],
        index=0
    )
    hue = None if hue == "(none)" else hue

    epoch_col = st.selectbox(
        "Epoch column (optional)",
        options=["(none)"] + results.columns.tolist(),
        index=(results.columns.tolist().index("epoch")+1 if "epoch" in results.columns else 0)
    )
    epoch_col = None if epoch_col == "(none)" else epoch_col

    metrics = st.multiselect(
        "Metrics to visualize",
        options=metric_candidates,
        default=metric_candidates[: min(5, len(metric_candidates))]
    )

    # Simple filtering
    st.subheader("Filter")
    filter_cols = [c for c in results.columns if c not in {"__source__"}]
    sel_col = st.selectbox("Column", options=filter_cols)
    unique_vals = sorted(results[sel_col].dropna().unique().tolist())
    selected_vals = st.multiselect("Keep values", options=unique_vals, default=unique_vals)
    st.caption("Leave defaults to keep everything.")

# Apply filter
if selected_vals is not None and len(unique_vals) != len(selected_vals):
    results = results[results[sel_col].isin(selected_vals)]

st.markdown("### 📊 Results overview")
st.dataframe(results.head(200), use_container_width=True)

# ---- Summary table ----
def build_summary(df, metrics, group_by, hue):
    gcols = [group_by] + ([hue] if hue and hue in df.columns else [])
    agg = {}
    for m in metrics:
        if m in df.columns:
            agg[m] = ["mean", "std", "count"]
    if not agg:
        return pd.DataFrame()
    out = df.groupby(gcols, dropna=False).agg(agg)
    out.columns = ["_".join(col).strip() for col in out.columns.values]
    return out.reset_index()

summary = build_summary(results, metrics, group_by, hue)
if not summary.empty:
    st.markdown("### 🧾 Summary by group")
    st.dataframe(summary, use_container_width=True)

# ---- Charts ----
st.markdown("### 📈 Visualizations")

for m in metrics:
    cols = st.columns(3)

    # Bar (mean + std)
    with cols[0]:
        st.caption(f"Bar • {m}")
        gcols = [group_by] + ([hue] if hue and hue in results.columns else [])
        agg = results.groupby(gcols, dropna=False)[m].agg(["mean", "std", "count"]).reset_index()
        agg["label"] = agg[group_by].astype(str) + ((" | " + agg[hue].astype(str)) if hue and hue in agg.columns else "")
        fig = px.bar(agg, x="label", y="mean", error_y="std",
                     labels={"label": group_by, "mean": m})
        fig.update_layout(xaxis_tickangle=-30, height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Line over epoch
    with cols[1]:
        st.caption(f"Epoch line • {m}")
        if epoch_col and epoch_col in results.columns:
            df_line = results.sort_values(epoch_col)
            color = hue if (hue and hue in df_line.columns) else group_by
            fig2 = px.line(df_line, x=epoch_col, y=m, color=color, markers=True)
            fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Set an epoch column in the sidebar to see learning curves.")

    # Box by group
    with cols[2]:
        st.caption(f"Box • {m}")
        fig3 = px.box(results, x=group_by, y=m,
                      color=hue if hue and hue in results.columns else None,
                      points="all")
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ---- Errors / confusion matrix ----
if uploaded_errors is not None:
    try:
        errors = pd.read_csv(uploaded_errors)
        st.markdown("### ❌ Errors / Confusion Matrix")

        needed = {"model", "true", "pred"}
        if not needed.issubset(errors.columns):
            st.warning(f"Errors CSV must have columns: {needed}.")
        else:
            model_choices = ["(all)"] + sorted(errors["model"].dropna().unique().tolist())
            choose_model = st.selectbox("Select model for confusion", options=model_choices, index=0)
            df_err = errors.copy()
            if choose_model != "(all)":
                df_err = df_err[df_err["model"] == choose_model]

            labels = sorted(set(df_err["true"].unique()).union(set(df_err["pred"].unique())))
            if SKLEARN_OK and len(labels) > 0:
                cm = confusion_matrix(df_err["true"], df_err["pred"], labels=labels)
                fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                                   labels=dict(x="pred", y="true", color="count"))
                fig_cm.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("scikit-learn not available or no labels found; cannot build confusion matrix.")

            # Top confusions
            df_top = (
                df_err[df_err["true"] != df_err["pred"]]
                .groupby(["true", "pred"]).size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.markdown("**Top confusions**")
            st.dataframe(df_top.head(50), use_container_width=True)

    except Exception as e:
        st.warning(f"Could not parse errors CSV: {e}")
else:
    st.caption("Upload an errors CSV to unlock confusion matrix and top confusions.")

st.markdown("---")
st.caption("Pro tip: save a pre-filtered results CSV per run, then drag-and-drop multiple files to compare models, splits, and epochs interactively.")
