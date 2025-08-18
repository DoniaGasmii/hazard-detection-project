#!/usr/bin/env python3
"""
benchmark_viz.py
----------------
A compact CLI to:
- Load one or more CSVs of model results (per run/fold/epoch)
- Summarize and compare metrics across models
- Plot key charts (bar, line-over-epoch, box-over-folds)
- (Optional) Analyze errors and build a confusion matrix if you pass an errors CSV.

Expected columns in results CSV (flexible):
- Required: model
- Optional: split, epoch, fold
- Metrics: any numeric columns like accuracy, f1, precision, recall, mAP50, mAP50_95, auc_roc, auc_pr, loss, etc.

Expected columns in errors CSV (optional):
- Required: model, true, pred
- Optional: confidence (float in [0,1])

Usage examples:
    python benchmark_viz.py --results results.csv --output-dir out/
    python benchmark_viz.py --results r1.csv r2.csv --output-dir out/ --metrics accuracy f1 mAP50
    python benchmark_viz.py --results results.csv --errors errors.csv --labels helmet harness background --output-dir out/
    python benchmark_viz.py --results results.csv --group-by model --hue split --epoch-col epoch --metrics f1 auc_pr --output-dir out/
"""
import argparse
import os
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def _guess_metric_cols(df: pd.DataFrame) -> List[str]:
    # Heuristic: numeric cols that look like metrics (exclude epoch/fold and ids)
    exclude = {'epoch','fold','seed'}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c.lower() not in exclude]

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def plot_bars_by_group(df: pd.DataFrame, metric: str, group: str, hue: Optional[str], out_dir: str):
    gcols = [group] + ([hue] if hue and hue in df.columns else [])
    agg = df.groupby(gcols, dropna=False)[metric].agg(['mean','std','count']).reset_index()
    # Build labels and bars
    labels = []
    means = []
    yerr = []
    for _, row in agg.iterrows():
        if len(gcols) == 2:
            label = f"{row[group]} | {row[hue]}"
        else:
            label = f"{row[group]}"
        labels.append(str(label))
        means.append(row['mean'])
        yerr.append(row['std'] if row['count']>1 else 0.0)

    plt.figure(figsize=(max(6, len(labels)*0.6), 4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=yerr, capsize=3)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f"{metric} by {group}" + (f" and {hue}" if hue and hue in df.columns else ""))
    plt.tight_layout()
    fname = os.path.join(out_dir, f"bar_{metric}_by_{group}" + (f"_and_{hue}" if hue and hue in df.columns else "") + ".png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

def plot_lines_over_epoch(df: pd.DataFrame, metric: str, group: str, epoch_col: str, out_dir: str, hue: Optional[str]):
    if epoch_col not in df.columns:
        return None
    # One line per group (optionally per hue)
    plt.figure(figsize=(7,4))
    gcols = [group] + ([hue] if hue and hue in df.columns else [])
    for keys, gdf in df.sort_values(epoch_col).groupby(gcols):
        label = " | ".join([str(k) for k in (keys if isinstance(keys, tuple) else (keys,)) if pd.notna(k)])
        plt.plot(gdf[epoch_col], gdf[metric], marker='o', label=label)
    plt.xlabel(epoch_col)
    plt.ylabel(metric)
    plt.title(f"{metric} over {epoch_col}")
    if len(df[group].unique()) > 1 or (hue and hue in df.columns):
        plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"line_{metric}_over_{epoch_col}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

def plot_box_by_fold(df: pd.DataFrame, metric: str, group: str, out_dir: str):
    if 'fold' not in df.columns:
        return None
    # Build data per group
    data = []
    labels = []
    for name, gdf in df.groupby(group):
        vals = gdf[metric].dropna().values.tolist()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(str(name))
    if not data:
        return None
    plt.figure(figsize=(max(6, len(labels)*0.6), 4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(metric)
    plt.title(f"{metric} distribution by {group} (folds)")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"box_{metric}_by_{group}_folds.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

def summarize_table(df: pd.DataFrame, metrics: List[str], group: str, hue: Optional[str]) -> pd.DataFrame:
    gcols = [group] + ([hue] if hue and hue in df.columns else [])
    agg_map = {m:['mean','std','count'] for m in metrics if m in df.columns}
    if not agg_map:
        return pd.DataFrame()
    out = df.groupby(gcols, dropna=False).agg(agg_map)
    # flatten columns
    out.columns = ['_'.join(col).strip() for col in out.columns.values]
    return out.reset_index()

def analyze_errors(errors_df: pd.DataFrame, labels: Optional[List[str]], out_dir: str, model_filter: Optional[str]=None):
    if not SKLEARN_OK:
        print("[warn] scikit-learn not available; skipping confusion matrix.")
        return None, None

    df = errors_df.copy()
    if model_filter is not None:
        df = df[df['model']==model_filter]
        if df.empty:
            print(f"[warn] No errors for model={model_filter}")
            return None, None

    # Use unique labels if not provided
    if not labels:
        labels = sorted(set(df['true'].unique()).union(set(df['pred'].unique())))
    cm = confusion_matrix(df['true'], df['pred'], labels=labels, normalize=None)
    plt.figure(figsize=(max(6, len(labels)*0.6), max(5, len(labels)*0.6)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, xticks_rotation=45, cmap=None)  # default colormap; do not force colors
    plt.title("Confusion Matrix" + (f" – {model_filter}" if model_filter else ""))
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix" + (f"_{model_filter}" if model_filter else "") + ".png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Top confusions table
    df_err = df[df['true'] != df['pred']].copy()
    top_pairs = (df_err.groupby(['true','pred']).size()
                 .sort_values(ascending=False)
                 .reset_index(name='count'))
    top_path = os.path.join(out_dir, f"top_confusions" + (f"_{model_filter}" if model_filter else "") + ".csv")
    top_pairs.to_csv(top_path, index=False)
    return cm_path, top_path

def main():
    ap = argparse.ArgumentParser(description="Visualize and compare benchmark CSVs.")
    ap.add_argument('--results', nargs='+', required=True, help='Path(s) to results CSV(s).')
    ap.add_argument('--errors', type=str, default=None, help='Optional path to errors CSV with columns: model,true,pred[,confidence].')
    ap.add_argument('--labels', nargs='*', default=None, help='Optional ordered class labels for confusion matrix.')
    ap.add_argument('--output-dir', type=str, required=True, help='Directory to save charts/tables.')
    ap.add_argument('--metrics', nargs='*', default=None, help='Metrics to visualize; if omitted, auto-detect numeric columns.')
    ap.add_argument('--group-by', type=str, default='model', help='Column to group bars/boxes by (default: model).')
    ap.add_argument('--hue', type=str, default=None, help='Optional secondary grouping column (e.g., split).')
    ap.add_argument('--epoch-col', type=str, default='epoch', help='Epoch column name if present (default: epoch).')
    ap.add_argument('--filter', type=str, default=None, help='Optional pandas query string to filter rows, e.g., "split==\'test\'".')
    ap.add_argument('--show', action='store_true', help='Also display plots interactively.')
    args = ap.parse_args()

    _safe_mkdir(args.output_dir)

    # Load and concatenate all result CSVs
    frames = []
    for p in args.results:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df['__source__'] = os.path.basename(p)
        frames.append(df)
    results = pd.concat(frames, ignore_index=True)
    if args.filter:
        try:
            results = results.query(args.filter)
        except Exception as e:
            print(f"[warn] Failed to apply filter '{args.filter}': {e}")

    # Determine metrics
    if args.metrics:
        metrics = [m for m in args.metrics if m in results.columns]
        if not metrics:
            print("[warn] None of the requested metrics are in the CSV; falling back to auto-detect.")
            metrics = _guess_metric_cols(results)
    else:
        metrics = _guess_metric_cols(results)

    if not metrics:
        print("[error] Could not find numeric metrics to visualize.")
        return

    # Coerce metric columns numeric
    results = _coerce_numeric(results, metrics)

    # Save a summary table
    summary = summarize_table(results, metrics, group=args.group_by, hue=args.hue)
    if not summary.empty:
        summary_path = os.path.join(args.output_dir, "summary_by_group.csv")
        summary.to_csv(summary_path, index=False)
        print(f"[ok] Saved summary: {summary_path}")

    # Plot per-metric comparisons
    made = []
    for m in metrics:
        try:
            p1 = plot_bars_by_group(results, m, group=args.group_by, hue=args.hue, out_dir=args.output_dir)
            if p1: made.append(p1)
        except Exception as e:
            print(f"[warn] bar plot for {m} failed: {e}")

        try:
            p2 = plot_lines_over_epoch(results, m, group=args.group_by, epoch_col=args.epoch_col, out_dir=args.output_dir, hue=args.hue)
            if p2: made.append(p2)
        except Exception as e:
            print(f"[warn] line plot for {m} failed: {e}")

        try:
            p3 = plot_box_by_fold(results, m, group=args.group_by, out_dir=args.output_dir)
            if p3: made.append(p3)
        except Exception as e:
            print(f"[warn] box plot for {m} failed: {e}")

    # Errors / confusion matrix (optional)
    if args.errors:
        if not os.path.isfile(args.errors):
            print(f"[warn] errors file not found: {args.errors}")
        else:
            err_df = pd.read_csv(args.errors)
            needed = {'model','true','pred'}
            if not needed.issubset(set(err_df.columns)):
                print(f"[warn] errors CSV must have columns: {needed}")
            else:
                # If multiple models exist, create one confusion per model and also a combined one
                models = sorted(err_df['model'].dropna().unique())
                if len(models) > 1:
                    for m in models:
                        cm_path, top_path = analyze_errors(err_df, labels=args.labels, out_dir=args.output_dir, model_filter=m)
                        if cm_path: print(f"[ok] Saved confusion matrix for {m}: {cm_path}")
                        if top_path: print(f"[ok] Saved top confusions for {m}: {top_path}")
                cm_path, top_path = analyze_errors(err_df, labels=args.labels, out_dir=args.output_dir, model_filter=None)
                if cm_path: print(f"[ok] Saved confusion matrix (all models): {cm_path}")
                if top_path: print(f"[ok] Saved top confusions (all models): {top_path}")

    # Minimal index.html to preview outputs
    try:
        items = sorted([f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir,f))])
        html = ["<html><head><meta charset='utf-8'><title>Benchmark Report</title></head><body>"]
        html.append("<h1>Benchmark Report</h1>")
        html.append("<h2>Generated Files</h2><ul>")
        for it in items:
            html.append(f"<li><a href='{it}'>{it}</a></li>")
        html.append("</ul>")
        for it in items:
            if it.lower().endswith(".png"):
                html.append(f"<div><h3>{it}</h3><img src='{it}' style='max-width:100%;height:auto;'/></div>")
        html.append("</body></html>")
        with open(os.path.join(args.output_dir,"index.html"), "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"[ok] Wrote HTML preview: {os.path.join(args.output_dir,'index.html')}")
    except Exception as e:
        print(f"[warn] Could not write HTML preview: {e}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
