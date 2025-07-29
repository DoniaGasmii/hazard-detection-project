import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# ============== Comparison Plot ============== #
def plot_metric_comparison(
    csv_paths,
    metric_column,
    labels=None,
    title=None,
    ylabel=None,
    save_path=None
):
    """
    Plot any metric over epochs for multiple training runs using a pastel 'Paired' palette.

    Args:
        csv_paths (list of str): Paths to training logs (CSV files).
        metric_column (str): Column name of the metric to plot (e.g., 'metrics/mAP50(B)', 'val/loss').
        labels (list of str, optional): Legend labels. Defaults to "Run 1", "Run 2", etc.
        title (str, optional): Plot title.
        ylabel (str, optional): Y-axis label. Defaults to metric_column.
        save_path (str, optional): If set, saves the plot to this path.
    """
    if labels and len(labels) != len(csv_paths):
        raise ValueError("Length of labels must match length of csv_paths")

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Paired", n_colors=len(csv_paths))

    plt.figure(figsize=(10, 6))

    for i, path in enumerate(csv_paths):
        if not os.path.isfile(path):
            print(f"[Warning] File not found: {path}")
            continue

        df = pd.read_csv(path)
        if 'epoch' not in df.columns or metric_column not in df.columns:
            print(f"[Warning] Missing required columns in {path}. Skipped.")
            continue

        label = labels[i] if labels else f"Run {i+1}"
        plt.plot(df['epoch'], df[metric_column], label=label, color=palette[i], linewidth=1)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel if ylabel else metric_column, fontsize=12)
    plt.title(title if title else f"{metric_column} Comparison", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Info] Plot saved to: {save_path}")
    else:
        plt.show()
