import os
import math
import matplotlib.pyplot as plt


def box_plots(df, plot_folder="eda_plots"):
    """
    Automatically creates box plots for useful continuous numeric columns.
    Excludes ID-like columns, binary columns, target-like columns, and categorical-like numeric columns.
    """

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    continuous_cols = []

    total_rows = len(df)

    for col in numeric_df.columns:
        col_lower = col.lower()

        unique_count = numeric_df[col].dropna().nunique()
        unique_ratio = unique_count / total_rows

        if unique_count <= 2:
            continue

        if any(word in col_lower for word in ["id", "udi", "index"]):
            continue

        if any(word in col_lower for word in ["target", "label", "class", "failure"]):
            continue

        if unique_ratio > 0.90:
            continue

        continuous_cols.append(col)

    continuous_cols = continuous_cols[:20]

    if len(continuous_cols) == 0:
        return None

    os.makedirs(plot_folder, exist_ok=True)

    n_cols = 3
    n_rows = math.ceil(len(continuous_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    if not hasattr(axes, "flatten"):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        axes[i].boxplot(numeric_df[col].dropna(), vert=True)

        axes[i].set_title(f"Box Plot of {col}")
        axes[i].set_ylabel(col)

    for j in range(len(continuous_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    box_plot_path = os.path.join(plot_folder, "box_plots.png")

    plt.savefig(box_plot_path, bbox_inches="tight")
    plt.close()


    return box_plot_path