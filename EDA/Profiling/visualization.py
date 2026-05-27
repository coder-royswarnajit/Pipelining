import os
import math
import matplotlib.pyplot as plt


'''Shows the data distribution for numerical columns'''
def distribution_plots(df, plot_folder="eda_plots"):

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    binary_cols = []
    continuous_cols = []

    # Separate binary and continuous columns
    for col in numeric_df.columns:
        unique_values = numeric_df[col].dropna().unique()

        if len(unique_values) == 2:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    continuous_cols = continuous_cols[:20]

    os.makedirs(plot_folder, exist_ok=True)
    plot_paths = []

    # Continuous column distributions
    if continuous_cols:
        n_cols = 3
        n_rows = math.ceil(len(continuous_cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        if not hasattr(axes, "flatten"):
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(continuous_cols):
            series = numeric_df[col].dropna().reset_index(drop=True)

            axes[i].plot(
                series.index,
                series.values,
                marker="o",
                linewidth=1
            )

            axes[i].set_title(f"Line Plot of {col}")
            axes[i].set_xlabel("Record Index")
            axes[i].set_ylabel(col)

        # Hide empty subplots
        for j in range(len(continuous_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        continuous_plot_path = os.path.join(
            plot_folder,
            "continuous_distribution_plots.png"
        )
        plt.savefig(continuous_plot_path, bbox_inches="tight")
        plt.close()

        print(f"Continuous distribution plots saved at: {continuous_plot_path}")
        plot_paths.append(continuous_plot_path)

    # Binary column distributions
    for col in binary_cols:
        value_counts = df[col].value_counts()

        plt.figure(figsize=(5, 4))
        plt.pie(
            x=value_counts,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )

        plt.title(f'Binary Distribution - {col}')
        plt.tight_layout()

        safe_col_name = str(col).replace(" ", "_").replace("/", "_").replace("\\", "_")
        binary_plot_path = os.path.join(
            plot_folder,
            f"binary_distribution_{safe_col_name}.png"
        )
        plt.savefig(binary_plot_path, bbox_inches="tight")
        plt.close()

        print(f"Binary distribution plot saved at: {binary_plot_path}")
        plot_paths.append(binary_plot_path)

    return plot_paths