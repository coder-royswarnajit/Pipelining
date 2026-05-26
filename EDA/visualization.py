import matplotlib.pyplot as plt
import math


'''Shows the data distribution for numerical columns'''
def distribution_plots(df):

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    binary_cols = []
    continuous_cols = []

    for col in numeric_df.columns:
        unique_values = numeric_df[col].dropna().unique()

        if len(unique_values) == 2:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    continuous_cols = continuous_cols[:20]

    # Continuous column distributions
    if continuous_cols:

        n_cols = 3
        n_rows = math.ceil(len(continuous_cols) / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(15, 4 * n_rows)
        )

        axes = axes.flatten()

        for i, col in enumerate(continuous_cols):
            axes[i].hist(
                numeric_df[col].dropna(),
                bins=30,
                edgecolor='black'
            )

            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

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
        plt.show()