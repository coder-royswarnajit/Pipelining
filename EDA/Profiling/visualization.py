import os
import pandas as pd
import matplotlib.pyplot as plt


def safe_column_name(col):
    return (
        str(col)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def summarize_numeric_series(series):
    series = series.dropna()

    if series.empty:
        return {"count": 0,
                "message": "No valid numeric values available."}

    return {
        "count": int(series.count()),
        "mean": round(float(series.mean()), 3),
        "median": round(float(series.median()), 3),
        "std": round(float(series.std()), 3),
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "skewness": round(float(series.skew()), 3)
    }


def generate_ai_recommended_plots(df, plot_recommendations, plot_folder="eda_plots"):
    """
    Generates plots based on AI_Brain recommendations.
    """

    os.makedirs(plot_folder, exist_ok=True)

    plot_paths = []
    plot_summaries = []

    for i, rec in enumerate(plot_recommendations):

        plot_type = rec.get("plot_type")
        columns = rec.get("columns", [])
        title = rec.get("title", "")

        if plot_type == "skip":
            continue

        try:
            plt.figure(figsize=(8, 5))

            plot_key = (f"{i + 1}_{plot_type}_ of {safe_column_name('_'.join(map(str, columns)))}")

            summary = {
                "plot_key": plot_key,
                "plot_type": plot_type,
                "columns": columns,
                "title": title,
                "plot_path": None,
                "insights": {} }

            if plot_type == "histogram":
                col = columns[0]
                series = df[col].dropna()

                plt.hist(series, bins=30, edgecolor="black")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.title(title or f"Histogram of {col}")

                summary["insights"] = {
                    "numeric_summary": summarize_numeric_series(series),
                    "purpose": "Shows distribution, spread, and skewness."
                }

            elif plot_type == "boxplot":
                col = columns[0]
                series = df[col].dropna()

                plt.boxplot(series)
                plt.ylabel(col)
                plt.title(title or f"Boxplot of {col}")

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = series[
                    (series < lower_bound) |
                    (series > upper_bound)
                ]

                summary["insights"] = {
                    "numeric_summary": summarize_numeric_series(series),
                    "q1": round(float(q1), 3),
                    "q3": round(float(q3), 3),
                    "iqr": round(float(iqr), 3),
                    "outlier_count": int(len(outliers)),
                    "purpose": "Shows spread, median, quartiles, and possible outliers."
                }

            elif plot_type == "line":
                col = columns[0]
                series = df[col].dropna().reset_index(drop=True)

                plt.plot(series.index, series.values, marker="o", linewidth=1)
                plt.xlabel("Record Index")
                plt.ylabel(col)
                plt.title(title or f"Line Plot of {col}")

                summary["insights"] = {
                    "numeric_summary": summarize_numeric_series(series),
                    "first_value": round(float(series.iloc[0]), 3) if len(series) > 0 else None,
                    "last_value": round(float(series.iloc[-1]), 3) if len(series) > 0 else None,
                    "purpose": "Shows how values change across row order."
                }

            elif plot_type == "bar":
                col = columns[0]
                value_counts = df[col].dropna().value_counts().head(15)

                plt.bar(value_counts.index.astype(str), value_counts.values)
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.title(title or f"Bar Plot of {col}")

                summary["insights"] = {
                    "top_values": value_counts.to_dict(),
                    "unique_count": int(df[col].nunique(dropna=True)),
                    "purpose": "Shows frequency distribution of categories."
                }

            elif plot_type == "pie":
                col = columns[0]
                value_counts = df[col].dropna().value_counts().head(8)

                plt.pie(
                    value_counts.values,
                    labels=value_counts.index.astype(str),
                    autopct="%1.1f%%",
                    startangle=90
                )

                plt.title(title or f"Pie Chart of {col}")

                total = value_counts.sum()

                percentages = {
                    str(k): round((v / total) * 100, 2)
                    for k, v in value_counts.to_dict().items()
                }

                summary["insights"] = {
                    "category_percentages": percentages,
                    "purpose": "Shows category share as part of a whole."
                }

            elif plot_type == "scatter":
                x_col = columns[0]
                y_col = columns[1]

                if (not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col])):
                    print(f"Skipping scatter for non-numeric columns: {x_col}, {y_col}")
                    plt.close()
                    continue

                plot_df = df[[x_col, y_col]].dropna()

                plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.7)

                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title or f"Scatter Plot: {x_col} vs {y_col}")

                correlation = None

                if (pd.api.types.is_numeric_dtype(plot_df[x_col]) and pd.api.types.is_numeric_dtype(plot_df[y_col])):
                    correlation = plot_df[x_col].corr(plot_df[y_col])

                summary["insights"] = {
                    "x_summary": summarize_numeric_series(plot_df[x_col]),
                    "y_summary": summarize_numeric_series(plot_df[y_col]),
                    "correlation": round(float(correlation), 3) if correlation is not None else None,
                    "purpose": "Shows relationship between two numeric columns."
                }

            else:
                plt.close()
                continue

            plt.tight_layout()

            
            file_name = (f"{i + 1}_{plot_type}_ {safe_column_name('_'.join(map(str, columns)))}.png")
            plot_path = os.path.join(plot_folder, file_name)

            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

            summary["plot_path"] = plot_path

            plot_paths.append(plot_path)
            plot_summaries.append(summary)

        except Exception as e:
            plt.close()
            print(f"Failed to generate {plot_type} for {columns}: {e}")

    return {"plot_paths": plot_paths, "plot_summaries": plot_summaries}