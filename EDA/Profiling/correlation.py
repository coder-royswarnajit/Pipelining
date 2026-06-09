import os
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_detected_type(value):
    if value is None:
        return "unknown"

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_alt_text():
    # matplotalt removed — alt text generation disabled
    return None


def get_detected_type(type_info, col):
    if type_info is None:
        return "unknown"

    col_str = str(col).strip()

    value = None

    if col in type_info:
        value = type_info[col]

    elif col_str in type_info:
        value = type_info[col_str]

    else:
        for key, val in type_info.items():
            if str(key).strip().lower() == col_str.lower():
                value = val
                break

    if value is None:
        return "unknown"

    if isinstance(value, dict):
        return normalize_detected_type(
            value.get("detected_type", "unknown")
        )

    return normalize_detected_type(value)


def looks_like_identifier(series, col):
    col_lower = str(col).lower()

    unique_count = series.nunique(dropna=True)
    total_count = len(series.dropna())

    unique_ratio = (
        unique_count / total_count
        if total_count > 0
        else 0
    )

    if (
        "id" in col_lower
        or "udi" in col_lower
        or "serial" in col_lower
        or "product" in col_lower
        or "code" in col_lower
        or "name" in col_lower
    ):
        return True

    if unique_ratio >= 0.98:
        return True

    return False


def correlation_heatmap(
    df,
    plot_folder="eda_plots",
    type_info=None,
    target_column=None
):
    """
    Generates correlation heatmap and graph summary.

    Returns:
    {
        "plot_path": path or None,
        "plot_summary": {...}
    }
    """

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    continuous_columns = []
    excluded_columns = {}

    target_name = (
        str(target_column).strip().lower()
        if target_column is not None
        else None
    )

    for col in numeric_df.columns:

        series = numeric_df[col]
        col_name = str(col).strip().lower()

        if target_name is not None and col_name == target_name:
            excluded_columns[col] = "target_column"
            continue

        detected_type = get_detected_type(type_info, col)

        if detected_type == "identifier":
            excluded_columns[col] = "identifier"
            continue

        if detected_type in [
            "binary",
            "categorical",
            "categorical_numeric",
            "datetime",
            "text"
        ]:
            excluded_columns[col] = detected_type
            continue

        if detected_type == "continuous":
            continuous_columns.append(col)
            continue

        if looks_like_identifier(series, col):
            excluded_columns[col] = "identifier_like"
            continue

        unique_count = series.nunique(dropna=True)

        if unique_count <= 2:
            excluded_columns[col] = "binary_like"
            continue

        continuous_columns.append(col)

    summary = {
        "plot_type": "heatmap",
        "columns": continuous_columns,
        "excluded_columns": excluded_columns,
        "strong_correlations": [],
        "plot_path": None,
        "purpose": "Shows linear correlation between continuous numeric input features."
    }

    if len(continuous_columns) < 2:
        summary["message"] = (
            "Heatmap not created because fewer than 2 valid continuous columns were available."
        )

        return {
            "plot_path": None,
            "plot_summary": summary
        }

    filtered_df = numeric_df[continuous_columns]

    corr = filtered_df.corr()

    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            if j <= i:
                continue

            corr_value = corr.loc[col1, col2]

            if abs(corr_value) >= 0.7:
                summary["strong_correlations"].append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(corr_value), 3)
                    }
                )

    os.makedirs(plot_folder, exist_ok=True)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        cbar=True,
        annot=True,
        fmt=".2f",
        square=True
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    heatmap_path = os.path.join(
        plot_folder,
        "correlation_heatmap.png"
    )

    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()

    summary["plot_path"] = heatmap_path

    return {
        "plot_path": heatmap_path,
        "plot_summary": summary
    }