import pandas as pd


def detect_outliers(df, type_info=None):

    outlier_report = {}

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    for col in numeric_df.columns:

        detected_type = None

        if type_info is not None:
            detected_type = type_info.get(col, {}).get(
                "detected_type",
                None
            )

        # Skip identifier columns
        if detected_type == "identifier":
            print(f"Skipped identifier column from outlier detection: {col}")
            continue

        # Skip binary and categorical numeric columns
        if detected_type in ["binary", "categorical_numeric"]:
            continue

        # If type_info is provided, only allow continuous columns
        if type_info is not None and detected_type != "continuous":
            continue

        series = numeric_df[col].dropna()

        # Skip empty columns
        if series.empty:
            continue

        # Fallback skip binary columns
        if series.nunique() <= 2:
            continue

        # IQR Method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        # Prevent division/logic issues
        if IQR == 0:
            continue

        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        outlier_count = len(outliers)

        outlier_percent = round(
            (outlier_count / len(series)) * 100,
            2
        )

        outlier_report[col] = {
            "outlier_count": int(outlier_count),
            "outlier_percent": outlier_percent,
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        }

    return outlier_report