import pandas as pd


def detect_skewness(df, type_info=None, target_column=None):

    skewness_report = {}

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    for col in numeric_df.columns:

        if col == target_column:
            print(f"Skipped target column from skewness: {col}")
            continue

        detected_type = None

        if type_info is not None:
            detected_type = type_info.get(col, {}).get(
                "detected_type",
                None
            )

        if detected_type == "identifier":
            continue

        if detected_type in ["binary", "categorical_numeric"]:
            continue

        if type_info is not None and detected_type != "continuous":
            continue

        series = numeric_df[col].dropna()

        if series.empty:
            continue

        if series.nunique() <= 2:
            continue

        skew_value = series.skew()

        if abs(skew_value) < 0.5:
            skew_type = "Approximately Symmetric"

        elif 0.5 <= abs(skew_value) < 1:
            skew_type = "Moderately Skewed"

        else:
            skew_type = "Highly Skewed"

        if skew_value > 0:
            direction = "Right Skewed"

        elif skew_value < 0:
            direction = "Left Skewed"

        else:
            direction = "Symmetric"

        skewness_report[col] = {
            "skewness": round(skew_value, 3),
            "skew_type": skew_type,
            "direction": direction
        }

    return skewness_report