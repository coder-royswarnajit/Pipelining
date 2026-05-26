import pandas as pd


def detect_skewness(df):

    skewness_report = {}

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    for col in numeric_df.columns:

        series = numeric_df[col].dropna()

        # Skip empty columns
        if series.empty:
            continue

        # Skip binary columns
        if series.nunique() <= 2:
            continue

        # Calculate skewness
        skew_value = series.skew()

        # Determine skew category
        if abs(skew_value) < 0.5:
            skew_type = "Approximately Symmetric"

        elif 0.5 <= abs(skew_value) < 1:
            skew_type = "Moderately Skewed"

        else:
            skew_type = "Highly Skewed"

        # Direction
        if skew_value > 0:
            direction = "Right Skewed"

        elif skew_value < 0:
            direction = "Left Skewed"

        else:
            direction = "Symmetric"

        skewness_report[col] = {"skewness": round(skew_value, 3),
                                "skew_type": skew_type,
                                "direction": direction}

    return skewness_report