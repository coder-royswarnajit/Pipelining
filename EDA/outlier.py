import pandas as pd


def detect_outliers(df):

    outlier_report = {}

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        # Skip empty columns
        if series.empty:
            continue

        # Skip binary columns
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

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        outlier_count = len(outliers)

        outlier_percent = round((outlier_count / len(series)) * 100, 2)

        outlier_report[col] = {
            "outlier_count": int(outlier_count),
            "outlier_percent": outlier_percent,
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        }

    return outlier_report