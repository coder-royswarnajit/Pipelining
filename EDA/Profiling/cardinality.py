import pandas as pd


def analyze_cardinality(df, type_info):
    """
    Analyzes cardinality only for categorical-like columns
    using already detected column types from profiling_pipeline.py.
    """

    cardinality_report = {}

    allowed_types = [
        "binary",
        "categorical"
    ]

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            continue

        detected_type = type_info.get(col, {}).get(
            "detected_type",
            "unknown"
        )

        # Skip continuous columns
        if detected_type not in allowed_types:
            continue

        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)

        if unique_ratio > 0.9:
            cardinality_type = "Identifier-like"

        elif unique_ratio >= 0.50:
            cardinality_type = "High Cardinality"

        elif unique_ratio >= 0.10:
            cardinality_type = "Moderate Cardinality"

        else:
            cardinality_type = "Low Cardinality"

        cardinality_report[col] = {
            "detected_type": detected_type,
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 3),
            "cardinality_type": cardinality_type,
            "top_values": series.value_counts().head(5).to_dict()
        }

    return cardinality_report