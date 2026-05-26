import pandas as pd


def analyze_cardinality(df):

    found = False

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            continue

        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)

        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or (pd.api.types.is_numeric_dtype(series) and unique_count <= 20)
        )

        if not is_categorical:
            continue

        found = True

        if unique_ratio > 0.9:
            cardinality_type = "Identifier-like"
        elif unique_count > 50:
            cardinality_type = "High Cardinality"
        elif unique_count > 15:
            cardinality_type = "Moderate Cardinality"
        else:
            cardinality_type = "Low Cardinality"

        print(f"\nColumn: {col}")
        print(f"Unique Values: {unique_count}")
        print(f"Unique Ratio: {unique_ratio:.3f}")
        print(f"Cardinality Type: {cardinality_type}")
        print("Top Values:")
        print(series.value_counts().head(5))

    if not found:
        print("No categorical columns found.")