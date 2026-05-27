import pandas as pd


def detect_column_types(df):

    column_profile = {}

    total_rows = len(df)

    for col in df.columns:

        series = df[col]

        missing_count = int(series.isnull().sum())
        missing_percent = round((missing_count / total_rows) * 100, 2)

        unique_count = int(series.nunique(dropna=True))

        # Always define unique_ratio before using it
        if total_rows > 0:
            unique_ratio = round((unique_count / total_rows) * 100, 2)
        else:
            unique_ratio = 0

        profile = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "detected_type": None
        }

        unique_values = series.dropna().unique()

        # EMPTY COLUMN
        if unique_count == 0:
            profile["detected_type"] = "empty"

        # DATETIME
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile["detected_type"] = "datetime"

        # BOOLEAN / BINARY
        elif unique_count == 2:
            profile["detected_type"] = "binary"

        # IDENTIFIER
        elif unique_ratio >= 90:
            profile["detected_type"] = "identifier"

        # NUMERIC
        elif pd.api.types.is_numeric_dtype(series):

            if unique_count <= 10:
                profile["detected_type"] = "categorical_numeric"

            else:
                profile["detected_type"] = "continuous"

        # OBJECT / TEXT / CATEGORICAL
        elif pd.api.types.is_object_dtype(series):

            if unique_count <= 20:
                profile["detected_type"] = "categorical"

            elif unique_ratio >= 90:
                profile["detected_type"] = "identifier"

            else:
                profile["detected_type"] = "text"

        # FALLBACK
        else:
            profile["detected_type"] = "unknown"

        column_profile[col] = profile

    return column_profile