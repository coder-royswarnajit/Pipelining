import pandas as pd


def detect_column_types(df):

    column_profile = {}

    for col in df.columns:

        series = df[col]

        profile = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isnull().sum()),
            "missing_percent": round(
                (series.isnull().sum() / len(df)) * 100,
                2
            ),
            "unique_count": int(series.nunique(dropna=True)),
            "detected_type": None
        }

        
        #BOOLEAN/BINARY
        unique_values = series.dropna().unique()

        if len(unique_values) == 2:
            profile["detected_type"] = "binary"

        
        # NUMERIC
        elif pd.api.types.is_numeric_dtype(series):

            unique_count = series.nunique(dropna=True)

            if unique_count <= 10:
                profile["detected_type"] = "categorical_numeric"

            else:
                profile["detected_type"] = "continuous"

        
        # DATETIME
        elif pd.api.types.is_datetime64_any_dtype(series):

            profile["detected_type"] = "datetime"

        
        #TEXT/CATEGORICAL
        elif pd.api.types.is_object_dtype(series):

            unique_ratio = (
                series.nunique(dropna=True) / len(series)
            )

            avg_length = (
                series.dropna()
                .astype(str)
                .apply(len)
                .mean()
            )

            # Identifier-like columns
            if unique_ratio > 0.9:

                profile["detected_type"] = "identifier"

            # Long text columns
            elif avg_length > 30:

                profile["detected_type"] = "text"

            # Normal categorical
            else:

                profile["detected_type"] = "categorical"

        
        #FALLBACK
        else:
            profile["detected_type"] = "unknown"

        column_profile[col] = profile

    return column_profile