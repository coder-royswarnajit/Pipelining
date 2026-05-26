def dataset_summary(df):
    summary = {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Missing Cells": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (KB)": float(
            df.memory_usage(deep=True).sum() / 1024
        ),
        "Data Types": {
            str(k): int(v)
            for k, v in df.dtypes.value_counts().items()
        }
    }

    return summary