def dataset_summary(df):
    summary = {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Missing Cells": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        }

    return summary