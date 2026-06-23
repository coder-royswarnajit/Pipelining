import pandas as pd
import numpy as np


def fit_correlation_filter(X_train, threshold=0.95):
    """
    Fits correlation filter using X_train only.

    Returns:
    {
        "dropped_columns": [...]
    }
    """

    numeric_df = X_train.select_dtypes(include=[np.number])

    if numeric_df.shape[1] <= 1:
        return {"dropped_columns": []}

    corr_matrix = numeric_df.corr().abs()

    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))

    dropped_columns = []

    for column in upper_triangle.columns:

        if any(upper_triangle[column] > threshold):
            dropped_columns.append(column)

    return {"dropped_columns": dropped_columns}


def transform_correlation_filter(X, correlation_config):
    """
    Removes highly correlated columns.
    """

    X = X.copy()
    dropped_columns = (correlation_config.get("dropped_columns", []))
    X = X.drop(columns=dropped_columns, errors="ignore")

    return X