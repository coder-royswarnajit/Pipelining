import pandas as pd

from Profiling.outlier import detect_outliers
from Profiling.type_detection import detect_column_types


def handle_outliers(df, target_column=None):
    """
    Full-dataset outlier handling.
    Use this for EDA/report preprocessing only.
    Target column is protected.
    """

    df = df.copy()

    outlier_info = detect_outliers(df)
    type_info = detect_column_types(df)

    for col, info in outlier_info.items():

        if col == target_column:
            print(f"Skipped target column: {col}")
            continue

        if col not in type_info:
            continue

        detected_type = type_info[col]["detected_type"]

        if detected_type != "continuous":
            continue

        outlier_count = info["outlier_count"]

        if outlier_count == 0:
            continue

        lower_bound = info["lower_bound"]
        upper_bound = info["upper_bound"]

        df[col] = df[col].astype(float)

        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound


# LEAKAGE-SAFE MODELLING FUNCTIONS
def fit_outlier_bounds(X_train):
    """
    Calculates outlier bounds only from X_train.
    """

    bounds = {}

    numeric_df = X_train.select_dtypes(include=["int64", "float64"])

    for col in numeric_df.columns:

        series = numeric_df[col].dropna()

        if series.empty:
            continue

        if series.nunique() <= 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        bounds[col] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    return bounds


def transform_outliers(X, bounds):
    """
    Applies already-fitted outlier bounds to X_train or X_test.
    """

    X = X.copy()

    for col, bound_info in bounds.items():

        if col not in X.columns:
            continue

        X[col] = X[col].astype(float)

        lower_bound = bound_info["lower_bound"]
        upper_bound = bound_info["upper_bound"]

        X.loc[X[col] < lower_bound, col] = lower_bound
        X.loc[X[col] > upper_bound, col] = upper_bound

    return X