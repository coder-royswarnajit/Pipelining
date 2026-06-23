
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