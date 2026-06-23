from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train):
    """
    Fits scaler only on X_train.
    """

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()

    if len(numeric_cols) > 0:
        scaler.fit(X_train[numeric_cols])

    return scaler, numeric_cols


def transform_scaling(X, scaler, numeric_cols):
    """
    Applies fitted scaler to X_train or X_test.
    """

    X = X.copy()

    existing_numeric_cols = [
        col for col in numeric_cols
        if col in X.columns
    ]

    if len(existing_numeric_cols) > 0:
        X[existing_numeric_cols] = scaler.transform(
            X[existing_numeric_cols]
        )

    return X