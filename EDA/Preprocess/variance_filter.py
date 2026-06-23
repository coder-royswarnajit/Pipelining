from sklearn.feature_selection import VarianceThreshold


def fit_variance_filter(X_train, threshold=0.01):
    
    numeric_df = X_train.select_dtypes(include=["number"])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)

    kept_columns = numeric_df.columns[selector.get_support()].tolist()
    dropped_columns = [col for col in X_train.columns if col not in kept_columns]

    return {
        "selector": selector,
        "kept_columns": kept_columns,
        "dropped_columns": dropped_columns}


def transform_variance_filter(X, variance_config):
    X = X.copy()

    kept_columns = variance_config["kept_columns"]

    return X[[col for col in kept_columns if col in X.columns]]