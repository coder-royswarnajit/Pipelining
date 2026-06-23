import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from Profiling.type_detection import detect_column_types
from Profiling.skewness import detect_skewness



def fit_missing_imputers(X_train, metadata=None, imputation_recommendations=None):
    """
    Fits missing value imputers organized by column type.
    If `metadata` is provided (dict of column -> info with 'detected_type'),
    it will be used instead of calling detect_column_types (avoids LLM).
    """
    if metadata and isinstance(metadata, dict):
        type_info = {}
        for col in X_train.columns:
            detected = metadata.get(col, {}).get("detected_type", "unknown")
            type_info[col] = {"detected_type": detected}
    else:
        if metadata:
            type_info = {
                col: {"detected_type": metadata[col].get("detected_type", "unknown")}
                for col in X_train.columns
                if col in metadata}
        else:
            type_info = detect_column_types(X_train)

    numeric_cols = []
    categorical_cols = []
    text_cols = []
    
    mean_imputers = {}
    median_imputers = {}
    zero_imputers = {}
    
    recommendations = (imputation_recommendations or {})

    for col in X_train.columns:
        detected_type = type_info.get(col, {}).get("detected_type", "unknown")
        column_strategy = (recommendations.get(col, {}).get("strategy"))
        
        if detected_type == "continuous":

            if column_strategy == "knn":
                numeric_cols.append(col)

            elif column_strategy == "mean":
                imputer = SimpleImputer(strategy="mean")
                imputer.fit(X_train[[col]])

                mean_imputers[col] = imputer

            elif column_strategy == "median":
                imputer = SimpleImputer(strategy="median")
                imputer.fit(X_train[[col]])

                median_imputers[col] = imputer

            elif column_strategy == "zero":

                imputer = SimpleImputer(
                    strategy="constant",
                    fill_value=0
                )

                imputer.fit(X_train[[col]])

                zero_imputers[col] = imputer

            else:
                # fallback
                imputer = SimpleImputer(strategy="median")
                imputer.fit(X_train[[col]])

                median_imputers[col] = imputer
            
        elif detected_type in ["binary", "categorical"]:
            categorical_cols.append(col)
        elif detected_type == "text":
            text_cols.append(col)
        

    numeric_imputer = None
    if numeric_cols:
        knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        try:
            knn_imputer.fit(X_train[numeric_cols])
            numeric_imputer = {"imputer": knn_imputer, "columns": numeric_cols}
        except Exception:
            numeric_imputer = None

    categorical_imputers = {}
    for col in categorical_cols:
        if X_train[col].dropna().empty:
            imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        else:
            imputer = SimpleImputer(strategy="most_frequent")
        try:
            imputer.fit(X_train[[col]])
            categorical_imputers[col] = imputer
        except Exception:
            pass

    text_imputers = {}
    for col in text_cols:
        imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        try:
            imputer.fit(X_train[[col]])
            text_imputers[col] = imputer
        except Exception:
            pass

    return {
    "numeric_imputer": numeric_imputer,
    "mean_imputers": mean_imputers,
    "median_imputers": median_imputers,
    "zero_imputers": zero_imputers,
    "categorical_imputers": categorical_imputers,
    "text_imputers": text_imputers,
    }


def transform_missing_values(X, imputers):
    """
    Applies already-fitted imputers to X_train or X_test.
    
    Args:
        X: DataFrame to transform
        imputers: dict with structure {
            "numeric_imputer": KNNImputer,
            "categorical_imputers": {col: SimpleImputer},
            "text_imputers": {col: SimpleImputer},
            "datetime_columns": [col1, col2, ...]
        }
    
    Returns:
        Transformed DataFrame
    """

    X = X.copy()
    
    # Get numeric columns from imputer
    numeric_cols = []
    if imputers["numeric_imputer"] is not None:
        numeric_imputer_info = imputers.get("numeric_imputer")

        if numeric_imputer_info:
            numeric_cols = numeric_imputer_info["columns"]
            knn_imputer = numeric_imputer_info["imputer"]

            existing_cols = [col for col in numeric_cols if col in X.columns]
            X[existing_cols] = knn_imputer.transform(X[existing_cols])
            
    # Transform mean-imputed columns
    for col, imputer in imputers.get("mean_imputers", {}).items():
            if col in X.columns:
                X[[col]] = imputer.transform(X[[col]])

    # Transform median-imputed columns
    for col, imputer in imputers.get("median_imputers", {}).items():
            if col in X.columns:
                X[[col]] = imputer.transform(X[[col]])

    # Transform zero-imputed columns
    for col, imputer in imputers.get("zero_imputers", {}).items():
            if col in X.columns:
                X[[col]] = imputer.transform(X[[col]])
    
    # Transform categorical columns
    for col, imputer in imputers["categorical_imputers"].items():
        if col in X.columns:
            X[[col]] = imputer.transform(X[[col]])
    
    # Transform text columns
    for col, imputer in imputers["text_imputers"].items():
        if col in X.columns:
            X[[col]] = imputer.transform(X[[col]])

    return X