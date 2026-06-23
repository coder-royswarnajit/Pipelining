import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from Profiling.type_detection import detect_column_types




def fit_encoders(X_train, metadata=None):
    """
    Fits encoders only on X_train.
    If `metadata` provided, use metadata[col]['detected_type'] to avoid LLM.
    """
    encoders = {}
    if metadata and isinstance(metadata, dict):
        type_info = {col: {"detected_type": metadata.get(col, {}).get("detected_type", "unknown")}
                     for col in X_train.columns}
    else:
        type_info = detect_column_types(X_train)

    categorical_cols = [col for col, info in type_info.items() if info.get("detected_type") in ["binary", "categorical"]]

    for col in categorical_cols:
        unique_count = X_train[col].nunique(dropna=True)
        if unique_count <= 1:
            continue
        if unique_count == 2:
            encoder = LabelEncoder()
            encoder.fit(X_train[col].astype(str))
            encoders[col] = {"type": "label", "encoder": encoder}
        else:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(X_train[[col]])
            encoders[col] = {"type": "onehot", "encoder": encoder, "feature_names": encoder.get_feature_names_out([col])}

    return encoders


def transform_encoding(X, encoders):
    """
    Applies fitted encoders to X_train or X_test.
    """

    X = X.copy()

    encoded_df = X.drop(columns=list(encoders.keys()), errors="ignore")

    for col, encoder_info in encoders.items():
        if col not in X.columns:
            continue

        if encoder_info["type"] == "label":
            encoder = encoder_info["encoder"]
            values = X[col].astype(str)
            known_classes = set(encoder.classes_)
            values = values.apply(lambda value: value if value in known_classes else encoder.classes_[0]) #This maps unseen categories to an existing class instead of trying to mutate the encoder.
            encoded_df[col] = encoder.transform(values)

        elif encoder_info["type"] == "onehot":
            encoder = encoder_info["encoder"]
            feature_names = encoder_info["feature_names"]
            transformed = encoder.transform(X[[col]])
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=X.index)
            encoded_df = pd.concat([encoded_df, transformed_df], axis=1)

    if encoded_df.isnull().sum().sum() > 0:
        raise ValueError(f"NaNs introduced during encoding: {encoded_df.columns[encoded_df.isnull().any()].tolist()}")
        
    return encoded_df