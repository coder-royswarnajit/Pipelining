import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from Profiling.type_detection import detect_column_types
from Profiling.cardinality import analyze_cardinality


def encode_features(df, target_column=None):
    """
    Full-dataset encoding.
    Use this for EDA/report preprocessing only.
    Do not use this before model train-test split.
    """

    df = df.copy()
    encoded_df = pd.DataFrame(index=df.index)

    type_info = detect_column_types(df)
    cardinality_info = analyze_cardinality(df)

    for col in df.columns:

        if col == target_column:
            encoded_df[col] = df[col]
            continue

        if df[col].nunique(dropna=True) <= 1:
            continue

        detected_type = type_info[col]["detected_type"]

        if detected_type == "identifier":
            continue

        elif detected_type == "text":
            continue

        elif detected_type == "binary":
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str))

        elif detected_type == "categorical":
            cardinality_type = cardinality_info.get(col, {}).get("cardinality_type", "Low Cardinality")

            if cardinality_type in ["Low Cardinality", "Moderate Cardinality"]:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformed = encoder.fit_transform(df[[col]])
                feature_names = encoder.get_feature_names_out([col])
                transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)

                encoded_df = pd.concat([encoded_df, transformed_df], axis=1)


            else:
                freq_map = df[col].value_counts(normalize=True)
                encoded_df[col] = df[col].map(freq_map)


        else:
            encoded_df[col] = df[col]

    return encoded_df



def fit_encoders(X_train):
    """
    Fits encoders only on X_train.
    """

    encoders = {}
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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
            values = values.apply(lambda value: value if value in known_classes else "Unknown")

            if "Unknown" not in encoder.classes_:
                encoder.classes_ = list(encoder.classes_) + ["Unknown"]

            encoded_df[col] = encoder.transform(values)

        elif encoder_info["type"] == "onehot":
            encoder = encoder_info["encoder"]
            feature_names = encoder_info["feature_names"]
            transformed = encoder.transform(X[[col]])
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=X.index)
            encoded_df = pd.concat([encoded_df, transformed_df], axis=1)

    return encoded_df