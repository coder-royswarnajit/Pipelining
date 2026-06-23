def fit_text_removal(metadata):

    text_columns = []

    for col, info in metadata.items():

        if info.get("detected_type") == "text":
            text_columns.append(col)

    return {
        "text_columns": text_columns
    }


def transform_text_removal(X, config):

    X = X.copy()

    cols = config.get("text_columns", [])

    existing_cols = [
        col
        for col in cols
        if col in X.columns
    ]

    return X.drop(columns=existing_cols, errors="ignore")