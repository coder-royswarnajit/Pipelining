from Profiling.type_detection import detect_column_types


def fit_identifier_removal(column_info):
    """
    Detect identifier columns using existing type detection.

    Returns:
    {
        "identifier_columns": [...]
    }
    """

    identifier_columns = []

    for col, info in column_info.items():
        if info.get("detected_type") == "identifier":
            identifier_columns.append(col)

    return {"identifier_columns": identifier_columns}


def transform_identifier_removal(X, identifier_config):
    """
    Removes identifier columns using fitted configuration.
    """

    X = X.copy()

    identifier_columns = (identifier_config.get("identifier_columns", []))

    X = X.drop(columns=identifier_columns, errors="ignore")

    return X