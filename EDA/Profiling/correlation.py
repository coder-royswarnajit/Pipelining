
def normalize_detected_type(value):
    if value is None:
        return "unknown"

    return (str(value).strip().lower().replace(" ", "_").replace("-", "_"))


def get_detected_type(type_info, col):
    if type_info is None:
        return "unknown"

    col_str = str(col).strip()
    value = None

    if col in type_info:
        value = type_info[col]

    elif col_str in type_info:
        value = type_info[col_str]

    else:
        for key, val in type_info.items():
            if str(key).strip().lower() == col_str.lower():
                value = val
                break

    if value is None:
        return "unknown"

    if isinstance(value, dict):
        return normalize_detected_type(value.get("detected_type", "unknown"))

    return normalize_detected_type(value)


def looks_like_identifier(series, col):
    col_lower = str(col).lower()

    unique_count = series.nunique(dropna=True)
    total_count = len(series.dropna())

    unique_ratio = (unique_count / total_count if total_count > 0 else 0)

    if ("id" in col_lower or "udi" in col_lower or "serial" in col_lower or "product" in col_lower or "code" in col_lower or "name" in col_lower):
        return True

    if unique_ratio >= 0.98:
        return True

    return False

def correlation_analysis(df, type_info=None, target_column=None, correlation_threshold=0.7):
    """
    Returns strongly correlated continuous feature pairs.
    """

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    continuous_columns = []
    excluded_columns = {}

    target_name = (str(target_column).strip().lower() if target_column is not None else None)

    for col in numeric_df.columns:

        series = numeric_df[col]
        col_name = str(col).strip().lower()

        if target_name is not None and col_name == target_name:
            excluded_columns[col] = "target_column"
            continue

        detected_type = get_detected_type(type_info, col)

        if detected_type == "identifier":
            excluded_columns[col] = "identifier"
            continue

        if detected_type in ["binary", "categorical", "datetime", "text",]:
            excluded_columns[col] = detected_type
            continue

        if detected_type == "continuous":
            continuous_columns.append(col)
            continue

        if looks_like_identifier(series, col):
            excluded_columns[col] = "identifier_like"
            continue

        if series.nunique(dropna=True) <= 2:
            excluded_columns[col] = "binary_like"
            continue

        continuous_columns.append(col)

    summary = {
        "columns": continuous_columns,
        "excluded_columns": excluded_columns,
        "strong_correlations": [],
        "purpose": ("Shows pairs of continuous numeric features with strong linear correlation."),}

    if len(continuous_columns) < 2:
        return summary

    corr = numeric_df[continuous_columns].corr()

    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):

            if j <= i:
                continue

            corr_value = corr.loc[col1, col2]

            if abs(corr_value) >= correlation_threshold:
                summary["strong_correlations"].append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(corr_value), 3),})

    summary["strong_correlations"].sort(key=lambda x: abs(x["correlation"]), reverse=True,)
    return summary