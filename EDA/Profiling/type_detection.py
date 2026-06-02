import ast

from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import column_type_detection_prompt


ALLOWED_TYPES = {
    "binary",
    "categorical",
    "categorical_numeric",
    "continuous",
    "datetime",
    "identifier",
    "text",
    "unknown"
}


def normalize_detected_type(value):
    """
    Normalizes AI returned type.
    """

    if value is None:
        return "unknown"

    detected_type = (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if detected_type not in ALLOWED_TYPES:
        return "unknown"

    return detected_type


def build_column_info_for_ai(df):
    """
    Builds only the required information for AI_Brain
    to detect column types.
    """

    column_info = {}
    total_rows = len(df)

    for col in df.columns:
        series = df[col]

        unique_count = int(series.nunique(dropna=True))

        unique_ratio = (
            round((unique_count / total_rows) * 100, 2)
            if total_rows > 0
            else 0
        )

        sample_values = (
            series
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )

        column_info[col] = {
            "dtype": str(series.dtype),
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "sample_values": sample_values
        }

    return column_info


def parse_ai_response(ai_response):
    """
    Converts AI response into dictionary:
    {
        "column_name": "detected_type"
    }
    """

    cleaned_response = (
        str(ai_response)
        .replace("```python", "")
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        parsed_response = ast.literal_eval(cleaned_response)

        if isinstance(parsed_response, dict):
            return parsed_response

    except Exception:
        return {}

    return {}


def detect_column_types(df):
    """
    Detects column types using AI_Brain.

    Returns:
    {
        "column": {
            "dtype": "...",
            "missing_count": ...,
            "missing_percent": ...,
            "unique_count": ...,
            "unique_ratio": ...,
            "detected_type": "..."
        }
    }
    """

    column_profile = {}
    total_rows = len(df)

    column_info = build_column_info_for_ai(df)

    prompt = column_type_detection_prompt(
        column_metadata=column_info
    )

    ai_response = ask_llm(prompt)

    ai_detected_types = parse_ai_response(ai_response)

    for col in df.columns:
        series = df[col]

        missing_count = int(series.isnull().sum())

        missing_percent = (
            round((missing_count / total_rows) * 100, 2)
            if total_rows > 0
            else 0
        )

        unique_count = int(series.nunique(dropna=True))

        unique_ratio = (
            round((unique_count / total_rows) * 100, 2)
            if total_rows > 0
            else 0
        )

        detected_type = normalize_detected_type(
            ai_detected_types.get(col)
        )

        column_profile[col] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "detected_type": detected_type
        }

    return column_profile