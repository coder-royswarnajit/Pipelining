import ast

from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import plot_recommendation_prompt


ALLOWED_PLOTS = {"histogram", "boxplot", "line", "bar", "pie", "scatter", "skip"}


def parse_plot_recommendations(response):
    '''
    This function takes the raw LLM response and tries to convert it into a Python list.

Example input:

"```json\n[{'plot_type': 'histogram', 'columns': ['Age'], 'title': 'Age Plot'}]\n```"

Expected output:

[{'plot_type': 'histogram', 'columns': ['Age'], 'title': 'Age Plot'}]
    '''
    cleaned_response = (
        response
        .replace("```python", "")
        .replace("```json", "")
        .replace("```", "")
        .strip())

    try:
        parsed = ast.literal_eval(cleaned_response)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    return []


def validate_plot_recommendations(recommendations, df):
    valid_recommendations = []

    for item in recommendations:

        if not isinstance(item, dict):
            continue

        plot_type = str(item.get("plot_type", "")).strip().lower()
        columns = item.get("columns", [])
        title = item.get("title", "")

        if plot_type not in ALLOWED_PLOTS:
            continue

        if plot_type == "skip":
            continue

        if not isinstance(columns, list):
            continue

        valid_columns = [
            col for col in columns
            if col in df.columns
        ]

        if plot_type == "scatter" and len(valid_columns) != 2:
            continue

        if plot_type != "scatter" and len(valid_columns) != 1:
            continue

        valid_recommendations.append(
            {
                "plot_type": plot_type,
                "columns": valid_columns,
                "title": title
            }
        )

    return valid_recommendations


def recommend_plots(metadata, df, target_column=None, problem_type=None):
    prompt = plot_recommendation_prompt(
        metadata=metadata,
        target_column=target_column,
        problem_type=problem_type
    )

    response = ask_llm(prompt)

    recommendations = parse_plot_recommendations(response)

    valid_recommendations = validate_plot_recommendations(
        recommendations=recommendations,
        df=df
    )

    return valid_recommendations