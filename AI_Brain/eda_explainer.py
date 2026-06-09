from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import eda_explanation_prompt


def explain_eda_results(eda_results, target_column=None, problem_type=None, plot_recommendations=None, plot_summaries=None):
    """
    Uses AI_Brain to explain EDA results in simple business-friendly language.

    Inputs:
    - eda_results: output from run_eda_pipeline()
    - target_column: target detected by AI_Brain
    - problem_type: classification/regression
    - plot_recommendations: AI-recommended plots, if available

    Returns:
    - explanation text
    """

    summary_for_llm = {
        "dataset_summary": eda_results.get("dataset_summary", {}),
        "missing_report": eda_results.get("missing_report", {}),
        "warnings": eda_results.get("warnings", {}),
        "column_info": eda_results.get("column_info", {}),
        "outlier_report": eda_results.get("outlier_report", {}),
        "skewness_report": eda_results.get("skewness_report", {}),
        "cardinality_report": eda_results.get("cardinality_report", {}),
        "plot_recommendations": plot_recommendations or eda_results.get(
            "plot_recommendations",
            []
        ),
        "plot_summaries": plot_summaries or eda_results.get(
            "plot_summaries",
            []
        )
    }

    prompt = eda_explanation_prompt(eda_summary=summary_for_llm, target_column=target_column, problem_type=problem_type)

    response = ask_llm(prompt)

    cleaned_response = (str(response).replace("\\n", "\n").replace("**", "").strip())

    return cleaned_response