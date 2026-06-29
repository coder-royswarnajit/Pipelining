from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import shap_explanation_prompt


def explain_shap_results(shap_analysis, target_column=None, problem_type=None, model_name=None,):
    """
    Converts compact SHAP analysis into business-friendly model context.
    """

    if not isinstance(shap_analysis, dict):
        return ""

    if shap_analysis.get("status") != "success":
        return ""

    prompt = shap_explanation_prompt(shap_analysis=shap_analysis, target_column=target_column, problem_type=problem_type, model_name=model_name,)
    response = ask_llm(prompt)

    return str(response).replace("\\n", "\n").replace("**", "").strip()
