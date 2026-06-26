from AI_Brain.prompt_templates import optimization_explanation_prompt
from AI_Brain.llm_client import ask_llm


def explain_optimized_model(
    model_name,
    problem_type,
    optimization_result,
    shap_analysis,
):
    prompt = optimization_explanation_prompt(
        model_name=model_name,
        problem_type=problem_type,
        optimization_result=optimization_result,
        shap_analysis=shap_analysis,
    )

    response = ask_llm(prompt)

    return response