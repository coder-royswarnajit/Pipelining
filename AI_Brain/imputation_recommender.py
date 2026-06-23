import json

from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import build_imputation_recommendation_prompt


def recommend_imputations(metadata):
    """
    Uses AI Brain to recommend missing-value handling
    strategies for all columns.

    Returns:
        dict
    """

    try:

        prompt = (build_imputation_recommendation_prompt(metadata))
        response = ask_llm(prompt)

        if not response:
            return {}

        response = response.strip()

        if response.startswith("```json"):
            response = (
                response
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif response.startswith("```"):
            response = (
                response
                .replace("```", "")
                .strip()
            )

        return json.loads(response)

    except Exception as e:
        print(f"Failed to generate imputation recommendations: {e}")

        return {}