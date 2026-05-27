from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import problem_type_prompt


def analyze_problem_type(metadata, sample_rows, target_column):
    """
    Uses metadata, first 5 rows, and selected target column
    to identify whether the problem is classification or regression.
    """

    prompt = problem_type_prompt(
        metadata=metadata,
        sample_rows=sample_rows,
        target_column=target_column
    )

    response = ask_llm(prompt)

    return response