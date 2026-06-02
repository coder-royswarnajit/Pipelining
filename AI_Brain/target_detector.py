from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import target_column_prompt


def analyze_target_columns(metadata, sample_rows):
    """
    Uses metadata and first 5 rows to identify possible target columns.
    Returns a cleaner target-column analysis from the LLM.
    """

    prompt = target_column_prompt(
        metadata=metadata,
        sample_rows=sample_rows
    )

    response = ask_llm(prompt)

    cleaned_response = (
        response
        .replace("\\n", "\n")
        .replace("**", "")
        .strip()
    )

    return cleaned_response