from AI_Brain.llm_client import ask_llm
from AI_Brain.target_detector import analyze_target_columns
from AI_Brain.problem_type import analyze_problem_type


def run_ai_brain_pipeline(metadata, df):
    """
    Runs AI analysis using EDA metadata and first 5 rows of dataset.
    """

    sample_rows = df.head(5).to_dict(orient="records")

    print("\nSTARTING AI BRAIN ANALYSIS")

    # Target column analysis
    target_column_analysis = analyze_target_columns(metadata=metadata, sample_rows=sample_rows)

    # Problem type analysis
    problem_type_analysis = analyze_problem_type(metadata=metadata, sample_rows=sample_rows, target_column=target_column_analysis)


    ai_results = {"target_column_analysis": target_column_analysis,
                  "problem_type_analysis": problem_type_analysis}

    print("\nAI BRAIN ANALYSIS COMPLETED")

    return ai_results