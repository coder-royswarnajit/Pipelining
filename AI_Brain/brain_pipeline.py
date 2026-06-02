from AI_Brain.target_detector import analyze_target_columns
from AI_Brain.problem_type import analyze_problem_type
from AI_Brain.plot_recommender import recommend_plots
from AI_Brain.eda_explainer import explain_eda_results


def extract_target_from_analysis(target_column_analysis):
    """
    Extracts target column from text like:
    Best Target Column:
    column_name
    """

    text = str(target_column_analysis).replace("\\n", "\n")

    if "Best Target Column:" in text:
        target_column = (
            text
            .split("Best Target Column:")[-1]
            .strip()
            .split("\n")[0]
            .strip()
        )

        return target_column

    return None


def extract_problem_type_from_analysis(problem_type_analysis):
    """
    Extracts classification/regression from problem type response.
    """

    text = str(problem_type_analysis).lower()

    if "classification" in text:
        return "classification"

    if "regression" in text:
        return "regression"

    return None


def run_ai_brain_pipeline(
    metadata,
    df,
    eda_results=None,
    run_plot_recommendation=True,
    run_eda_explanation=True,
    plot_summaries=None
):
    """
    Runs AI analysis using EDA metadata and first 5 rows of dataset.

    Includes:
    - target column analysis
    - problem type analysis
    - optional plot recommendation
    - optional EDA explanation using graph summaries
    """

    sample_rows = df.head(5).to_dict(orient="records")

    target_column_analysis = analyze_target_columns(
        metadata=metadata,
        sample_rows=sample_rows
    )

    problem_type_analysis = analyze_problem_type(
        metadata=metadata,
        sample_rows=sample_rows,
        target_column=target_column_analysis
    )

    ai_results = {
        "target_column_analysis": target_column_analysis,
        "problem_type_analysis": problem_type_analysis
    }

    target_column = extract_target_from_analysis(
        target_column_analysis
    )

    problem_type = extract_problem_type_from_analysis(
        problem_type_analysis
    )

    if run_plot_recommendation:
        plot_recommendations = recommend_plots(
            metadata=metadata,
            df=df,
            target_column=target_column,
            problem_type=problem_type
        )

        ai_results["plot_recommendations"] = plot_recommendations

    if run_eda_explanation and eda_results is not None:
        eda_explanation = explain_eda_results(
            eda_results=eda_results,
            target_column=target_column,
            problem_type=problem_type,
            plot_recommendations=ai_results.get(
                "plot_recommendations",
                eda_results.get("plot_recommendations", [])
            ),
            plot_summaries=plot_summaries
        )

        ai_results["eda_explanation"] = eda_explanation

    return ai_results