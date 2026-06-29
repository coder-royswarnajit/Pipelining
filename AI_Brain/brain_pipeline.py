from AI_Brain.plot_recommender import recommend_plots
from AI_Brain.eda_explainer import explain_eda_results
from AI_Brain.imputation_recommender import recommend_imputations
from AI_Brain.metric_recommender import recommend_ranking_metric


def run_ai_brain_pipeline(
    metadata,
    df,
    eda_results=None,
    run_plot_recommendation=True,
    run_eda_explanation=True,
    run_imputation_recommendation=True,
    run_metric_recommendation=True,
    plot_summaries=None,
    target_column=None, 
    problem_type=None) -> dict:
    """
    Runs AI analysis using EDA metadata and first 5 rows of dataset.

    Includes:
    - optional plot recommendation
    - optional EDA explanation using graph summaries
    """

    ai_results = {"target_column": target_column, "problem_type": problem_type,}

    if run_plot_recommendation:
        plot_recommendations = recommend_plots(metadata=metadata, df=df, target_column=target_column, problem_type=problem_type)
        ai_results["plot_recommendations"] = plot_recommendations
    
    if run_imputation_recommendation:
        imputation_recommendations = recommend_imputations(metadata=metadata)

        ai_results["imputation_recommendations"] = imputation_recommendations

    if run_metric_recommendation:
        ranking_metric_recommendation = recommend_ranking_metric(metadata=metadata, df=df, target_column=target_column, problem_type=problem_type,)
        ai_results["ranking_metric_recommendation"] = ranking_metric_recommendation

    if run_eda_explanation and eda_results is not None:
        eda_explanation = explain_eda_results(
            eda_results=eda_results,
            target_column=target_column,
            problem_type=problem_type,
            plot_recommendations=ai_results.get(
                "plot_recommendations",
                eda_results.get("plot_recommendations", [])), 
            plot_summaries=plot_summaries)

        ai_results["eda_explanation"] = eda_explanation

    return ai_results