from typing import Any, cast

import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed

from Modelling.train_test_splitter import split_data
from Modelling.classification_model import get_classification_models
from Modelling.regression_model import get_regression_models
from Modelling.evaluations import evaluate_model, sort_models_by_ranking_metric
from Modelling.hyperparameter_tuning import tune_best_model
from Modelling.shap_analysis import generate_shap_analysis

from EDA.Preprocess.preprocess_pipeline import preprocess_train_test_for_model


def _get_best_successful_model_name(model_results):
    for model_name, result in model_results.items():
        if isinstance(result, dict) and result.get("status") == "success":
            return model_name

    return None


def train_model(model_name, model, training_data, problem_type):
    """
    Trains and evaluates one model on the given training data.
    
    """
    all_y_true = []
    all_y_pred = []
    last_x_train = None
    last_y_train = None

    
    X_train = training_data["X_train"]
    X_test = training_data["X_test"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]

    last_y_train = y_train
    last_x_train = X_train

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluation = evaluate_model(
        y_test=y_test,
        y_pred=y_pred,
        problem_type=problem_type,
    )

    all_y_true = y_test.tolist()
    all_y_pred = y_pred.tolist()

    if problem_type == "classification":
        cm = confusion_matrix(all_y_true, all_y_pred)

        result = {
            "status": "success",
            "model": model,
            "X_train": last_x_train,
            "y_train": last_y_train,
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1_score": evaluation["f1_score"],
            "confusion_matrix": cm.tolist()
        }
        
    else:

        result = {
            "status": "success",
            "model": model,
            "X_train": last_x_train,
            "y_train": last_y_train,
            "mae": evaluation["mae"],
            "mse": evaluation["mse"],
            "rmse": evaluation["rmse"],
            "r2_score": evaluation["r2_score"],
            "actual_values": all_y_true,
            "predicted_values": all_y_pred,
        }

    return model_name, result


def run_model_pipeline(
    df,
    target_column,
    problem_type,
    metadata=None,
    ranking_metric=None,
    ranking_metric_info=None,
    imputation_recommendations=None,
):
    """
    Runs the complete modelling pipeline.

    Steps:
    1. Split data
    2. Select models based on problem type
    3. Train each model
    4. Evaluate each model
    5. Return results
    """

    problem_type = problem_type.lower()

    df = df.copy()
    df = df.dropna(subset=[target_column])

    X_train, X_test, y_train, y_test=split_data(df=df, target_column=target_column, problem_type=problem_type, test_size=0.2, random_state=42)

    X_train_p, X_test_p, y_train_p, preprocessing_report = preprocess_train_test_for_model(X_train=X_train, X_test=X_test, y_train=y_train, problem_type=problem_type, metadata=metadata, balancing=True, imputation_recommendations=imputation_recommendations)

    preprocessed_data = {"X_train": X_train_p,
                          "X_test": X_test_p,
                          "y_train": y_train_p,
                          "y_test": y_test,}
    
    if problem_type == "classification":
        models = get_classification_models(random_state=42)

    elif problem_type == "regression":
        models = get_regression_models(random_state=42)
    
    print(preprocessing_report)

    try:
        parallel_results = cast(
            "list[tuple[str, dict[str, Any]]]",
            Parallel(n_jobs=-1)(delayed(train_model)(model_name, model, preprocessed_data, problem_type,) for model_name, model in models.items()),
        )

        model_results = dict(parallel_results)

    except Exception as e:
        model_results = {"pipeline": {"status": "failed", "error": str(e)}}

        import traceback

        print("\n===== PARALLEL ERROR =====")
        traceback.print_exc()
        print("==========================\n")
    

    successful_models, failed_models, ranking_info = sort_models_by_ranking_metric(
        model_results=model_results,
        ranking_metric=ranking_metric,
        problem_type=problem_type,
    )

    model_results = {**successful_models, **failed_models}

    info = ranking_metric_info if isinstance(ranking_metric_info, dict) else {}
    model_results["_ranking_info"] = {
        **ranking_info,
        "reason": info.get("reason", ""),
        "source": info.get("source", "default" if not ranking_metric else "override"),
    }


    best_model_name = next(iter(successful_models), None)

    if best_model_name is not None:
        best_result = model_results.get(best_model_name, {})
        optimization_metric = ranking_info.get("metric")

        before_metrics = {
            key: best_result.get(key)
            for key in (
                ["accuracy", "precision", "recall", "f1_score"]
                if problem_type == "classification"
                else ["mae", "mse", "rmse", "r2_score"]
            )
            if key in best_result
        }

        optimization_result, tuned_model, tuned_metrics, tuned_predictions = tune_best_model(
            model_name=best_model_name,
            model=best_result.get("model"),
            X_train=X_train_p,
            y_train=y_train_p,
            X_test=X_test_p,
            y_test=y_test,
            problem_type=problem_type,
            optimization_metric=optimization_metric,
            before_metrics=before_metrics,
            n_trials=30,
        )

        best_result["hyperparameter_optimization"] = optimization_result
        model_results["_optimization_info"] = {
            "best_model": best_model_name,
            "status": optimization_result.get("status"),
            "optimization_metric": optimization_result.get("optimization_metric"),
            "best_trial": optimization_result.get("best_trial"),
            "best_cv_score": optimization_result.get("best_cv_score"),
            "error": optimization_result.get("error"),
            "reason": optimization_result.get("reason"),
        }

        if optimization_result.get("status") == "success":
            best_result["model"] = tuned_model

            for metric_name, metric_value in tuned_metrics.items():
                best_result[metric_name] = metric_value

            if problem_type == "classification" and tuned_predictions is not None:
                best_result["confusion_matrix"] = confusion_matrix(
                    y_test.tolist(),
                    tuned_predictions.tolist(),
                ).tolist()
            elif problem_type == "regression" and tuned_predictions is not None:
                best_result["actual_values"] = y_test.tolist()
                best_result["predicted_values"] = tuned_predictions.tolist()

            shap_analysis = generate_shap_analysis(
                model=tuned_model,
                X_train=X_train_p,
                X_test=X_test_p,
                model_name=best_model_name,
                problem_type=problem_type,
            )

            best_result["shap_analysis"] = shap_analysis

            if shap_analysis.get("status") == "success":
                try:
                    from AI_Brain.optimization_explainer import explain_optimized_model

                    best_result["optimization_business_explanation"] = explain_optimized_model(
                        model_name=best_model_name,
                        problem_type=problem_type,
                        optimization_result=optimization_result,
                        shap_analysis=shap_analysis,
                    )
                except Exception as e:
                    best_result["optimization_business_explanation"] = (
                        f"Optimization explanation unavailable: {str(e)}"
                    )

                try:
                    from AI_Brain.shap_explainer import explain_shap_results

                    best_result["shap_business_explanation"] = explain_shap_results(
                        shap_analysis=shap_analysis,
                        target_column=target_column,
                        problem_type=problem_type,
                        model_name=best_model_name,
                    )
                except Exception as e:
                    best_result["shap_business_explanation"] = (
                        f"SHAP business explanation unavailable: {str(e)}"
                    )

            model_results["_shap_info"] = {
                "best_model": best_model_name,
                "status": shap_analysis.get("status"),
                "error": shap_analysis.get("error"),
            }

    return model_results