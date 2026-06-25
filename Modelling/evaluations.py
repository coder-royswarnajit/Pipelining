from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CLASSIFICATION_RANKING_METRICS = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1_score": True,
}

REGRESSION_RANKING_METRICS = {
    "mae": False,
    "mse": False,
    "rmse": False,
    "r2_score": True,
}


def get_default_ranking_metric(problem_type):
    problem_type = (problem_type or "").lower()

    if problem_type == "regression":
        return "r2_score"

    return "f1_score"


def normalize_ranking_metric(metric, problem_type):
    if metric is None:
        return None

    normalized = str(metric).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "f1": "f1_score",
        "f1score": "f1_score",
        "r2": "r2_score",
        "r_squared": "r2_score",
        "r2score": "r2_score",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
    }

    normalized = aliases.get(normalized, normalized)
    allowed = (
        CLASSIFICATION_RANKING_METRICS
        if problem_type == "classification"
        else REGRESSION_RANKING_METRICS
    )

    if normalized in allowed:
        return normalized

    return None


def resolve_ranking_metric(metric, problem_type):
    validated = normalize_ranking_metric(metric, problem_type)
    if validated is None:
        validated = get_default_ranking_metric(problem_type)

    higher_is_better = (
        CLASSIFICATION_RANKING_METRICS.get(validated)
        if problem_type == "classification"
        else REGRESSION_RANKING_METRICS.get(validated)
    )

    return {
        "metric": validated,
        "higher_is_better": higher_is_better,
    }


def sort_models_by_ranking_metric(model_results, ranking_metric, problem_type):
    """
    Sorts successful model results by the chosen ranking metric.
    Failed models remain at the end in their original order.
    """

    resolved = resolve_ranking_metric(ranking_metric, problem_type)
    metric = resolved["metric"]
    reverse = resolved["higher_is_better"]

    successful_models = {
        name: result
        for name, result in model_results.items()
        if isinstance(result, dict) and result.get("status") == "success"
    }
    failed_models = {
        name: result
        for name, result in model_results.items()
        if not (isinstance(result, dict) and result.get("status") == "success")
    }

    sorted_successful = dict(
        sorted(
            successful_models.items(),
            key=lambda item: item[1].get(metric, float("-inf") if reverse else float("inf")),
            reverse=reverse,
        )
    )

    return sorted_successful, failed_models, resolved


def evaluate_classification_model(y_test, y_pred):
    """
    Evaluates classification models.
    Works for binary and multiclass classification.
    """

    results = {"accuracy": round(accuracy_score(y_test, y_pred), 4), 
               "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
               "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
               "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
               "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()}

    return results


def evaluate_regression_model(y_test, y_pred):
    """
    Evaluates regression models.
    """

    mse = mean_squared_error(y_test, y_pred)

    results = {"mae": round(mean_absolute_error(y_test, y_pred), 4),
               "mse": round(mse, 4),
               "rmse": round(mse ** 0.5, 4), 
               "r2_score": round(r2_score(y_test, y_pred), 4)}

    return results


def evaluate_model(y_test, y_pred, problem_type):
    """
    Evaluates model based on problem type.
    """

    problem_type = problem_type.lower()

    if problem_type == "classification":
        return evaluate_classification_model(y_test, y_pred)

    elif problem_type == "regression":
        return evaluate_regression_model(y_test, y_pred)

    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'")