from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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