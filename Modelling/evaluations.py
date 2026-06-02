from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def evaluate_classification_model(y_test, y_pred):
    """
    Evaluates classification models.
    Works for binary and multiclass classification.
    """

    results = {"accuracy": round(accuracy_score(y_test, y_pred), 4), 
               "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
               "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
               "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
               "classification_report": classification_report(y_test, y_pred, zero_division=0)}

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