import time
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Modelling.evaluations import evaluate_model, resolve_ranking_metric


def _metric_scoring(metric, problem_type):
    problem_type = (problem_type or "").lower()

    if problem_type == "classification":
        return {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1_score": "f1_weighted",
        }.get(metric)

    return {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "r2_score": "r2",
    }.get(metric)


def _cv_splitter(problem_type):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return KFold(n_splits=5, shuffle=True, random_state=42)


def _space_logistic_regression(trial):
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
    penalty = "l2"
    if solver == "liblinear":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "solver": solver,
        "penalty": penalty,
        "max_iter": 2000,
    }


def _space_random_forest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_categorical(
            "max_depth",
            [None, 3, 5, 8, 12, 16, 24],
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        ),
    }


def _space_decision_tree(trial):
    return {
        "max_depth": trial.suggest_categorical(
            "max_depth",
            [None, 2, 3, 5, 8, 12, 16, 24],
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _space_decision_tree_regressor(trial):
    return {
        "max_depth": trial.suggest_categorical(
            "max_depth",
            [None, 2, 3, 5, 8, 12, 16, 24],
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["squared_error", "absolute_error", "friedman_mse"],
        ),
    }


def _space_svc(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    params = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 4)
    return params


def _space_svr(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    params = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "kernel": kernel,
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 4)
    return params


def _space_knn(trial, n_samples):
    max_neighbors = max(1, min(30, n_samples - 1))
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, max_neighbors),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    }


def _space_linear_regularized(trial, model):
    params = {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
    if isinstance(model, ElasticNet):
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
    return params


def _suggest_params(trial, model, problem_type, n_samples):
    if isinstance(model, LogisticRegression):
        return _space_logistic_regression(trial)

    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        return _space_random_forest(trial)

    if isinstance(model, DecisionTreeClassifier):
        return _space_decision_tree(trial)

    if isinstance(model, DecisionTreeRegressor):
        return _space_decision_tree_regressor(trial)

    if isinstance(model, SVC):
        return _space_svc(trial)

    if isinstance(model, SVR):
        return _space_svr(trial)

    if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
        return _space_knn(trial, n_samples=n_samples)

    if isinstance(model, (Ridge, Lasso, ElasticNet)):
        return _space_linear_regularized(trial, model)

    return None


def _trial_history(study):
    history = []
    for trial in study.trials:
        history.append(
            {
                "trial_number": int(trial.number),
                "value": None if trial.value is None else round(float(trial.value), 6),
                "params": dict(trial.params),
                "state": str(trial.state.name),
            }
        )
    return history


def _parameter_history(study):
    rows = []
    for trial in study.trials:
        for param_name, value in trial.params.items():
            rows.append(
                {
                    "trial_number": int(trial.number),
                    "parameter": str(param_name),
                    "value": value,
                }
            )
    return rows


def _comparison_rows(before_metrics, after_metrics, metric_names):
    rows = []
    for metric in metric_names:
        if metric in before_metrics and metric in after_metrics:
            before = before_metrics.get(metric)
            after = after_metrics.get(metric)
            higher_is_better = resolve_ranking_metric(metric, "classification")[
                "higher_is_better"
            ] if metric in {"accuracy", "precision", "recall", "f1_score"} else resolve_ranking_metric(metric, "regression")[
                "higher_is_better"
            ]
            improved = after > before if higher_is_better else after < before
            rows.append(
                {
                    "Metric": metric,
                    "Before Tuning": before,
                    "After Tuning": after,
                    "Improved": bool(improved),
                }
            )
    return rows


def _update_result_with_after_metrics(result, after_metrics, problem_type, y_test, y_pred):
    for metric, value in after_metrics.items():
        result[metric] = value

    if problem_type == "classification":
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    else:
        result["actual_values"] = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
        result["predicted_values"] = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)


def tune_best_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    problem_type,
    optimization_metric,
    before_metrics,
    n_trials=30,
):
    """
    Tunes only the already-ranked best model using training data CV.
    Test data is used only after tuning to report final performance.
    """

    try:
        import optuna
    except Exception as exc:
        return {
            "status": "failed",
            "model_name": model_name,
            "error": f"Optuna is not available: {exc}",
        }, model, before_metrics, None

    problem_type = (problem_type or "").lower()
    metric_info = resolve_ranking_metric(optimization_metric, problem_type)
    metric = metric_info["metric"]
    direction = "maximize" if metric_info["higher_is_better"] else "minimize"
    scoring = _metric_scoring(metric, problem_type)

    if scoring is None:
        return {
            "status": "failed",
            "model_name": model_name,
            "error": f"Unsupported optimization metric: {optimization_metric}",
        }, model, before_metrics, None

    probe_params = _suggest_params(
        trial=_NullTrial(),
        model=model,
        problem_type=problem_type,
        n_samples=len(X_train),
    )
    if probe_params is None:
        return {
            "status": "skipped",
            "model_name": model_name,
            "reason": "Selected model has no configured hyperparameter search space.",
        }, model, before_metrics, None

    cv = _cv_splitter(problem_type)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    start_time = time.time()

    def objective(trial):
        params = _suggest_params(
            trial=trial,
            model=model,
            problem_type=problem_type,
            n_samples=len(X_train),
        )
        candidate = clone(model)
        assert params is not None
        candidate.set_params(**params)
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        score = float(np.mean(scores))
        if metric in {"mae", "mse", "rmse"}:
            score = abs(score)
        return score

    try:
        study.optimize(objective, n_trials=n_trials)
        optimization_time = round(float(time.time() - start_time), 3)

        tuned_model = clone(model)
        tuned_model.set_params(**study.best_params)
        tuned_model.fit(X_train, y_train)

        y_pred = tuned_model.predict(X_test)
        after_metrics = evaluate_model(
            y_test=y_test,
            y_pred=y_pred,
            problem_type=problem_type,
        )

        metric_names = (
            ["accuracy", "precision", "recall", "f1_score"]
            if problem_type == "classification"
            else ["mae", "mse", "rmse", "r2_score"]
        )

        metadata = {
            "status": "success",
            "model_name": model_name,
            "optimization_metric": metric,
            "direction": direction,
            "best_trial": int(study.best_trial.number),
            "best_cv_score": round(float(study.best_value), 6),
            "n_trials": int(len(study.trials)),
            "optimization_time_seconds": optimization_time,
            "best_params": deepcopy(study.best_params),
            "history": _trial_history(study),
            "parameter_history": _parameter_history(study),
            "before_metrics": deepcopy(before_metrics),
            "after_metrics": deepcopy(after_metrics),
            "comparison": _comparison_rows(
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                metric_names=metric_names,
            ),
        }

        return metadata, tuned_model, after_metrics, y_pred

    except Exception as exc:
        return {
            "status": "failed",
            "model_name": model_name,
            "optimization_metric": metric,
            "direction": direction,
            "error": str(exc),
        }, model, before_metrics, None


class _NullTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_int(self, name, low, high, step=1):
        return low
