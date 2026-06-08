from sklearn.metrics import confusion_matrix

from Modelling.train_test_splitter import get_cv_strategy
from Modelling.classification_model import get_classification_models
from Modelling.regression_model import get_regression_models
from Modelling.evaluations import evaluate_model

from EDA.Preprocess.preprocess_pipeline import preprocess_train_test_for_model

import numpy as np


def run_model_pipeline(df, target_column, problem_type):
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

    X = df.drop(columns=[target_column])
    y = df[target_column]

    cv = get_cv_strategy(problem_type=problem_type, n_splits=5, n_repeats=3, random_state=42)

    if problem_type == "classification":
        models = get_classification_models(random_state=42)
    elif problem_type == "regression":
        models = get_regression_models(random_state=42)
    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'")

    model_results = {}

    for model_name, model in models.items():
        feature_importance_sum = None
        coefficient_sum = None
        num_folds = 0

        feature_importance = None
        coefficients = None

        try:
            fold_results = []
            all_y_true = []
            all_y_pred = []

            for train_idx, test_idx in cv.split(X, y):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                X_train, X_test = preprocess_train_test_for_model(X_train=X_train, X_test=X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                evaluation = evaluate_model(y_test=y_test, y_pred=y_pred, problem_type=problem_type)
                fold_results.append(evaluation)

                all_y_true.extend(y_test.tolist())
                all_y_pred.extend(y_pred.tolist())

                num_folds += 1

                if hasattr(model, "feature_importances_"):
                    current_importance = model.feature_importances_

                    if feature_importance_sum is None:
                        feature_importance_sum = current_importance.copy()
                    else:
                        feature_importance_sum += current_importance

                if hasattr(model, "coef_"):
                    current_coef = model.coef_

                    if current_coef.ndim > 1:
                        current_coef = np.mean(current_coef, axis=0)

                    if coefficient_sum is None:
                        coefficient_sum = current_coef.copy()
                    else:
                        coefficient_sum += current_coef

        
            if feature_importance_sum is not None:
                avg_importance = feature_importance_sum / num_folds
                feature_importance = {feature: round(score, 6) 
                                      for feature, score in zip(X_train.columns, avg_importance)}

            if coefficient_sum is not None:
                avg_coefficients = coefficient_sum / num_folds
                coefficients = {feature: round(score, 6) 
                                for feature, score in zip(X_train.columns, avg_coefficients)}

            if problem_type == "classification":
                accuracy_scores = [fold["accuracy"] for fold in fold_results]
                precision_scores = [fold["precision"] for fold in fold_results]
                recall_scores = [fold["recall"] for fold in fold_results]
                f1_scores = [fold["f1_score"] for fold in fold_results]
                cm = confusion_matrix(all_y_true, all_y_pred)

                model_results[model_name] = {
                                                "status": "success",
                                                "model": model,
                                                "X_train": X_train,
                                                "accuracy_mean": round(np.mean(accuracy_scores), 4),
                                                "accuracy_std": round(np.std(accuracy_scores), 4),
                                                "precision_mean": round(np.mean(precision_scores), 4),
                                                "precision_std": round(np.std(precision_scores), 4),
                                                "recall_mean": round(np.mean(recall_scores), 4),
                                                "recall_std": round(np.std(recall_scores), 4),
                                                "f1_score_mean": round(np.mean(f1_scores), 4),
                                                "f1_score_std": round(np.std(f1_scores), 4),
                                                "confusion_matrix": cm.tolist(),
                                                "feature_importance": feature_importance,
                                                "coefficients": coefficients,}
                
            else:
                mae_scores = [fold["mae"] for fold in fold_results]
                mse_scores = [fold["mse"] for fold in fold_results]
                rmse_scores = [fold["rmse"] for fold in fold_results]
                r2_scores = [fold["r2_score"] for fold in fold_results]

                model_results[model_name] = {
                                                "status": "success",
                                                "model": model,
                                                "X_train": X_train,
                                                "mae_mean": round(np.mean(mae_scores), 4),
                                                "mae_std": round(np.std(mae_scores), 4),
                                                "mse_mean": round(np.mean(mse_scores), 4),
                                                "mse_std": round(np.std(mse_scores), 4),
                                                "rmse_mean": round(np.mean(rmse_scores), 4),
                                                "rmse_std": round(np.std(rmse_scores), 4),
                                                "r2_score_mean": round(np.mean(r2_scores), 4),
                                                "r2_score_std": round(np.std(r2_scores), 4),
                                                "feature_importance": feature_importance,
                                                "coefficients": coefficients,
                                                "actual_values": all_y_true,
                                                "predicted_values": all_y_pred,}

        except Exception as e:
            model_results[model_name] = {"status": "failed", "error": str(e)}

    successful_models = {name: result for name, result in model_results.items() if result["status"] == "success"}
    failed_models = {name: result for name, result in model_results.items() if result["status"] == "failed"}

    if problem_type == "classification":
        successful_models = dict(sorted(successful_models.items(), key=lambda x: x[1]["f1_score_mean"], reverse=True))
        
    else:
        successful_models = dict(sorted(successful_models.items(), key=lambda x: x[1]["r2_score_mean"], reverse=True))

    model_results = {**successful_models, **failed_models}

    return model_results