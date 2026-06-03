from Modelling.train_test_splitter import get_cv_strategy
from Modelling.classification_model import get_classification_models
from Modelling.regression_model import get_regression_models
from Modelling.evaluations import evaluate_model

from EDA.Preprocess.preprocess_pipeline import preprocess_train_test_for_model


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
        try:

            fold_results = []

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

            

            if problem_type == "classification":
                avg_accuracy = sum(fold["accuracy"] for fold in fold_results) / len(fold_results)
                avg_precision = sum(fold["precision"] for fold in fold_results) / len(fold_results)
                avg_recall = sum(fold["recall"] for fold in fold_results) / len(fold_results)
                avg_f1 = sum(fold["f1_score"] for fold in fold_results) / len(fold_results)

                model_results[model_name] = {"status": "success",
                                             "accuracy": round(avg_accuracy, 4),
                                             "precision": round(avg_precision, 4),
                                             "recall": round(avg_recall, 4),
                                             "f1_score": round(avg_f1, 4)}

            else:
                avg_mae = sum(fold["mae"] for fold in fold_results) / len(fold_results)
                avg_mse = sum(fold["mse"] for fold in fold_results) / len(fold_results)
                avg_rmse = sum(fold["rmse"] for fold in fold_results) / len(fold_results)
                avg_r2 = sum(fold["r2_score"] for fold in fold_results) / len(fold_results)

                model_results[model_name] = {"status": "success",
                                             "mae": round(avg_mae, 4),
                                             "mse": round(avg_mse, 4),
                                             "rmse": round(avg_rmse, 4),
                                             "r2_score": round(avg_r2, 4)}

        except Exception as e:
            model_results[model_name] = {"status": "failed", "error": str(e)}
    
    
    successful_models={name: result for name, result in model_results.items() if result["status"] == "success"}
    failed_models={name: result for name, result in model_results.items() if result["status"] == "failed"}

    if problem_type == "classification":
        successful_models = dict(sorted(successful_models.items(), key=lambda x: x[1]["f1_score"], reverse=True))

    else:
        successful_models = dict(sorted(successful_models.items(), key=lambda x: x[1]["r2_score"], reverse=True))
    
    model_results = {**successful_models, **failed_models}
        
    return model_results