import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed

from Modelling.train_test_splitter import split_data
from Modelling.classification_model import get_classification_models
from Modelling.regression_model import get_regression_models
from Modelling.evaluations import evaluate_model

from EDA.Preprocess.preprocess_pipeline import preprocess_train_test_for_model


def drop_identifier_columns(X, metadata=None):
    """
    Removes identifier columns using only saved metadata.
    """

    X = X.copy()

    identifier_columns = []

    for col in X.columns:
        column_metadata = metadata.get(col, {}) if isinstance(metadata, dict) else {}
        detected_type = str(column_metadata.get("detected_type", "unknown")).lower()
        cardinality_type = str(column_metadata.get("cardinality_info", {}).get("cardinality_type", "")).lower()

        if detected_type == "identifier" or cardinality_type == "identifier-like":
            identifier_columns.append(col)
            continue

    if identifier_columns:
        X = X.drop(columns=identifier_columns, errors="ignore")

    return X, identifier_columns


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


def run_model_pipeline(df, target_column, problem_type, metadata=None):
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

    X_train_p, X_test_p, y_train_p, preprocessing_report = (preprocess_train_test_for_model(X_train=X_train, X_test=X_test, y_train=y_train, problem_type=problem_type, metadata=metadata, balancing=True,))

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
        parallel_results = Parallel(n_jobs=-1)(delayed(train_model)(model_name, model, preprocessed_data, problem_type,) for model_name, model in models.items())

        model_results = dict(parallel_results)

    except Exception as e:
        model_results = {"pipeline": {"status": "failed", "error": str(e)}}

        import traceback

        print("\n===== PARALLEL ERROR =====")
        traceback.print_exc()
        print("==========================\n")
    

    successful_models = {name: result for name, result in model_results.items() if result["status"] == "success"}
    failed_models = {name: result for name, result in model_results.items() if result["status"] == "failed"}

    if problem_type == "classification":
        successful_models = dict(sorted(successful_models.items(), key=lambda x: x[1]["f1_score"], reverse=True))
        
    else:
        successful_models = dict(sorted(successful_models.items(), key=lambda x: x[1]["r2_score"], reverse=True))

    model_results = {**successful_models, **failed_models}

    return model_results