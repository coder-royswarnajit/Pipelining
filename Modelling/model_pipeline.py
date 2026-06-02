from Modelling.train_test_splitter import split_data
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

    X_train, X_test, y_train, y_test = split_data(df=df, target_column=target_column, problem_type=problem_type, test_size=0.2, random_state=42)
    X_train, X_test = preprocess_train_test_for_model(X_train=X_train, X_test=X_test)

    if problem_type == "classification":
        models = get_classification_models(random_state=42)

    elif problem_type == "regression":
        models = get_regression_models(random_state=42)

    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'")

    model_results = {}

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            evaluation = evaluate_model(y_test=y_test, y_pred=y_pred, problem_type=problem_type)

            model_results[model_name] = {"status": "success", "evaluation": evaluation}

        except Exception as e:
            model_results[model_name] = {"status": "failed", "error": str(e)}

    return model_results