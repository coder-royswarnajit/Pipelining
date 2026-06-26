import numpy as np
import pandas as pd


def _as_dataframe(data, columns=None):
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if data is None:
        return None

    array = np.asarray(data)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if columns is None:
        columns = [f"feature_{idx}" for idx in range(array.shape[1])]

    return pd.DataFrame(array, columns=columns)


def _sample_frame(df, max_rows=80, random_state=42):
    if df is None or len(df) == 0:
        return df

    if len(df) <= max_rows:
        return df.copy()

    return df.sample(n=max_rows, random_state=random_state)


def _to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (np.ndarray,)):
        return [_to_serializable(item) for item in value.tolist()]
    if pd.isna(value):
        return None
    return value


def _aggregate_shap_values(shap_values, feature_names):
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=(0, 2))
        mean_signed = values.mean(axis=(0, 2))
    elif values.ndim == 2:
        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
    else:
        raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    total_abs_shap = float(np.sum(mean_abs))

    feature_rows = []
    for feature, importance, signed_effect in zip(feature_names, mean_abs, mean_signed):
        importance_value = float(importance)
        importance_pct = (
            (importance_value / total_abs_shap) * 100
            if total_abs_shap > 0
            else 0.0
        )

        feature_rows.append(
            {
                "feature": str(feature),
                "importance_pct": round(float(importance_pct), 2),
                "contribution": f"{round(float(importance_pct), 1)}%",
                "mean_abs_shap": round(importance_value, 6),
                "mean_shap": round(float(signed_effect), 6),
            }
        )

    feature_rows.sort(key=lambda item: item["importance_pct"], reverse=True)
    return feature_rows


def _prediction_summary(model, X_sample, problem_type):
    summary = {}

    try:
        predictions = model.predict(X_sample)
        summary["prediction_sample"] = [
            _to_serializable(value)
            for value in np.asarray(predictions).ravel()[:10]
        ]
    except Exception:
        pass

    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_sample)
            probabilities = np.asarray(probabilities)
            if probabilities.ndim == 2:
                summary["mean_predicted_probability_by_class"] = [
                    round(float(value), 6)
                    for value in probabilities.mean(axis=0)
                ]
        except Exception:
            pass

    return summary


def generate_shap_analysis(
    model,
    X_train,
    X_test,
    model_name=None,
    problem_type=None,
    max_background_rows=80,
    max_explain_rows=80,
    top_n=10,
    random_state=42,
):
    """
    Generates compact SHAP information for the selected best model.

    The returned object is intentionally small so it can be displayed in the app
    and passed to the AI Brain without sending the full dataset.
    """

    try:
        import shap

        model_feature_names = getattr(model, "feature_names_in_", None)
        if model_feature_names is not None:
            model_feature_names = [str(col) for col in model_feature_names]

        X_train = _as_dataframe(X_train, columns=model_feature_names)
        train_columns = list(X_train.columns) if X_train is not None else None
        X_test = _as_dataframe(X_test, columns=train_columns)

        X_background = _sample_frame(
            X_train,
            max_rows=max_background_rows,
            random_state=random_state,
        )
        X_explain = _sample_frame(
            X_test,
            max_rows=max_explain_rows,
            random_state=random_state,
        )

        if X_background is None or X_explain is None:
            raise ValueError("Training and test data are required for SHAP analysis.")

        if X_background.empty or X_explain.empty:
            raise ValueError("Not enough rows available for SHAP analysis.")

        feature_names = [str(col) for col in X_explain.columns]
        problem_type = (problem_type or "").lower()

        if problem_type == "classification" and hasattr(model, "predict_proba"):
            prediction_function = model.predict_proba
            output_type = "predicted_probability"
        else:
            prediction_function = model.predict
            output_type = "prediction"
        
        masker = shap.maskers.Independent(X_background) #type: ignore
        explainer = shap.Explainer(
            prediction_function,
            masker,
            feature_names=feature_names,
        )
        shap_values = explainer(X_explain)

        feature_importance = _aggregate_shap_values(
            shap_values=shap_values,
            feature_names=feature_names,
        )

        
        print("\n===== SHAP FEATURES =====")

        for item in feature_importance[:5]:
            print(item)

        print("=========================\n")
        
        return {
            "status": "success",
            "model_name": model_name,
            "problem_type": problem_type,
            "output_type": output_type,
            "background_rows": int(len(X_background)),
            "explained_rows": int(len(X_explain)),
            "feature_count": int(len(feature_names)),
            "top_features": feature_importance[:top_n],
            "prediction_summary": _prediction_summary(
                model=model,
                X_sample=X_explain,
                problem_type=problem_type,
            ),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "model_name": model_name,
            "error": str(exc),
        }
