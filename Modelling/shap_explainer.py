import shap
import pandas as pd


def generate_shap_explanation(model, X_train, X_sample):

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_sample)

        return {"status": "success", "explainer": explainer, "shap_values": shap_values}

    except Exception as e:
        return {"status": "failed", "error": str(e)}