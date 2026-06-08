import joblib


def save_model(model, filepath):
    joblib.dump(model, filepath)