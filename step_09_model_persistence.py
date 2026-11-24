import os
import joblib


MODEL_DIR = "models"


def dump_model(models):
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        path = os.path.join("models", f"{name}_model.pkl")
        joblib.dump(model, path)
        print(f"Pipeline saved successfully: {path}")


def load_model(name):
    path = os.path.join("models", f"{name}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
