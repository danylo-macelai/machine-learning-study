import os
import joblib


def dump_model(models):
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        path = f"models/{name}_model.pkl"
        joblib.dump(model, path)


def load_model(name):
    path = f"models/{name}_model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    return model
