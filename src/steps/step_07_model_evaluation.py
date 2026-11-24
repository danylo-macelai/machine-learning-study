import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_models(models, X_val, y_val):

    results = []

    for name, model in models.items():

        predictions = model.predict(X_val)

        results.append({
            "model": name,
            "mae": mean_absolute_error(y_val, predictions),
            "mse": mean_squared_error(y_val, predictions),
            "rmse": mean_squared_error(y_val, predictions) ** 0.5,
            "r2": r2_score(y_val, predictions)
        })

    return pd.DataFrame(results).sort_values(
        by="rmse"
    ).reset_index(drop=True)
