import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def test_best_model(models, X_test, y_test):

    model = models["random_forest"]

    predictions = model.predict(X_test)

    results = {
        "model": "random_forest",
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions)
    }

    comparison = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions.round(2)
    })

    return results, comparison
