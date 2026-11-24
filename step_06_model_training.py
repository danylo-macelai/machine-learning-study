from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, y_train):

    models = {
        "linear_regression": LinearRegression(),

        "decision_tree": DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
