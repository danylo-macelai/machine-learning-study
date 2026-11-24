from step_01_dataset import prediction_tests

from step_09_model_persistence import load_model


def main():
    model_names = [
        "linear_regression",
        "decision_tree",
        "random_forest"
    ]

    for model_name in model_names:
        print(f"\n========== MODEL: {model_name.upper()} ==========")
        model = load_model(model_name)
        for scenario, data in prediction_tests.items():
            X = data.drop(
                columns=["ID"],
                errors="ignore"
            )
            predictions = model.predict(X)
            results = data.copy()
            results["predicted_spending"] = predictions.round(2)
            print(f"\nScenario: {scenario}")
            print(
                results[
                    ["ID", "predicted_spending"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
