from step_01_dataset import create_synthetic_people_dataset

from step_02_exploration import analyze_dataset, analyze_target, visualize_dataset

from step_03_data_cleaning import clean_dataset

from step_04_data_splitting import split_data

from step_05_preprocessing import preprocess_data

from step_06_model_training import train_models

from step_07_model_evaluation import evaluate_models


def main():
    limits = {
        "age": (18, 80),
        "income": (1000, 8000),
        "dependents": (0, 5),
        "spending": (1000, 4000)
    }

    print("\n========== STEP 01 DATASET ==========")
    df = create_synthetic_people_dataset()
    print(df.head(10))

    print("\n========== STEP 02 EXPLORATION ==========")
    analyze = analyze_dataset(df, limits);
    for k, v in analyze.items():
        print(f"\n{k}:\n{v}")
    visualize_dataset(df)

    print("\n========== STEP 03 DATA CLEANING ==========")
    df = clean_dataset(df, limits)
    analyze = analyze_target(df, limits)
    for k, v in analyze.items():
        print(f"\n{k}:\n{v}")

    print("\n========== STEP 04 DATA SPLITTING ==========")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("\n========== STEP 05 PREPROCESSING ==========")
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    print(X_train.head(10))

    print("\n========== STEP 06 MODEL TRAINING ==========")
    models = train_models(X_train, y_train)
    for name, model in models.items():
        print(f"{name}: trained successfully")

    print("\n========== STEP 07 MODEL EVALUATION ==========")
    results = evaluate_models(models, X_val, y_val)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()