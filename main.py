from step_01_dataset import create_synthetic_people_dataset

from step_02_exploration import analyze_dataset, analyze_target, visualize_dataset

from step_03_data_cleaning import clean_dataset

from step_04_data_splitting import split_data


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


if __name__ == "__main__":
    main()