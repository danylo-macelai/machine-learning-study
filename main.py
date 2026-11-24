from step_01_dataset import create_synthetic_people_dataset

from step_02_exploration import analyze_dataset, visualize_dataset

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


if __name__ == "__main__":
    main()