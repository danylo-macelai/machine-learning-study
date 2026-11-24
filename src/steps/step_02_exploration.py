import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_target(df: pd.DataFrame, limits: dict) -> dict:
    return {
        "dimensions": df.shape,
        "missing_values": df.isnull().sum(),
        "duplicates": df.duplicated(
            subset=df.columns.drop("ID"),
            keep="first"
        ).sum(),
        "outliers": identify_outliers(df, limits)
    }


def analyze_dataset(df: pd.DataFrame, limits: dict) -> dict:
    series = df["spending"]

    return {
        "target_distribution": {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "unique_values": series.nunique(),
            "frequency": series.value_counts().sort_index()
        },
        "data_types": df.dtypes,
        **analyze_target(df, limits)
    }


def identify_outliers(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    results = []

    for column, (min_limit, max_limit) in limits.items():
        mask = (df[column] < min_limit) | (df[column] > max_limit)

        results.append({
            "column": column,
            "min_limit": min_limit,
            "max_limit": max_limit,
            "values": df.loc[mask, column].tolist(),
            "outliers": int(mask.sum())
        })

    return pd.DataFrame(results)


def visualize_dataset(df: pd.DataFrame):
    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=df,
        x="spending",
        bins=10,
        kde=True
    )

    plt.title("Spending Distribution")
    plt.xlabel("Total Spending")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()