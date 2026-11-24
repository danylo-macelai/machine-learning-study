import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates(
            subset=df.columns.drop("ID"),
            keep="first"
        )
        .copy()
        .reset_index(drop=True)
    )


def handle_outliers(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    df = df.copy()

    for column, (min_limit, max_limit) in limits.items():
        df[column] = df[column].clip(
            lower=min_limit,
            upper=max_limit
        )

    return df


def clean_dataset(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = handle_outliers(df, limits)

    return df