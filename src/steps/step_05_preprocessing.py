from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)


def create_preprocessor():

    # Feature groups
    minmax_features = ["age"]
    standard_features = ["income", "dependents"]
    onehot_features = [
        "job",
        "marital",
        "region"
    ]
    ordinal_features = ["education"]

    # Numerical preprocessing
    age_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Nominal categorical preprocessing
    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Ordinal categorical preprocessing
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Medio", "Superior", "Pos-graduacao"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("age", age_pipeline, minmax_features),
            ("numerical", numerical_pipeline, standard_features),
            ("onehot", onehot_pipeline, onehot_features),
            ("ordinal", ordinal_pipeline, ordinal_features)
        ],
        remainder="drop"
    )

    return preprocessor
