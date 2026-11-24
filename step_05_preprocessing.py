import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)


def preprocess_data(X_train, X_val, X_test):

    # Feature groups
    minmax_features = ["age"]
    standard_features = ["income", "dependents"]
    onehot_features = ["job", "region"]
    ordinal_features = ["education"]
    
    # Imputation
    imputer = SimpleImputer(strategy="median")

    X_train[standard_features] = imputer.fit_transform(
        X_train[standard_features]
    )
    X_val[standard_features] = imputer.transform(
        X_val[standard_features]
    )
    X_test[standard_features] = imputer.transform(
        X_test[standard_features]
    )

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()

    X_train[minmax_features] = minmax_scaler.fit_transform(
        X_train[minmax_features]
    )
    X_val[minmax_features] = minmax_scaler.transform(
        X_val[minmax_features]
    )
    X_test[minmax_features] = minmax_scaler.transform(
        X_test[minmax_features]
    )

    # StandardScaler
    standard_scaler = StandardScaler()

    X_train[standard_features] = standard_scaler.fit_transform(
        X_train[standard_features]
    )
    X_val[standard_features] = standard_scaler.transform(
        X_val[standard_features]
    )
    X_test[standard_features] = standard_scaler.transform(
        X_test[standard_features]
    )

    # LabelEncoder
    label_encoder = LabelEncoder()

    X_train["marital"] = label_encoder.fit_transform(
        X_train["marital"]
    )

    X_val["marital"] = label_encoder.transform(
        X_val["marital"]
    )

    X_test["marital"] = label_encoder.transform(
        X_test["marital"]
    )

    # OneHotEncoder
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_encoded = onehot_encoder.fit_transform(
        X_train[onehot_features]
    )
    val_encoded = onehot_encoder.transform(
        X_val[onehot_features]
    )
    test_encoded = onehot_encoder.transform(
        X_test[onehot_features]
    )

    onehot_columns = onehot_encoder.get_feature_names_out(
        onehot_features
    )

    X_train[onehot_columns] = train_encoded
    X_val[onehot_columns] = val_encoded
    X_test[onehot_columns] = test_encoded

    X_train.drop(columns=onehot_features, inplace=True)
    X_val.drop(columns=onehot_features, inplace=True)
    X_test.drop(columns=onehot_features, inplace=True)

    # OrdinalEncoder
    ordinal_encoder = OrdinalEncoder(
        categories=[["Medio", "Superior", "Pos-graduacao"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train[ordinal_features] = ordinal_encoder.fit_transform(
        X_train[ordinal_features]
    )
    X_val[ordinal_features] = ordinal_encoder.transform(
        X_val[ordinal_features]
    )
    X_test[ordinal_features] = ordinal_encoder.transform(
        X_test[ordinal_features]
    )

    return X_train, X_val, X_test
