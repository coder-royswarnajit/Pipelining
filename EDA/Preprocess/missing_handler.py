import pandas as pd
from sklearn.impute import SimpleImputer

from Profiling.type_detection import detect_column_types
from Profiling.skewness import detect_skewness


def handle_missing_values(df):
    """
    Full-dataset missing value handling.
    Use this for EDA/report preprocessing only.
    Do not use this before model train-test split.
    """

    df = df.copy()

    type_info = detect_column_types(df)
    skew_info = detect_skewness(df)

    for col in df.columns:

        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100

        if missing_count == 0:
            continue

        detected_type = type_info[col]["detected_type"]

        if missing_percent > 50:
            df.drop(columns=[col], inplace=True)
            continue

        if detected_type == "continuous":
            skewness = 0
            
            if col in skew_info:
                skewness = skew_info[col]["skewness"]

            if abs(skewness) > 1:
                imputer = SimpleImputer(strategy="median")
                df[[col]] = imputer.fit_transform(df[[col]])
                

            else:
                imputer = SimpleImputer(strategy="mean")
                df[[col]] = imputer.fit_transform(df[[col]])
                

        elif detected_type in ["binary", "categorical", "categorical_numeric"]:
            imputer = SimpleImputer(strategy="most_frequent")
            df[[col]] = imputer.fit_transform(df[[col]])

        elif detected_type == "text":
            imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
            df[[col]] = imputer.fit_transform(df[[col]])

        elif detected_type == "identifier":
            pass
        
        elif detected_type == "datetime":
            df[col] = df[col].ffill()

        else:
            pass
    

    return df


def fit_missing_imputers(X_train):
    """
    Fits missing value imputers only on X_train.
    """

    imputers = {}

    for col in X_train.columns:

        missing_count = X_train[col].isnull().sum()

        def fit_missing_imputers(X_train):  
            imputers = {}

            for col in X_train.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    imputer = SimpleImputer(strategy="median")

                else:
                    imputer = SimpleImputer(strategy="most_frequent")

                imputer.fit(X_train[[col]])
                imputers[col] = imputer

            return imputers

        if pd.api.types.is_numeric_dtype(X_train[col]):
            imputer = SimpleImputer(strategy="median")
        else:
            imputer = SimpleImputer(strategy="most_frequent")

        imputer.fit(X_train[[col]])

        imputers[col] = imputer

    return imputers


def transform_missing_values(X, imputers):
    """
    Applies already-fitted imputers to X_train or X_test.
    """

    X = X.copy()

    for col, imputer in imputers.items():

        if col in X.columns:
            X[[col]] = imputer.transform(X[[col]])

    return X