from Preprocess.outlier_handling import handle_outliers
from Preprocess.outlier_handling import fit_outlier_bounds
from Preprocess.outlier_handling import transform_outliers

from Preprocess.missing_handler import handle_missing_values
from Preprocess.missing_handler import fit_missing_imputers
from Preprocess.missing_handler import transform_missing_values

from Preprocess.encoding import encode_features
from Preprocess.encoding import fit_encoders
from Preprocess.encoding import transform_encoding

from Preprocess.scaling import fit_scaler
from Preprocess.scaling import transform_scaling


def run_preprocessing_pipeline(df, target_column=None, outliers=True, missing=True, encoding=True):
    """
    Full-dataset preprocessing.
    Use this for EDA/report/preview only.
    Do not use this for final model training.
    """


    if target_column is not None:
        pass
    
    if outliers:
        df = handle_outliers(
            df,
            target_column=target_column
        )

    if missing:
        df = handle_missing_values(df)

    if encoding:
        df = encode_features(
            df,
            target_column=target_column
        )


    return df


def preprocess_train_test_for_model(X_train, X_test, outliers=True, missing=True, encoding=True, scaling=True):
    """
    Leakage-safe preprocessing for modelling.

    Fits preprocessing only on X_train.
    Applies the same transformations to X_train and X_test.
    """


    X_train = X_train.copy()
    X_test = X_test.copy()

    if outliers:
        outlier_bounds = fit_outlier_bounds(X_train)

        X_train = transform_outliers(X_train, outlier_bounds)
        X_test = transform_outliers(X_test, outlier_bounds)

    if missing:
        imputers = fit_missing_imputers(X_train)

        X_train = transform_missing_values(X_train, imputers)
        X_test = transform_missing_values(X_test, imputers)

    if encoding:
        encoders = fit_encoders(X_train)

        X_train = transform_encoding(X_train, encoders)
        X_test = transform_encoding(X_test, encoders)

        # Important after one-hot encoding
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

    if scaling:
        scaler, numeric_cols = fit_scaler(X_train)

        X_train = transform_scaling(X_train, scaler, numeric_cols)
        X_test = transform_scaling(X_test, scaler, numeric_cols)

    return X_train, X_test