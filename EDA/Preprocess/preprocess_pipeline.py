from Preprocess.outlier_handling import fit_outlier_bounds
from Preprocess.outlier_handling import transform_outliers

from Preprocess.missing_handler import fit_missing_imputers
from Preprocess.missing_handler import transform_missing_values

from Preprocess.encoding import fit_encoders
from Preprocess.encoding import transform_encoding

from Preprocess.scaling import fit_scaler
from Preprocess.scaling import transform_scaling

from Preprocess.identifier_removal import fit_identifier_removal
from Preprocess.identifier_removal import transform_identifier_removal

from Preprocess.correlation_filter import fit_correlation_filter
from Preprocess.correlation_filter import transform_correlation_filter

from Preprocess.variance_filter import fit_variance_filter
from Preprocess.variance_filter import transform_variance_filter

from Preprocess.balancing import fit_balancer
from Preprocess.balancing import transform_balancer

from Preprocess.text_removal import fit_text_removal
from Preprocess.text_removal import transform_text_removal

from Preprocess.pca import fit_pca
from Preprocess.pca import transform_pca


def preprocess_train_test_for_model(
    X_train,
    X_test,
    problem_type=None,
    y_train=None,
    metadata=None,
    imputation_recommendations=None,
    outliers=True,
    missing=True,
    encoding=True,
    correlation=True,
    variance=True,
    balancing=True,
    scaling=True,
    pca=True,
    **kwargs,
):
    """
    Leakage-safe preprocessing for modelling.

    Fits preprocessing only on X_train.
    Applies the same transformations to X_train and X_test.
    """


    if metadata is None:
        metadata = kwargs.get("column_info", {})

    X_train = X_train.copy()
    X_test = X_test.copy()
    
    preprocessing_report = {}
    if metadata:
        identifier_config = fit_identifier_removal(metadata)

        X_train = transform_identifier_removal(X_train, identifier_config)
        X_test = transform_identifier_removal(X_test,identifier_config)

        preprocessing_report["dropped_identifier_columns"] = identifier_config.get("identifier_columns", [])
        
    text_config = fit_text_removal(metadata)

    X_train = transform_text_removal(X_train, text_config)
    X_test = transform_text_removal(X_test, text_config)
    
    preprocessing_report["dropped_text_columns"] = (text_config.get("text_columns", []))

    
    if missing:
        imputers = fit_missing_imputers(X_train, metadata=metadata, imputation_recommendations=imputation_recommendations)

        X_train = transform_missing_values(X_train, imputers)
        X_test = transform_missing_values(X_test, imputers)
        
    if (balancing and problem_type == "classification" and y_train is not None):
        balancing_config = fit_balancer(X_train, y_train, metadata)
        X_train, y_train = transform_balancer(X_train, y_train, balancing_config)

        preprocessing_report["balancing"] = balancing_config
        
    if outliers:
        outlier_bounds = fit_outlier_bounds(X_train)

        X_train = transform_outliers(X_train, outlier_bounds)
        X_test = transform_outliers(X_test, outlier_bounds)

    
    if scaling:
        scaler, numeric_cols = fit_scaler(X_train)

        X_train = transform_scaling(X_train, scaler, numeric_cols)
        X_test = transform_scaling(X_test, scaler, numeric_cols)

    if encoding:
        encoders = fit_encoders(X_train, metadata=metadata)

        X_train = transform_encoding(X_train, encoders)
        X_test = transform_encoding(X_test, encoders)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    if correlation:
        correlation_config = fit_correlation_filter(X_train, threshold=0.95)

        X_train = transform_correlation_filter(X_train, correlation_config)
        X_test = transform_correlation_filter(X_test, correlation_config)
        
        preprocessing_report["dropped_correlated_columns"] = correlation_config.get("dropped_columns",[])
    
    if variance:
        variance_config = fit_variance_filter(X_train)

        X_train = transform_variance_filter(X_train, variance_config)
        X_test = transform_variance_filter(X_test,variance_config)
        
        preprocessing_report["dropped_low_variance_columns"] = variance_config.get("dropped_columns", [])
    
    if pca:

        pca_config = fit_pca(X_train, feature_threshold=100, variance_retained=0.95)

        X_train = transform_pca(X_train, pca_config)
        X_test = transform_pca(X_test, pca_config)

        preprocessing_report["pca"] = {
            "applied": pca_config.get("apply_pca", False),
            "original_features": pca_config.get("original_features"),
            "reduced_features": pca_config.get("reduced_features"),
            "variance_retained": round(pca_config.get("variance_retained", 0), 4)}
    

    train_nan_cols = X_train.columns[X_train.isnull().any()].tolist()
    test_nan_cols = X_test.columns[X_test.isnull().any()].tolist()

    if train_nan_cols:
        raise ValueError(f"NaN values remain in X_train columns: {train_nan_cols}")

    if test_nan_cols:
        raise ValueError(f"NaN values remain in X_test columns: {test_nan_cols}")
        
    return X_train, X_test, y_train, preprocessing_report