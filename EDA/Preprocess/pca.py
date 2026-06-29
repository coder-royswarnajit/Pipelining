from sklearn.decomposition import PCA
import pandas as pd


def fit_pca(X_train, feature_threshold=50, variance_retained=0.95):
    """
    Fit PCA only when feature count exceeds threshold.
    """

    n_features = X_train.shape[1]

    if n_features <= feature_threshold:
        return {"apply_pca": False, "original_features": n_features}

    pca = PCA(n_components=variance_retained, random_state=42)
    pca.fit(X_train)

    return {
        "apply_pca": True,
        "pca": pca,
        "original_features": n_features,
        "reduced_features": pca.n_components_,
        "variance_retained": float(
            pca.explained_variance_ratio_.sum()
        )
    }


def transform_pca(X, pca_config):
    """
    Transform dataset using fitted PCA.
    """

    if not pca_config.get("apply_pca"):
        return X

    pca = pca_config["pca"]
    transformed = pca.transform(X)

    columns = [f"PC_{i+1}" for i in range(transformed.shape[1])]

    return pd.DataFrame(transformed, columns=columns, index=X.index)