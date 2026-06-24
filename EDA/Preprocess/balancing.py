from imblearn.over_sampling import SMOTENC
from collections import Counter

def fit_balancer(X_train, y_train, metadata, imbalance_threshold=0.80, random_state=42):
    class_counts = Counter(y_train)

    total = len(y_train)

    majority_percent = (max(class_counts.values()) / total)
    minority_count = min(class_counts.values())

    use_smote = (len(class_counts) > 1 and majority_percent >= imbalance_threshold and minority_count > 1)

    categorical_features = []

    for idx, col in enumerate(X_train.columns):

        detected_type = (metadata.get(col, {}).get("detected_type", "").lower())
        if detected_type in ["categorical", "binary", "ordinal"]:
            categorical_features.append(idx)

    return {
        "use_smote": use_smote,
        "random_state": random_state,
        "k_neighbors": min(5, minority_count - 1),
        "categorical_features": categorical_features,
        "class_distribution": dict(class_counts)
    }


def transform_balancer(X_train, y_train, balancing_config):

    if not balancing_config["use_smote"]:
        return X_train, y_train

    smote = SMOTENC(
        categorical_features=balancing_config["categorical_features"],
        random_state=balancing_config["random_state"],
        k_neighbors=balancing_config["k_neighbors"])

    X_resampled, y_resampled = (smote.fit_resample(X_train, y_train))

    return X_resampled, y_resampled