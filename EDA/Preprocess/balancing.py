from collections import Counter

from imblearn.over_sampling import SMOTE


def fit_balancer(
    y_train,
    imbalance_threshold=0.80,
    random_state=42
):
    """
    Determines whether SMOTE should be applied.

    Returns configuration only.
    """

    class_counts = Counter(y_train)

    total = len(y_train)

    majority_percent = (
        max(class_counts.values()) / total
    )

    minority_count = min(class_counts.values())

    use_smote = (
        majority_percent >= imbalance_threshold
        and minority_count > 1
    )

    k_neighbors = min(
        5,
        minority_count - 1
    ) if minority_count > 1 else None

    return {
        "use_smote": use_smote,
        "random_state": random_state,
        "k_neighbors": k_neighbors,
        "class_distribution": dict(class_counts)
    }
    
def transform_balancer(
    X_train,
    y_train,
    balancing_config
):
    """
    Applies SMOTE only on training data.
    """

    if not balancing_config["use_smote"]:
        return X_train, y_train

    smote = SMOTE(
        random_state=balancing_config["random_state"],
        k_neighbors=balancing_config["k_neighbors"]
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled