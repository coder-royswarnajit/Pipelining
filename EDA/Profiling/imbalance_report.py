import pandas as pd


def analyze_class_imbalance(target_series, imbalance_threshold=0.80):
    """
    Analyzes class imbalance for classification targets.

    Parameters
    ----------
    target_series : pd.Series
        Target column.

    imbalance_threshold : float
        Majority class threshold.
        Example:
        0.80 -> if one class exceeds 80%
        dataset is considered imbalanced.

    Returns
    -------
    dict

    Example Output
    --------------
    {
        "total_samples": 1000,
        "num_classes": 2,
        "class_distribution": {
            "0": {
                "count": 920,
                "percent": 92.0
            },
            "1": {
                "count": 80,
                "percent": 8.0
            }
        },
        "majority_class": "0",
        "majority_percent": 92.0,
        "imbalanced": True,
        "recommendation": "Consider SMOTE, class weights, or resampling."
    }
    """

    if target_series is None:
        return {
            "status": "error",
            "message": "Target series is None."
        }

    target_series = pd.Series(target_series)

    total_samples = len(target_series)

    if total_samples == 0:
        return {
            "status": "error",
            "message": "Target series is empty."
        }

    value_counts = target_series.value_counts(dropna=False)

    distribution = {}

    for cls, count in value_counts.items():

        percent = round(
            (count / total_samples) * 100,
            2
        )

        distribution[str(cls)] = {
            "count": int(count),
            "percent": percent
        }

    majority_class = str(value_counts.idxmax())

    majority_percent = round(
        (value_counts.max() / total_samples) * 100,
        2
    )

    imbalanced = (
        majority_percent >= (imbalance_threshold * 100)
    )

    recommendation = (
        "Consider SMOTE, class weights, or resampling."
        if imbalanced
        else "Class distribution appears reasonably balanced."
    )

    return {
        "total_samples": int(total_samples),
        "num_classes": int(value_counts.shape[0]),
        "class_distribution": distribution,
        "majority_class": majority_class,
        "majority_percent": majority_percent,
        "imbalanced": imbalanced,
        "recommendation": recommendation
    }