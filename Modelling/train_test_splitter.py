from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold


def split_data(df, target_column, problem_type, test_size=0.2, random_state=42):
    """
    Splits dataset into train and test sets.

    Important:
    - Rows with missing target values are removed before splitting.
    - Classification uses stratified split when possible.
    - Regression uses normal train-test split.
    """

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    problem_type = problem_type.lower()

    if problem_type not in ["classification", "regression"]:
        raise ValueError("problem_type must be either 'classification' or 'regression'.")

    # Remove rows where target is missing
    missing_target_count = df[target_column].isnull().sum()

    if missing_target_count > 0:
        df = df.dropna(subset=[target_column])

    if df.empty:
        raise ValueError("No rows left after removing missing target values.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_value = None

    if problem_type == "classification":

        class_counts = y.value_counts()

        if len(class_counts) < 2:
            pass

        elif class_counts.min() >= 2:
            stratify_value = y

        else:
            pass
    else:
        pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_value)

    return X_train, X_test, y_train, y_test


def get_cv_strategy(problem_type, n_splits=5, n_repeats=3, random_state=42):
    """
    Returns appropriate cross-validation strategy.
    """

    problem_type = problem_type.lower()

    if problem_type == "classification":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    elif problem_type == "regression":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'")