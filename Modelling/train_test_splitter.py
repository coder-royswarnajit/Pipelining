from sklearn.model_selection import train_test_split


def split_data(
    df,
    target_column,
    problem_type,
    test_size=0.2,
    random_state=42
):
    """
    Splits dataset into train and test sets.

    Important:
    - Rows with missing target values are removed before splitting.
    - Classification uses stratified split when possible.
    - Regression uses normal train-test split.
    """

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    problem_type = problem_type.lower()

    if problem_type not in ["classification", "regression"]:
        raise ValueError(
            "problem_type must be either 'classification' or 'regression'."
        )

    # Remove rows where target is missing
    missing_target_count = df[target_column].isnull().sum()

    if missing_target_count > 0:
        print(
            f"\nDropping {missing_target_count} rows because "
            f"target column '{target_column}' contains missing values."
        )

        df = df.dropna(subset=[target_column])

    if df.empty:
        raise ValueError(
            "No rows left after removing missing target values."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_value = None

    if problem_type == "classification":

        class_counts = y.value_counts()

        if len(class_counts) < 2:
            print(
                "\nStratified split skipped because only one class is present."
            )

        elif class_counts.min() >= 2:
            stratify_value = y
            print("\nUsing stratified train-test split.")

        else:
            print(
                "\nStratified split skipped because at least one class "
                "has fewer than 2 samples."
            )

    else:
        print("\nUsing normal train-test split for regression.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value
    )

    print("\nTrain-test split completed.")
    print(f"Problem Type: {problem_type}")
    print(f"Target Column: {target_column}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    if problem_type == "classification":
        print("\nTrain target distribution:")
        print(y_train.value_counts())

        print("\nTest target distribution:")
        print(y_test.value_counts())

    return X_train, X_test, y_train, y_test