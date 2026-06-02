from sklearn.model_selection import train_test_split


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        df: pandas DataFrame
        target_column: column to predict
        test_size: percentage of data used for testing
        random_state: fixed value for reproducible split

    Returns:
        X_train, X_test, y_train, y_test
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print("\nTrain-test split completed")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test