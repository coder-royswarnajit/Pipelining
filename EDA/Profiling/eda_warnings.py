def generate_warnings(df):

    warnings = []

    # Constant columns
    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            warnings.append(f'{col} is a constant column')

    # High missing values
    for col in df.columns:
        missing_percent = (df[col].isnull().sum() / len(df)) * 100

        if missing_percent > 50:
            warnings.append(f'{col} has more than 50% missing values')

    # Highly correlated features
    corr_matrix = df.corr(numeric_only=True)

    checked_pairs = set()

    for col in corr_matrix.columns:
        for row in corr_matrix.index:

            if col != row:

                pair = tuple(sorted([col, row]))

                if pair not in checked_pairs:

                    corr_value = corr_matrix.loc[row, col]

                    if abs(corr_value) > 0.9:
                        warnings.append(
                            f'{col} and {row} are highly correlated '
                            f'({corr_value:.2f})'
                        )

                    checked_pairs.add(pair)

    if warnings is None or len(warnings) == 0:
        warnings.append("No warnings detected.")
        
    return warnings