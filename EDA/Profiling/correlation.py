import matplotlib.pyplot as plt
import seaborn as sns

'''Finds Correlation Heatmap Between Each and Every Numerical Feature'''
def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Remove binary and classification-like columns
    continuous_columns = []
    excluded_columns = []

    for col in numeric_df.columns:

        unique_values = numeric_df[col].dropna().unique()
        unique_count = len(unique_values)

        # Binary or classification-like feature
        if unique_count <= 2:
            excluded_columns.append(col)
        # Continuous feature
        else:
            continuous_columns.append(col)

    print('Excluded Columns From Correlation:')
    print(excluded_columns)

    filtered_df = numeric_df[continuous_columns]

    # Correlation matrix
    if filtered_df.empty:
        print("No continuous columns available for correlation heatmap.")
        return

    corr = filtered_df.corr()

    # Heatmap
    plt.figure(figsize=(14, 10))

    sns.heatmap(corr,cmap='coolwarm',center=0,cbar=True,annot=True,fmt='.2f',square=True)

    plt.title('Correlation Heatmap (Continuous Features Only)')
    plt.show()

    # Strong correlations
    print('Highly Correlated Features:')

    columns = corr.columns

    checked_pairs = set()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            corr_value = corr.iloc[i, j]

            if abs(corr_value) >= 0.8:

                pair = tuple(sorted([columns[i], columns[j]]))

                if pair not in checked_pairs:

                    print(f'{columns[i]} <-> {columns[j]} : '
                          f'{corr_value:.2f}')

                    checked_pairs.add(pair)