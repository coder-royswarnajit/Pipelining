import pandas as pd
from sklearn.impute import SimpleImputer

from Profiling.type_detection import detect_column_types
from Profiling.skewness import detect_skewness


def handle_missing_values(df):

    df = df.copy()

    # GET METADATA
    type_info = detect_column_types(df)
    skew_info = detect_skewness(df)

    # PROCESS EACH COLUMN
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100

        # Skip columns with no missing values
        if missing_count == 0:
            continue

        detected_type = type_info[col]["detected_type"]

        print(f"\nColumn: {col}")
        print(f"Missing Values: {missing_count}")
        print(f"Missing Percentage: {missing_percent:.2f}%")
        print(f"Detected Type: {detected_type}")

        # DROP HIGHLY MISSING COLUMNS
        if missing_percent > 50:
            df.drop(columns=[col], inplace=True)
            print("Dropped column (>50% missing values)")
            continue

        # CONTINUOUS FEATURES
        if detected_type == "continuous":
            skewness = 0
            if col in skew_info:
                skewness = skew_info[col]["skewness"]

            # Highly skewed -> median
            if abs(skewness) > 1:
                imputer = SimpleImputer(strategy="median")
                df[[col]] = imputer.fit_transform(df[[col]])
                print("Filled using MEDIAN")

            # Normal distribution -> mean
            else:
                imputer = SimpleImputer(strategy="mean")
                df[[col]] = imputer.fit_transform(df[[col]])
                print("Filled using MEAN")

        # CATEGORICAL / BINARY FEATURES
        elif detected_type in ["binary",
                               "categorical",
                               "categorical_numeric" ]:

            imputer=SimpleImputer(strategy="most_frequent")
            df[[col]] = imputer.fit_transform(df[[col]])
            print("Filled using MODE")

        # TEXT FEATURES
        elif detected_type == "text":

            imputer = SimpleImputer(strategy="constant",
                                    fill_value="Unknown")
            df[[col]] = imputer.fit_transform(df[[col]])
            print("Filled text with Unknown")

        # IDENTIFIER FEATURES
        elif detected_type == "identifier":
            print("Identifier column left unchanged")

        # DATETIME FEATURES
        elif detected_type == "datetime":
            df[col] = df[col].ffill()
            print("Forward fill applied")

        # UNKNOWN TYPES
        else:
            print("Action: No strategy applied")

    print("\nMissing value handling completed.")

    return df