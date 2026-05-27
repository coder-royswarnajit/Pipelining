import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from Profiling.type_detection import detect_column_types
from Profiling.cardinality import analyze_cardinality


def encode_features(df):
    df = df.copy()
    encoded_df = pd.DataFrame(index=df.index)


    type_info = detect_column_types(df)
    cardinality_info = analyze_cardinality(df)

    for col in df.columns:

        #Drop constant columns
        if df[col].nunique() <= 1:
            print(f"Dropped constant column: {col}")
            continue

        detected_type = type_info[col]["detected_type"]

        # IDENTIFIER COLUMNS
        if detected_type == "identifier":
            print(f"Dropped identifier column: {col}")
            continue

        # TEXT COLUMNS
        elif detected_type == "text":
            print(f"Skipped text column: {col}")
            continue

        # BINARY COLUMNS
        elif detected_type == "binary":
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str))
            print(f"Label Encoding applied on: {col}")

        # STRICT categorical handling (no numeric leakage)
        elif detected_type == "categorical":

            cardinality_type = cardinality_info.get(col, {}).get(
                "cardinality_type", "Low Cardinality"
            )

            print(f"{col} → Cardinality: {cardinality_type}")

            # LOW/MODERATE CARDINALITY → OneHot
            if cardinality_type in ["Low Cardinality", "Moderate Cardinality"]:
                encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore'
                )

                transformed = encoder.fit_transform(df[[col]])
                feature_names = encoder.get_feature_names_out([col])

                transformed_df = pd.DataFrame(
                    transformed,
                    columns=feature_names,
                    index=df.index
                )

                encoded_df = pd.concat([encoded_df, transformed_df], axis=1)

                print(f"OneHotEncoder applied on: {col}")

            # HIGH CARDINALITY → Frequency Encoding
            else:
                freq_map = df[col].value_counts(normalize=True)
                encoded_df[col] = df[col].map(freq_map)

                print(f"Frequency Encoding applied on: {col}")

        #ALL numeric columns (continuous + categorical_numeric)
        else:
            encoded_df[col] = df[col]
            print(f"Numeric column retained: {col}")

    print("\nEncoding completed.")

    return encoded_df