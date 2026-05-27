import os
import sys
import pandas as pd

# PROJECT ROOT
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ADD EDA FOLDER TO PATH
EDA_PATH = os.path.join(PROJECT_ROOT, "EDA")

if EDA_PATH not in sys.path:
    sys.path.append(EDA_PATH)

# IMPORT PROFILING PIPELINE
from EDA.Profiling.profiling_pipeline import run_eda_pipeline

# IMPORT AI BRAIN
from AI_Brain.brain_pipeline import run_ai_brain_pipeline


def create_metadata_from_eda_results(df, eda_results):
    """
    Creates metadata directly from EDA pipeline results.
    This bypasses build_metadata().
    """

    metadata = {}

    column_types = eda_results.get("column_types", {})
    missing_report = eda_results.get("missing_report", {})
    outlier_report = eda_results.get("outlier_report", {})
    skewness_report = eda_results.get("skewness_report", {})
    cardinality_report = eda_results.get("cardinality_report", {})

    for col in df.columns:

        metadata[col] = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percent": round(
                (df[col].isnull().sum() / len(df)) * 100,
                2
            ),
            "unique_count": int(df[col].nunique(dropna=True)),
            "sample_values": df[col].dropna().head(5).tolist()
        }

        # Add detected type
        if col in column_types:
            metadata[col]["detected_type"] = column_types[col].get(
                "detected_type"
            )
        else:
            metadata[col]["detected_type"] = "unknown"

        # Add missing report if available
        if isinstance(missing_report, dict) and col in missing_report:
            metadata[col]["missing_info"] = missing_report[col]

        # Add outlier report if available
        if isinstance(outlier_report, dict) and col in outlier_report:
            metadata[col]["outlier_info"] = outlier_report[col]

        # Add skewness report if available
        if isinstance(skewness_report, dict) and col in skewness_report:
            metadata[col]["skewness_info"] = skewness_report[col]

        # Add cardinality report if available
        if isinstance(cardinality_report, dict) and col in cardinality_report:
            metadata[col]["cardinality_info"] = cardinality_report[col]

    return metadata


def main():

    print("\n==============================")
    print("AI BRAIN TEST STARTED")
    print("==============================")

    dataset_path = os.path.join(
        PROJECT_ROOT,
        "Data",
        "ai4i2020.csv"
    )

    # Alternative dataset
    # dataset_path = os.path.join(
    #     PROJECT_ROOT,
    #     "Data",
    #     "uci-secom.csv"
    # )

    print("\nLoading dataset...")
    print(f"Dataset path: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print("\nDataset loaded successfully")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nRunning profiling pipeline...")

    eda_results = run_eda_pipeline(
        df,
        save_plots=False
    )

    print("\nCreating metadata directly inside app.py...")

    metadata = create_metadata_from_eda_results(
        df=df,
        eda_results=eda_results
    )

    print("\nMetadata created successfully.")

    print("\nRunning AI Brain...")

    ai_results = run_ai_brain_pipeline(
        metadata=metadata,
        df=df
    )

    print("\n==============================")
    print("AI BRAIN OUTPUT")
    print("==============================")

    for key, value in ai_results.items():
        print(f"\n--- {key.upper()} ---")
        print(value)

    print("\n==============================")
    print("AI BRAIN TEST COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()