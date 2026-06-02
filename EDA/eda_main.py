import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


from Profiling.profiling_pipeline import run_eda_pipeline
from Preprocess.preprocess_pipeline import run_preprocessing_pipeline


def create_metadata_from_eda_results(df, eda_results):
    """
    Creates metadata directly from EDA pipeline results.
    This bypasses build_metadata().
    """

    metadata = {}

    column_types = eda_results.get("column_info", {})
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

        if col in column_types:
            metadata[col]["detected_type"] = column_types[col].get(
                "detected_type",
                "unknown"
            )

        else:
            metadata[col]["detected_type"] = "unknown"

        if isinstance(missing_report, dict) and col in missing_report:
            metadata[col]["missing_info"] = missing_report[col]

        if isinstance(outlier_report, dict) and col in outlier_report:
            metadata[col]["outlier_info"] = outlier_report[col]

        if isinstance(skewness_report, dict) and col in skewness_report:
            metadata[col]["skewness_info"] = skewness_report[col]

        if isinstance(cardinality_report, dict) and col in cardinality_report:
            metadata[col]["cardinality_info"] = cardinality_report[col]

    return metadata


def run_eda_workflow(
    df,
    save_plots=False,
    plot_folder="eda_plots",
    target_column=None,
    run_preprocessing=True
):
    """
    Runs profiling and optional full-dataset preprocessing.

    It does NOT:
    - load CSV
    - create HTML
    - create PDF
    - call AI Brain

    It only returns results to app.py.
    """

    eda_results = run_eda_pipeline(
        df,
        save_plots=save_plots,
        plot_folder=plot_folder,
        target_column=target_column
    )

    metadata = create_metadata_from_eda_results(
        df=df,
        eda_results=eda_results
    )

    preprocessed_df = None

    if run_preprocessing:
        preprocessed_df = run_preprocessing_pipeline(
            df,
            target_column=target_column
        )

    final_results = {
        "eda_results": eda_results,
        "metadata": metadata,
        "preprocessed_df": preprocessed_df
    }

    return final_results