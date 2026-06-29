from Profiling.summary import dataset_summary
from Profiling.imbalance_report import analyze_class_imbalance
from Profiling.eda_warnings import generate_warnings
from Profiling.type_detection import detect_column_types
from Profiling.outlier import detect_outliers
from Profiling.skewness import detect_skewness
from Profiling.cardinality import analyze_cardinality
from Profiling.correlation import correlation_analysis



def run_eda_pipeline(df, target_column=None):

    eda_results = {}

    eda_results["dataset_summary"] = dataset_summary(df)

    eda_results["warnings"] = generate_warnings(df)

    column_types = detect_column_types(df)
    eda_results["column_info"] = column_types

    eda_results["outlier_report"] = detect_outliers(df, type_info=column_types)
    eda_results["skewness_report"] = detect_skewness(df, type_info=column_types, target_column=target_column)
    eda_results["cardinality_report"] = analyze_cardinality(df,type_info=column_types)
    eda_results["correlation_report"] = correlation_analysis(df, type_info=column_types, target_column=target_column,)

    imbalance_report = None
    
    if target_column is not None:

        target_series = df[target_column]
        unique_values = target_series.nunique(dropna=True)
        
        if unique_values <= 20:
            imbalance_report = analyze_class_imbalance(target_series)
            
    # Store metadata directly here using already detected column types
    metadata = {}

    for col in df.columns:
        metadata[col] = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percent": round(
                (df[col].isnull().sum() / len(df)) * 100,
                2
            ),
            "unique_count": int(df[col].nunique(dropna=True)),
            "unique_ratio": round(
                (df[col].nunique(dropna=True) / len(df)) * 100,
                2
            ) if len(df) > 0 else 0,
            "detected_type": column_types[col]["detected_type"],
            "sample_values": df[col].dropna().astype(str).head(5).tolist(),
            "imbalance_report": imbalance_report if col == target_column else None,}
    
        if col in eda_results["skewness_report"]:
            metadata[col]["skewness"] = (eda_results["skewness_report"][col]["skewness"])

        if col in eda_results["cardinality_report"]:
            metadata[col]["cardinality_type"] = (eda_results["cardinality_report"][col]["cardinality_type"])


    
    eda_results["metadata"] = metadata
    

    return eda_results