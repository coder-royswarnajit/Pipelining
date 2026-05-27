from Profiling.type_detection import detect_column_types
from Profiling.skewness import detect_skewness
from Profiling.outlier import detect_outliers
from Profiling.cardinality import analyze_cardinality


def build_metadata(df):

    type_info = detect_column_types(df)
    skew_info = detect_skewness(df)
    outlier_info = detect_outliers(df)
    cardinality_info = analyze_cardinality(df)

    metadata = {}
    for col in df.columns:
        metadata[col] = {}
        series = df[col]

        metadata[col]["dtype"] = (str(series.dtype))
        metadata[col]["missing_count"] = int(series.isnull().sum())
        metadata[col]["missing_percent"] = round((series.isnull().sum()/ len(df)) * 100,2)
        metadata[col]["unique_count"] = int(series.nunique(dropna=True))
        metadata[col]["detected_type"] = (type_info[col]["detected_type"])

        # SKEWNESS
        if col in skew_info:
            metadata[col]["skewness"] = (skew_info[col]["skewness"])         
            metadata[col]["skew_type"] = (skew_info[col]["skew_type"])
            metadata[col]["skew_direction"] = (skew_info[col]["direction"])

        else:
            metadata[col]["skewness"] = None
            metadata[col]["skew_type"] = None
            metadata[col]["skew_direction"] = None

        # OUTLIERS
        if col in outlier_info:
            metadata[col]["outlier_count"] = (outlier_info[col]["outlier_count"])
            metadata[col]["outlier_percent"] = (outlier_info[col]["outlier_percent"])
            metadata[col]["lower_bound"] = (outlier_info[col]["lower_bound"])
            metadata[col]["upper_bound"] = (outlier_info[col]["upper_bound" ])

        else:
            metadata[col]["outlier_count"] = 0
            metadata[col]["outlier_percent"] = 0
            metadata[col]["lower_bound"] = None
            metadata[col]["upper_bound"] = None

        # CARDINALITY
        if col in cardinality_info:
            metadata[col]["cardinality_type"] = (cardinality_info[col]["cardinality_type"])
            metadata[col]["unique_ratio"] = (cardinality_info[col]["unique_ratio"])

        else:
            metadata[col]["cardinality_type"] = (None)
            metadata[col]["unique_ratio"] = (None)

    print("\nMetadata generation completed.")

    return metadata