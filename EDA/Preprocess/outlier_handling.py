import pandas as pd

from Profiling.outlier import detect_outliers
from Profiling.type_detection import detect_column_types


def handle_outliers(df):
    df=df.copy()
    

    outlier_info = detect_outliers(df)
    type_info = detect_column_types(df)


    for col, info in outlier_info.items():
        detected_type = (type_info[col]["detected_type"] )

        # Only continuous features
        if detected_type != "continuous":
            continue

        outlier_count = (info["outlier_count"])

        # Skipping clean columns
        if outlier_count == 0:
            continue

        lower_bound = (info["lower_bound"])
        upper_bound = (info["upper_bound"])
        
        df[col] = df[col].astype(float)        
        
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound

    print("\nOutlier handling completed.")

    return df