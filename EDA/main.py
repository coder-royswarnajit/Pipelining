import pandas as pd
from summary import dataset_summary
from missing import missing_report
from correlation import correlation_heatmap
from visualization import distribution_plots
from eda_warnings import generate_warnings
from type_detection import detect_column_types
from outlier import detect_outliers
from skewness import detect_skewness
from cardinality import analyze_cardinality


df = pd.read_csv('C:/Users/309168/Desktop/CODES/Pipelining/uci-secom.csv')

print("\nDATASET SUMMARY:")
print(dataset_summary(df))

print("\nMISSING VALUES:")
print(missing_report(df))

print("\nWARNINGS:")
print(generate_warnings(df))


types=detect_column_types(df)
print("\nCOLUMN TYPES:")
for col, info in types.items():
    print(f"{col} -> {info['detected_type']}")
    
    
print("\nOUTLIER REPORT:")
outliers = detect_outliers(df)
for col, info in outliers.items():
    print(f"\n{col}")
    print(f"Outliers: {info['outlier_count']} ({info['outlier_percent']}%)")
    print(f"Bounds:{info['lower_bound']} to  {info['upper_bound']}")
    

print("\nSKEWNESS REPORT:")
skew_report = detect_skewness(df)
for col, info in skew_report.items():
    print(f"\n{col}")
    print(f"Skewness: {info['skewness']}")
    print(f"Type: {info['skew_type']}")
    print(f"Direction: {info['direction']}")


print("\nCARDINALITY REPORT:")
print(analyze_cardinality(df))

#Plots
correlation_heatmap(df)
distribution_plots(df)