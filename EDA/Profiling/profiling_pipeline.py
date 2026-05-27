import os

# PROFILING MODULES
from Profiling.summary import dataset_summary
from Profiling.missing import missing_report
from Profiling.correlation import correlation_heatmap
from Profiling.visualization import distribution_plots
from Profiling.eda_warnings import generate_warnings
from Profiling.type_detection import detect_column_types
from Profiling.outlier import detect_outliers
from Profiling.skewness import detect_skewness
from Profiling.cardinality import analyze_cardinality
from Profiling.box_plots import box_plots


def run_eda_pipeline(df, save_plots=True, plot_folder="eda_plots"):

    eda_results = {}

    # Create plot_paths before using it
    plot_paths = []

    eda_results["dataset_summary"] = dataset_summary(df)
    eda_results["warnings"] = generate_warnings(df)
    eda_results["column_types"] = detect_column_types(df)
    eda_results["outlier_report"] = detect_outliers(df)
    eda_results["skewness_report"] = detect_skewness(df)
    eda_results["cardinality_report"] = analyze_cardinality(df)

    if save_plots:
        os.makedirs(plot_folder, exist_ok=True)

        # Correlation heatmap
        heatmap_path = correlation_heatmap(
            df,
            plot_folder=plot_folder
        )

        if heatmap_path is not None:
            plot_paths.append(heatmap_path)

        # Distribution plots
        dist_paths = distribution_plots(
            df,
            plot_folder=plot_folder
        )

        if dist_paths is not None:
            if isinstance(dist_paths, list):
                plot_paths.extend(dist_paths)
            else:
                plot_paths.append(dist_paths)

        # Box plots
        box_plot_path = box_plots(
            df,
            plot_folder=plot_folder
        )

        if box_plot_path is not None:
            plot_paths.append(box_plot_path)

    eda_results["plot_paths"] = plot_paths

    return eda_results