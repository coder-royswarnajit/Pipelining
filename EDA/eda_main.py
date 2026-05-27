import os
import pandas as pd
from datetime import datetime

# PIPELINES
from Profiling.profiling_pipeline import run_eda_pipeline
from Preprocess.preprocess_pipeline import run_preprocessing_pipeline


def convert_to_html_section(title, data):
    """
    Converts EDA outputs into HTML sections.
    """

    html = f"<h2>{title}</h2>"

    if isinstance(data, pd.DataFrame):
        html += data.to_html(index=True, border=1)

    elif isinstance(data, pd.Series):
        html += data.to_frame().to_html(border=1)

    elif isinstance(data, dict):
        try:
            html += pd.DataFrame(data).T.to_html(border=1)
        except Exception:
            html += f"<pre>{data}</pre>"

    elif isinstance(data, list):
        html += "<ul>"
        for item in data:
            html += f"<li>{item}</li>"
        html += "</ul>"

    else:
        html += f"<pre>{data}</pre>"

    return html


def generate_html_report(eda_results, preprocessed_df, output_path):
    """
    Generates HTML report using EDA results, preprocessing output,
    and saved EDA plots.
    """

    html = """
    <html>
    <head>
        <title>EDA Report</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
                color: #222;
            }

            h1 {
                color: #1f4e79;
                border-bottom: 3px solid #1f4e79;
                padding-bottom: 10px;
            }

            h2 {
                color: #2f5597;
                margin-top: 35px;
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            }

            h3 {
                color: #444;
                margin-top: 20px;
            }

            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
                background-color: white;
                font-size: 14px;
            }

            th {
                background-color: #d9eaf7;
                padding: 8px;
                text-align: left;
            }

            td {
                padding: 8px;
                border: 1px solid #ccc;
            }

            pre {
                background-color: white;
                padding: 12px;
                border: 1px solid #ccc;
                overflow-x: auto;
            }

            .section {
                background-color: white;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.08);
            }

            .plot-container {
                margin-top: 20px;
                margin-bottom: 30px;
                text-align: center;
            }

            .plot-container img {
                width: 95%;
                max-width: 1100px;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px;
                background-color: white;
            }

            .note {
                background-color: #fff3cd;
                padding: 12px;
                border-left: 5px solid #ffc107;
                margin-top: 10px;
            }
        </style>
    </head>

    <body>
    """

    html += "<h1>Exploratory Data Analysis Report</h1>"

    html += f"""
    <p><b>Generated On:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    """

    # EDA RESULTS
    for key, value in eda_results.items():

        # Do not show plot paths as normal table/list
        if key == "plot_paths":
            continue

        section_title = key.replace("_", " ").title()

        html += "<div class='section'>"
        html += convert_to_html_section(section_title, value)
        html += "</div>"

    # EDA GRAPHS
    html += "<div class='section'>"
    html += "<h2>EDA Graphs</h2>"

    if "plot_paths" in eda_results and len(eda_results["plot_paths"]) > 0:

        for plot_path in eda_results["plot_paths"]:

            plot_name = (
                os.path.basename(plot_path)
                .replace("_", " ")
                .replace(".png", "")
                .title()
            )

            # Convert Windows backslash to forward slash for HTML
            html_plot_path = plot_path.replace("\\", "/")

            html += f"""
            <div class="plot-container">
                <h3>{plot_name}</h3>
                <img src="{html_plot_path}" alt="{plot_name}">
            </div>
            """

    else:
        html += """
        <div class="note">
            No plots were found. Make sure your profiling pipeline saves plots
            and returns their file paths inside eda_results["plot_paths"].
        </div>
        """

    html += "</div>"

    # PREPROCESSED DATA PREVIEW
    html += "<div class='section'>"
    html += "<h2>Preprocessed Dataset Preview</h2>"
    html += preprocessed_df.head(10).to_html(index=False, border=1)
    html += "</div>"

    # PREPROCESSED DATA SHAPE
    html += "<div class='section'>"
    html += "<h2>Preprocessed Dataset Shape</h2>"
    html += f"<p><b>Rows:</b> {preprocessed_df.shape[0]}</p>"
    html += f"<p><b>Columns:</b> {preprocessed_df.shape[1]}</p>"
    html += "</div>"

    html += """
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"\nHTML report generated successfully: {output_path}")


def main(dataset_path, output_path="eda_report.html"):
    """
    Loads dataset, passes it into EDA and preprocessing pipelines,
    then creates an HTML report.
    """

    print("\nLoading dataset...")

    df = pd.read_csv(dataset_path)

    print("\nDataset loaded successfully")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # RUN EDA PIPELINE
    eda_results = run_eda_pipeline(
    df,
    save_plots=True,
    plot_folder="eda_plots")

    # RUN PREPROCESSING PIPELINE
    preprocessed_df = run_preprocessing_pipeline(df)

    # GENERATE HTML REPORT
    generate_html_report(
        eda_results=eda_results,
        preprocessed_df=preprocessed_df,
        output_path=output_path
    )


if __name__ == "__main__":

    dataset = "C:/Users/309168/Desktop/CODES/Pipelining/Data/uci-secom.csv"
    #dataset = "C:/Users/309168/Desktop/CODES/Pipelining/Data/ai4i2020.csv"

    main(
        dataset_path=dataset,
        output_path="eda_report.html"
    )