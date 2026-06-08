import os
import sys
import re
from unittest import result
import pandas as pd
import streamlit as st
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EDA_PATH = os.path.join(PROJECT_ROOT, "EDA")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if EDA_PATH not in sys.path:
    sys.path.append(EDA_PATH)


from Profiling.profiling_pipeline import run_eda_pipeline
from Profiling.visualization import generate_ai_recommended_plots
from Profiling.correlation import correlation_heatmap

from AI_Brain.brain_pipeline import run_ai_brain_pipeline

from Modelling.model_pipeline import run_model_pipeline
from Modelling.shap_explainer import generate_shap_explanation


def load_uploaded_dataset(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format. Please upload CSV or XLSX.")


def create_metadata_from_eda_results(df, eda_results):
    metadata = {}

    column_info = eda_results.get("column_info", {})
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

        if col in column_info:
            metadata[col]["detected_type"] = column_info[col].get(
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


def extract_target_and_problem_type(ai_results, df):
    target_column = None
    problem_type = None

    if isinstance(ai_results, dict):
        target_text = str(ai_results.get("target_column_analysis", ""))
        problem_text = str(ai_results.get("problem_type_analysis", ""))
    else:
        target_text = str(ai_results)
        problem_text = str(ai_results)

    target_text = target_text.replace("\\n", "\n")
    problem_text = problem_text.replace("\\n", "\n")

    full_text = target_text + "\n" + problem_text

    if re.search(r"\bclassification\b", problem_text, re.IGNORECASE):
        problem_type = "classification"

    elif re.search(r"\bregression\b", problem_text, re.IGNORECASE):
        problem_type = "regression"

    elif re.search(r"\bclassification\b", full_text, re.IGNORECASE):
        problem_type = "classification"

    elif re.search(r"\bregression\b", full_text, re.IGNORECASE):
        problem_type = "regression"

    patterns = [
        r"Best Target Column:\s*([^\n\r]+)",
        r"Recommended Target Column:\s*([^\n\r]+)",
        r"Recommended Target:\s*([^\n\r]+)",
        r"Selected Target Column:\s*([^\n\r]+)",
        r"Target Column:\s*([^\n\r]+)",
        r"Best Target:\s*([^\n\r]+)",
        r"Target:\s*([^\n\r]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, target_text, re.IGNORECASE)

        if match:
            possible_target = match.group(1).strip()

            possible_target = (
                possible_target
                .replace("`", "")
                .replace('"', "")
                .replace("'", "")
                .replace(",", "")
                .replace(".", "")
                .strip()
            )

            for col in df.columns:
                if possible_target.lower() == str(col).lower():
                    target_column = col
                    break

        if target_column is not None:
            break

    if target_column is None:
        raise ValueError(
            "Could not extract target column from AI_Brain output."
        )

    if problem_type is None:
        raise ValueError(
            "Could not extract problem type from AI_Brain output."
        )

    return target_column, problem_type


def display_eda_report(eda_results):
    st.subheader("EDA Report")

    for key, value in eda_results.items():

        if key in ["plot_paths", "metadata","plot_recommendations"]:
            continue

        section_title = key.replace("_", " ").title()

        with st.expander(section_title, expanded=False):

            if isinstance(value, pd.DataFrame):
                st.dataframe(value)

            elif isinstance(value, pd.Series):
                st.dataframe(value.to_frame())

            elif isinstance(value, dict):
                try:
                    temp_df = pd.DataFrame(value).T
                    temp_df = temp_df.astype(str)
                    st.dataframe(temp_df)
                except Exception:
                    st.write(value)

            elif isinstance(value, list):
                st.write(value)

            else:
                st.write(value)

    if "plot_paths" in eda_results and len(eda_results["plot_paths"]) > 0:
        st.subheader("EDA Graphs")

        for plot_path in eda_results["plot_paths"]:
            html_plot_path = plot_path.replace("\\", "/")

            if os.path.exists(plot_path):
                st.image(html_plot_path, caption=os.path.basename(plot_path))
            else:
                st.warning(f"Plot not found: {plot_path}")


def generate_eda_html_report(eda_results):
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

            img {
                width: 95%;
                max-width: 1100px;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px;
                background-color: white;
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
    """

    html += "<h1>Exploratory Data Analysis Report</h1>"

    for key, value in eda_results.items():

        if key == "plot_paths":
            continue

        section_title = key.replace("_", " ").title()

        html += "<div class='section'>"
        html += f"<h2>{section_title}</h2>"

        if isinstance(value, pd.DataFrame):
            html += value.astype(str).to_html(index=True, border=1)

        elif isinstance(value, pd.Series):
            html += value.astype(str).to_frame().to_html(border=1)

        elif isinstance(value, dict):
            try:
                temp_df = pd.DataFrame(value).T
                html += temp_df.astype(str).to_html(border=1)
            except Exception:
                html += f"<pre>{str(value)}</pre>"

        elif isinstance(value, list):
            html += f"<pre>{str(value)}</pre>"

        else:
            html += f"<pre>{str(value)}</pre>"

        html += "</div>"

    if "plot_paths" in eda_results and len(eda_results["plot_paths"]) > 0:
        html += "<div class='section'>"
        html += "<h2>EDA Graphs</h2>"

        for plot_path in eda_results["plot_paths"]:
            html_plot_path = plot_path.replace("\\", "/")
            plot_name = os.path.basename(plot_path)

            html += f"""
            <h3>{plot_name}</h3>
            <img src="{html_plot_path}">
            """

        html += "</div>"

    html += """
    </body>
    </html>
    """

    return html


def main():

    st.set_page_config(page_title="Auto EDA + Modelling", layout="wide")

    st.title("Auto EDA Analysis")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Upload a CSV or Excel dataset to start.")
        return

    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = uploaded_file.name

    elif st.session_state["uploaded_file_name"] != uploaded_file.name:
        st.session_state.clear()
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.rerun()

    df = load_uploaded_dataset(uploaded_file)

    st.success("Dataset uploaded successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.divider()

    button_col1, button_col2 = st.columns(2)

    with button_col1:
        run_eda_analysis = st.button("Run EDA Analysis", use_container_width=True)

    with button_col2:
        run_model = st.button("Run Modelling", use_container_width=True)

    if run_eda_analysis:
        with st.spinner("Running EDA pipeline..."):
            eda_results = run_eda_pipeline(
                df,
                save_plots=False,
                plot_folder="eda_plots",
                target_column=None
            )

        metadata = create_metadata_from_eda_results(
            df=df,
            eda_results=eda_results
        )

        eda_results["metadata"] = metadata

        with st.spinner("Running AI Brain analysis..."):
            ai_results = run_ai_brain_pipeline(
                metadata=metadata,
                df=df,
                eda_results=eda_results,
                run_plot_recommendation=True,
                run_eda_explanation=True
            )

        st.session_state["ai_results"] = ai_results

        target_column = None
        problem_type = None

        try:
            target_column, problem_type = extract_target_and_problem_type(
                ai_results=ai_results,
                df=df
            )

            st.session_state["target_column"] = target_column
            st.session_state["problem_type"] = problem_type

            plot_recommendations = ai_results.get(
                "plot_recommendations",
                []
            )

            eda_explanation = ai_results.get(
                "eda_explanation",
                None
            )

            with st.spinner("Generating AI-recommended plots..."):
                plot_output = generate_ai_recommended_plots(
                    df=df,
                    plot_recommendations=plot_recommendations,
                    plot_folder="eda_plots"
                )

            ai_plot_paths = plot_output.get("plot_paths", [])
            plot_summaries = plot_output.get("plot_summaries", [])

            column_info = eda_results.get("column_info", {})

            heatmap_output = correlation_heatmap(
                df=df,
                plot_folder="eda_plots",
                type_info=column_info,
                target_column=target_column
            )

            heatmap_path = heatmap_output.get("plot_path")
            heatmap_summary = heatmap_output.get("plot_summary")

            if heatmap_path is not None:
                ai_plot_paths.append(heatmap_path)
                st.success("Correlation heatmap generated.")
            else:
                st.info(
                    "Correlation heatmap was not generated because fewer than 2 valid "
                    "continuous columns were available."
                )

            if heatmap_summary is not None:
                plot_summaries.append(heatmap_summary)

            eda_results["plot_recommendations"] = plot_recommendations
            eda_results["plot_paths"] = ai_plot_paths
            eda_results["plot_summaries"] = plot_summaries

            if eda_explanation is not None:
                eda_results["eda_explanation"] = eda_explanation

        except Exception as e:
            st.warning(
                f"AI Brain ran, but target/problem extraction, plot generation, or explanation failed: {str(e)}"
            )

            eda_results["plot_paths"] = []

        st.session_state["eda_results"] = eda_results

        st.success("EDA analysis completed.")

        display_eda_report(eda_results)

        


        if target_column is not None and problem_type is not None:
            st.subheader("Detected ML Setup")

            setup_col1, setup_col2 = st.columns(2)
            setup_col1.metric("Target Column", target_column)
            setup_col2.metric("Problem Type", problem_type)

        html_report = generate_eda_html_report(eda_results)

        st.download_button(
            label="Download EDA Report",
            data=html_report,
            file_name="eda_report.html",
            mime="text/html",
            use_container_width=True
        )

    elif "eda_results" in st.session_state:

        st.info("Previous EDA analysis is available below.")

        display_eda_report(st.session_state["eda_results"])



        if (
            "target_column" in st.session_state
            and "problem_type" in st.session_state
        ):
            st.subheader("Detected ML Setup")

            setup_col1, setup_col2 = st.columns(2)
            setup_col1.metric(
                "Target Column",
                st.session_state["target_column"]
            )
            setup_col2.metric(
                "Problem Type",
                st.session_state["problem_type"]
            )

        html_report = generate_eda_html_report(
            st.session_state["eda_results"]
        )

        st.download_button(
            label="Download EDA Report",
            data=html_report,
            file_name="eda_report.html",
            mime="text/html",
            use_container_width=True
        )

    if run_model:

        if "ai_results" not in st.session_state:
            st.error(
                "Please run EDA Analysis first. "
                "AI Brain must detect the target column and problem type before modelling."
            )
            return

        if (
            "target_column" not in st.session_state
            or "problem_type" not in st.session_state
        ):
            try:
                target_column, problem_type = extract_target_and_problem_type(
                    ai_results=st.session_state["ai_results"],
                    df=df
                )

                st.session_state["target_column"] = target_column
                st.session_state["problem_type"] = problem_type

            except Exception as e:
                st.error(
                    f"Could not extract target column/problem type from AI Brain output: {str(e)}"
                )
                return

        target_column = st.session_state["target_column"]
        problem_type = st.session_state["problem_type"]

        st.subheader("Modelling Setup")

        setup_col1, setup_col2 = st.columns(2)
        setup_col1.metric("Target Column", target_column)
        setup_col2.metric("Problem Type", problem_type)

        if target_column not in df.columns:
            st.error(
                f"Target column '{target_column}' not found in dataset."
            )
            return

        st.info(
            "Modelling uses the raw dataset. "
            "Preprocessing is done inside the modelling pipeline after train-test split "
            "to avoid data leakage."
        )

        with st.spinner("Running modelling pipeline..."):
            model_results = run_model_pipeline(df=df, target_column=target_column, problem_type=problem_type)

        st.session_state["model_results"] = model_results

        st.success("Modelling completed.")

        st.subheader("Modelling Results")

        for model_name, result in model_results.items():            
            with st.expander(model_name):

                display_result = {k: v for k, v in result.items() if k not in ["model", "actual_values", "predicted_values","confusion_matrix", "feature_importance", "coefficients" ]}
                st.write(display_result)

                #Classification
                if (result.get("status") == "success" and "confusion_matrix" in result):
                    st.subheader("Confusion Matrix")
                    cm_df = pd.DataFrame(result["confusion_matrix"])
                    st.dataframe(cm_df)
                
                # Feature Importance (Tree Models)
                if (result.get("status") == "success" and result.get("feature_importance")):
                    st.subheader("Feature Importance")
                    
                    importance_df = pd.DataFrame({"Feature": list(result["feature_importance"].keys()), 
                                                  "Importance": list(result["feature_importance"].values() )})
                    importance_df = (importance_df.sort_values("Importance", ascending=False))
                    
                    st.dataframe(importance_df, use_container_width=True)
                
                # Coefficients (Linear / Logistic)
                if (result.get("status") == "success" and result.get("coefficients")):
                    st.subheader("Feature Coefficients")
                    
                    coef_df = pd.DataFrame({"Feature": list(result["coefficients"].keys()), 
                                            "Coefficient": list(result["coefficients"].values())})
                    coef_df = (coef_df.sort_values("Coefficient", ascending=False))
                    
                    st.dataframe(coef_df, use_container_width=True)
                
                # Regression Error Histogram
                if (result.get("status") == "success" and "actual_values" in result and "predicted_values" in result):
                    st.subheader("Prediction Error Distribution")

                    actual = np.array(result["actual_values"])
                    predicted = np.array(result["predicted_values"])

                    errors = actual - predicted

                    fig, ax = plt.subplots(figsize=(8, 4))

                    ax.hist(errors, bins=20)

                    ax.axvline(0,linestyle="--")
                    ax.set_title("Prediction Error Distribution")
                    ax.set_xlabel("Error (Actual - Predicted)")
                    ax.set_ylabel("Frequency")
                    
                    st.pyplot(fig)
                    st.caption("Values near 0 indicate accurate predictions. "
                        "Positive errors indicate underprediction. "
                        "Negative errors indicate overprediction.")
                
                
                col1, col2 = st.columns(2)

                with col1:

                    if (
                        result.get("status") == "success"
                        and "model" in result
                    ):

                        buffer = io.BytesIO()

                        joblib.dump(
                            result["model"],
                            buffer
                        )

                        st.download_button(
                            label="Download Model",
                            data=buffer.getvalue(),
                            file_name=(
                                model_name
                                .replace(" ", "_")
                                .lower()
                                + ".pkl"
                            ),
                            mime="application/octet-stream",
                            key=f"download_{model_name}"
                        )


            with col2:

                if (
                    result.get("status") == "success"
                    and "model" in result
                    and "X_train" in result
                ):

                    if st.button(
                        "Generate SHAP",
                        key=f"shap_{model_name}"
                    ):

                        with st.spinner(
                            "Generating SHAP explanation..."
                        ):

                            shap_result = (
                                generate_shap_explanation(
                                    model=result["model"],
                                    X_train=result["X_train"],
                                    X_sample=result["X_train"]
                                )
                            )

                        if shap_result["status"] == "success":

                            st.success(
                                "SHAP explanation generated."
                            )

                            plt.figure(
                                figsize=(10, 6)
                            )

                            shap.summary_plot(
                                shap_result["shap_values"],
                                result["X_train"],
                                show=False
                            )

                            st.pyplot(
                                plt.gcf()
                            )

                            plt.close()

                        else:

                            st.error(
                                shap_result["error"]
                            )
                        
if __name__ == "__main__":
    main()