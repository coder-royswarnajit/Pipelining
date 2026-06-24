import os
import re
import sys
from unittest import result
import pandas as pd
import streamlit as st
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import base64
import html
import json
import ast


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EDA_PATH = os.path.join(PROJECT_ROOT, "EDA")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if EDA_PATH not in sys.path:
    sys.path.append(EDA_PATH)


from Profiling import summary
from Profiling.profiling_pipeline import run_eda_pipeline
from Profiling.visualization import generate_ai_recommended_plots
from Profiling.correlation import correlation_heatmap

from AI_Brain.brain_pipeline import run_ai_brain_pipeline
from AI_Brain.plot_explainer import explain_all_plots

from Modelling.model_pipeline import run_model_pipeline
from Modelling.train_test_splitter import split_data

from Preprocess.preprocess_pipeline import preprocess_train_test_for_model


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
    if not isinstance(column_info, dict):
        column_info = {}
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


def display_eda_report(eda_results):
    st.subheader("EDA Report")

    def _render_text_value(val):
        if isinstance(val, dict):
            lines = [f"{k}: {v}" for k, v in val.items()]
            return "\n".join(lines)
        if isinstance(val, list):
            return "\n".join([str(x) for x in val])
        return str(val)

    # Dataset summary boxes shown at top of EDA Report (single-line boxed layout)
    try:
        df_for_summary = st.session_state.get("uploaded_df")
        if df_for_summary is not None:
            rows = df_for_summary.shape[0]
            cols = df_for_summary.shape[1]
            dup = int(rows - df_for_summary.drop_duplicates().shape[0])
            missing_pct = round(df_for_summary.isnull().sum().sum() / (rows * cols) * 100, 2) if rows*cols>0 else 0.0
            

            metrics = [
                ("Rows", f"{rows:,}"),
                ("Columns", f"{cols:,}"),
                ("Missing %", f"{missing_pct}%"),
                ("Duplicate Rows", f"{dup:,}"),
            ]

            cols_ui = st.columns(len(metrics))
            box_html = """
                <div style="
                    border:1px solid #e2e8f0;
                    border-radius:8px;
                    padding:10px 14px;
                    background:#ffffff;
                    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
                    text-align:center;
                    min-width:140px;
                ">
                    <div style="font-size:12px;color:#6b7280;">{label}</div>
                    <div style="font-weight:700;font-size:16px;margin-top:6px;color:#0f172a">{value}</div>
                </div>
            """
            for c, (label, val) in zip(cols_ui, metrics):
                c.markdown(box_html.format(label=label, value=val), unsafe_allow_html=True)
    except Exception:
        pass

    for key, value in eda_results.items():
        # Exclude internal keys from downloaded report
        if key in ["plot_paths", "metadata", "dataset_summary", "plot_recommendations", "plot_summaries", "plot_explanations", "heatmap_path", "heatmap_summary"]:
            continue

        section_title = key.replace("_", " ").title()

        with st.expander(section_title, expanded=False):

            # Render warnings/messages as plain textual content
            if any(k in key.lower() for k in ("warn", "message", "note")):
                st.text(_render_text_value(value))
                continue

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
                    st.text(_render_text_value(value))

            
            elif isinstance(value, list):
                st.text(_render_text_value(value))

            else:
                st.write(value)


def generate_full_html_report(eda_results,model_results):
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
        # Exclude internal keys from downloaded report
        if key in ["plot_paths", "metadata", "dataset_summary","plot_summaries", "plot_explanations", "plot_recommendations", "heatmap_path", "heatmap_summary"]:
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

        if key == "eda_explanation":
            plot_explanations = eda_results.get("plot_explanations",{})
            if plot_explanations is not None:
                html += "<h3>Plot Explanations</h3>"
                try:
                    if isinstance(plot_explanations, dict):
                        pe_df = pd.DataFrame(plot_explanations).T
                        html += pe_df.astype(str).to_html(border=1)
                    else:
                        html += f"<pre>{str(plot_explanations)}</pre>"
                except Exception:
                    html += f"<pre>{str(plot_explanations)}</pre>"

        html += "</div>"

    html += "<h1>Plots</h1>"

    for plot_path in eda_results.get("plot_paths", []):

        if not os.path.exists(plot_path):
            continue

        with open(plot_path, "rb") as img:
                encoded = base64.b64encode(
                    img.read()
                ).decode()
        
        plot_name = os.path.splitext(
            os.path.basename(plot_path)
        )[0]

        summary = ""

        if plot_name in plot_explanations:
            explanation = plot_explanations[plot_name]

            if isinstance(explanation, dict):
                summary = explanation.get(
                    "summary",
                    ""
                )
            else:
                summary = str(explanation)
        
    

        plot_name = os.path.splitext(
            os.path.basename(plot_path)
        )[0]

        plot_name = re.sub(
            r"^\d+_",
            "",
            plot_name
        )

        plot_name = plot_name.replace(
            "_",
            " "
        ).title()

        html += f"""
            <div class='section'>
                <h3>{plot_name}</h3>
                <img src="data:image/png;base64,{encoded}">
                <p>{summary}</p>
            </div>
            """
    html += "<h1>Modelling Results</h1>"

    for model_name, result in model_results.items():

        html += f"<h2>{model_name}</h2>"

        metrics = {
            k: v
            for k, v in result.items()
            if k not in [
                "model",
                "actual_values",
                "predicted_values",
                "confusion_matrix",
                "feature_importance",
                "coefficients",
                "X_train",
                "X_test",
                "y_train",
                "y_test"
            ]
        }

        metrics_df = pd.DataFrame(
            metrics.items(),
            columns=["Metric", "Value"]
        )

        html += metrics_df.to_html(index=False)

        if "confusion_matrix" in result:

            html += "<h3>Confusion Matrix</h3>"

            cm_df = pd.DataFrame(
                result["confusion_matrix"]
            )

            html += cm_df.to_html(index=False)

    html += """
                </body>
                </html>
                """

    return html


def _parse_possible_json_or_literal(val):
    """If val is a JSON-like or Python-literal string, return parsed object; else return original."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return val
    # Try JSON first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try Python literal (e.g., single quotes)
    try:
        return ast.literal_eval(s)
    except Exception:
        return val


def normalize_eda_results(eda_results):
    """Parse any JSON/str-encoded sections in eda_results except text/list plot fields."""
    for k, v in list(eda_results.items()):
        if k in [
            "eda_explanation",
            "plot_paths",
            "plot_summaries",
            "plot_explanations",
        ]:
            continue
        eda_results[k] = _parse_possible_json_or_literal(v)
    return eda_results


def main():

    st.set_page_config(page_title="Auto EDA + Modelling", layout="wide")

    # Increase base font size slightly for the Streamlit app UI
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {
            font-size: 16px;
        }
        /* Adjust common component text sizes */
        .stButton>button, .stMetricValue, .stTextInput>div>label {
            font-size: 15px !important;
        }
        .stExpanderHeader, .streamlit-expanderHeader, .stHeader, .stMarkdown {
            font-size: 15px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI Assisted Auto EDA-ML Platform")

    tab1, tab2, tab3 = st.tabs(["EDA Analysis", "All Plots", "Modelling"])

    # --- Tab 1: EDA Analysis ---
    with tab1:
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

        if uploaded_file is None:
            st.info("Upload a CSV or Excel dataset to start.")
        else:
            if "uploaded_file_name" not in st.session_state:
                st.session_state["uploaded_file_name"] = uploaded_file.name
            elif st.session_state["uploaded_file_name"] != uploaded_file.name:
                st.session_state.clear()
                st.session_state["uploaded_file_name"] = uploaded_file.name
                st.rerun()

            df = load_uploaded_dataset(uploaded_file)
            # persist uploaded dataframe for later summary display
            st.session_state["uploaded_df"] = df

            st.subheader("Model Setup")
            target_index = 0
            if len(df.columns) > 0 and st.session_state.get("target_column") in list(df.columns):
                target_index = list(df.columns).index(st.session_state["target_column"])

            target_column = st.selectbox(
                "Choose the target column",
                options=list(df.columns),
                index=target_index,
                key="target_column_choice",
            )

            problem_type = st.radio(
                "Choose the problem type",
                options=["classification", "regression"],
                index=0 if st.session_state.get("problem_type", "classification") == "classification" else 1,
                horizontal=True,
                key="problem_type_choice",
            )
            
            selected_dtype = str(df[target_column].dtype)

            is_numeric = pd.api.types.is_numeric_dtype(df[target_column])

            if problem_type == "regression" and not is_numeric:
                st.warning(
                    f"'{target_column}' appears to be categorical. "
                    "Regression typically requires a continuous numeric target."
                )

            elif problem_type == "classification" and is_numeric:
                unique_count = df[target_column].nunique(dropna=True)

                if unique_count > 20:
                    st.warning(
                        f"'{target_column}' appears to be continuous "
                        "Classification usually requires categorical classes."
                    )

            st.session_state["target_column"] = target_column
            st.session_state["problem_type"] = problem_type

            st.success("Dataset uploaded successfully.")
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            col1, col2 = st.columns(2)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])

            st.divider()

            run_analysis = st.button("Run Full Analysis", use_container_width=True)

            if run_analysis:
                with st.spinner("Running EDA pipeline..."):
                    eda_results = run_eda_pipeline(
                        df,
                        save_plots=True,
                        plot_folder="eda_plots",
                        target_column=None
                    )

                metadata = create_metadata_from_eda_results(df=df, eda_results=eda_results)
                eda_results["metadata"] = metadata

                with st.spinner("Running AI Brain analysis..."):
                    ai_results = run_ai_brain_pipeline(
                        metadata=metadata,
                        df=df,
                        eda_results=eda_results,
                        run_plot_recommendation=True,
                        run_eda_explanation=True,
                        run_imputation_recommendation=True,
                        target_column=st.session_state.get("target_column"),
                        problem_type=st.session_state.get("problem_type"),
                    )
                    
                    print(ai_results.get("imputation_recommendations"))

                st.session_state["ai_results"] = ai_results
                ai_plot_paths = []
                plot_summaries = []
                plot_explanations = {}

                try:
                    target_column = st.session_state.get("target_column")
                    problem_type = st.session_state.get("problem_type")

                    plot_recommendations = ai_results.get("plot_recommendations", [])
                    eda_explanation = ai_results.get("eda_explanation", None)

                    with st.spinner("Generating AI-recommended plots..."):
                        plot_output = generate_ai_recommended_plots(df, plot_recommendations, plot_folder="eda_plots")
                    
                    
                    ai_plot_paths = plot_output.get("plot_paths", [])
                    plot_summaries = plot_output.get("plot_summaries", [])
                    
                    print("\n========== PLOT DEBUG ==========")
                    print("Total plot files:", len(ai_plot_paths))
                    print("Total plot summaries:", len(plot_summaries))

                    for i, item in enumerate(plot_summaries):
                        print(f"{i+1}.", item.get("plot_key"))
                    print("================================\n")

                    plot_explanations = explain_all_plots(plot_summaries)
                    print(type(plot_explanations))
                    print(len(plot_explanations))
                    print(plot_explanations.keys())

                    # store results so Tab 2 can read them
                    existing_plots = eda_results.get("plot_paths", [])
                    eda_results["plot_paths"] = (existing_plots + ai_plot_paths)
                    eda_results["plot_summaries"] = plot_summaries
                    eda_results["plot_explanations"] = plot_explanations

                    if eda_explanation is not None:
                        eda_results["eda_explanation"] = eda_explanation
                    
                    with st.spinner("Running modelling pipeline..."):
                        try:
                            df = df.copy()
                            df = df.drop_duplicates()
                            df = df.dropna(subset=[target_column])
                            
                            model_results = run_model_pipeline(
                                df=df,
                                target_column=target_column,
                                problem_type=problem_type,
                                metadata=metadata
                            )
                            st.session_state["model_results"] = model_results
                            
                        except Exception as e:
                            st.error(f"Modelling failed: {str(e)}")

                
            

                except Exception as e:
                    st.warning(f"AI Brain ran, but some post-processing failed: {str(e)}")
                    eda_results.setdefault("plot_paths", [])

                # Normalize any JSON/string-encoded results (but keep text/list fields intact)
                eda_results = normalize_eda_results(eda_results)
                st.session_state["eda_results"] = eda_results
                st.success("Full Analysis Completed.")

            # show existing results if present
            if "eda_results" in st.session_state:
                display_eda_report(st.session_state["eda_results"])

                html_report = generate_full_html_report(eda_results=st.session_state["eda_results"],model_results=st.session_state.get("model_results", {}))
                st.download_button(
                    label="Download Report",
                    data=html_report,
                    file_name="eda_report.html",
                    mime="text/html",
                    use_container_width=True
                )

    # --- Tab 2: All Plots ---
    with tab2:
        st.subheader("All Plots")
        eda_results = st.session_state.get("eda_results")

        if not eda_results:
            st.info("No plots available. Run EDA Analysis first.")
        else:
            
            #heatmap = eda_results.get("heatmap", {})
            #heatmap_path = heatmap.get("heatmap_path")

            #if heatmap_path and os.path.exists(heatmap_path):
             #   st.subheader("Correlation Heatmap")
              #  st.image(heatmap_path, use_container_width=True)
                #st.divider()
                
            plot_paths = eda_results.get("plot_paths", [])
            plot_explanations = eda_results.get("plot_explanations", {})
            plot_summaries = eda_results.get("plot_summaries", [])
            
            def _resolve_plot_summary(plot_path):
                plot_stem = os.path.splitext(os.path.basename(plot_path))[0]

                if isinstance(plot_explanations, dict):
                    direct_match = plot_explanations.get(plot_stem)
                    if isinstance(direct_match, dict):
                        return direct_match.get("summary", "")
                    if isinstance(direct_match, str):
                        return direct_match

                    for key, value in plot_explanations.items():
                        if key == plot_stem or key == os.path.basename(plot_path):
                            if isinstance(value, dict):
                                return value.get("summary", "")
                            return str(value)

                if isinstance(plot_explanations, list):
                    for item in plot_explanations:
                        if not isinstance(item, dict):
                            continue
                        item_key = item.get("plot_key") or item.get("plot_name")
                        item_path = os.path.splitext(os.path.basename(str(item.get("plot_path", ""))))[0]
                        if item_key == plot_stem or item_path == plot_stem:
                            return item.get("summary", "")

                if isinstance(plot_summaries, list):
                    for item in plot_summaries:
                        if not isinstance(item, dict):
                            continue
                        item_key = item.get("plot_key")
                        item_path = os.path.splitext(os.path.basename(str(item.get("plot_path", ""))))[0]
                        if item_key == plot_stem or item_path == plot_stem:
                            insights = item.get("insights", {})
                            return item.get("summary", "") or item.get("title", "") or str(insights)

                return ""


            for plot_path in plot_paths:

                if not os.path.exists(plot_path):
                    st.warning(f"Missing image: {plot_path}")
                    continue

                summary_text = _resolve_plot_summary(plot_path)

                plot_name = os.path.splitext(
                    os.path.basename(plot_path)
                )[0]

                import re

                plot_name = re.sub(r"^\d+_", "", plot_name)
                plot_name = plot_name.replace("_", " ").title()

                left_col, right_col = st.columns([3, 2])

                with left_col:
                    st.image(
                        plot_path,
                        width=700
                    )

                with right_col:
                    st.markdown(f"""<div style="
                                        border:1px solid #ddd;
                                        border-radius:10px;
                                        padding:15px;
                                        background:#f8f9fa;
                                    ">
                                        <h4>{plot_name}</h4>
                                        <p>{summary_text}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True)


                st.divider()

                    

    # --- Tab 3: Modelling ---
    with tab3:
        
        st.subheader("Modelling")

        # use persisted uploaded dataframe
        df = st.session_state.get("uploaded_df")
        target_column = st.session_state.get("target_column")
        problem_type = st.session_state.get("problem_type")
        metadata = st.session_state.get("eda_results", {}).get("metadata", {})

        if df is None:
            st.error("Upload a dataset first.")
        elif target_column is None or problem_type is None:
            st.error("Choose a target column and problem type in the 'EDA Analysis' tab first.")
        elif not metadata:
            st.error("Run EDA Analysis first so metadata is available for modelling.")
        elif target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in dataset.")
        else:

            setup_col1, setup_col2 = st.columns(2)
            setup_col1.metric("Target Column", target_column)
            setup_col2.metric("Problem Type", problem_type)

            st.info(
                "Modelling uses the raw dataset. "
                "Preprocessing is done inside the modelling pipeline after train-test split "
                "to avoid data leakage."
            )

            # Better Modelling removed per user request

            # Preview preprocessed dataset (leakage-safe: fit on train only)
            if st.button("Preview Preprocessed Dataset", use_container_width=True):
                
                with st.spinner("Preparing preprocessed preview..."):
                    X_train, X_test, y_train, y_test = split_data(df, target_column, problem_type)

                    X_train_p, X_test_p, y_train_p, preprocessing_report = preprocess_train_test_for_model(
                            X_train,
                            X_test,
                            outliers=True,
                            missing=True,
                            encoding=True,
                            scaling=True,
                            balancing=True,
                            metadata=metadata,
                            problem_type=problem_type,
                            y_train=y_train,
                            imputation_recommendations=
                                st.session_state
                                .get("ai_results", {})
                                .get(
                                    "imputation_recommendations",
                                    {}
                                )
                        )

                        # Reset indices and attach target column back for display
                    X_train_p = X_train_p.reset_index(drop=True)
                    X_test_p = X_test_p.reset_index(drop=True)
                    y_train_r = y_train.reset_index(drop=True)
                    y_test_r = y_test.reset_index(drop=True)

                    X_train_p[target_column] = y_train_r
                    X_test_p[target_column] = y_test_r

                    st.subheader("Preprocessed Dataset")
                    st.dataframe(X_train_p.head(10), use_container_width=True)

                    st.download_button(
                            label="Download Preprocessed Train",
                            data=X_train_p.to_csv(index=False).encode("utf-8"),
                            file_name="preprocessed_train.csv",
                            mime="text/csv",
                        )
                        
                #    st.subheader("Preprocessing Report")

                 #   report_rows = []

                  #  for key, value in preprocessing_report.items():
                        
                   #     if key=='balancing':
                    #        continue

                     #   if isinstance(value, list):
                      #      value = ", ".join(map(str, value))

                       # elif isinstance(value, dict):
                        #    value = json.dumps(value, indent=2)

                        #report_rows.append({"Preprocessing Step": key, "Details": value})

                   # report_df = pd.DataFrame(report_rows)

                    #st.dataframe(report_df, use_container_width=True, hide_index=True)

            
            if "model_results" in st.session_state:
                    st.subheader("Modelling Results")

                    for model_name, result in st.session_state["model_results"].items():
                        with st.expander(model_name):

                                        def _sanitize_display_value(v):
                                            if isinstance(v, pd.DataFrame):
                                                return f"DataFrame {v.shape}"
                                            if isinstance(v, pd.Series):
                                                return f"Series {v.shape}"
                                            if isinstance(v, np.ndarray):
                                                return f"ndarray {v.shape}"
                                            if isinstance(v, (list, tuple)):
                                                return f"{type(v).__name__} len={len(v)}"
                                            return v

                                        # hide raw training data and other large internals from the summary display
                                        exclude_keys = {
                                            "model",
                                            "actual_values",
                                            "predicted_values",
                                            "confusion_matrix",
                                            "feature_importance",
                                            "coefficients",
                                            "X_train",
                                            "X_test",
                                            "y_train",
                                            "y_test"
                                        }

                                        display_result = {
                                            k: _sanitize_display_value(v)
                                            for k, v in result.items()
                                            if k not in exclude_keys and not k.lower().startswith("x_")
                                        }

                                        result_df = pd.DataFrame(
                                            [
                                                {
                                                    "Metric": k,
                                                    "Value": str(v)
                                                }
                                                for k, v in display_result.items()
                                            ]
                                        )

                                        st.dataframe(
                                            result_df,
                                            use_container_width=True,
                                            hide_index=True
                                        )

                                        # Classification: Confusion Matrix
                                        if (result.get("status") == "success" and "confusion_matrix" in result):
                                            st.subheader("Confusion Matrix")
                                            cm_df = pd.DataFrame(result["confusion_matrix"])
                                            st.dataframe(cm_df)
                                            st.caption("""
                                                Rows = Actual Classes,
                                                Columns = Predicted Classes

                                                Diagonal values represent correct predictions.
                                                Off-diagonal values represent misclassifications.
                                                """)

                                        
                                        # Regression Error Histogram
                                        if (result.get("status") == "success" and "actual_values" in result and "predicted_values" in result):
                                            st.subheader("Prediction Error Distribution")

                                            actual = np.array(result["actual_values"])
                                            predicted = np.array(result["predicted_values"])
                                            errors = actual - predicted

                                            fig, ax = plt.subplots(figsize=(8, 4))
                                            ax.hist(errors, bins=20)
                                            ax.axvline(0, linestyle="--", color="red", label="Zero Error")
                                            ax.set_title("Prediction Error Distribution")
                                            ax.set_xlabel("Error (Actual - Predicted)")
                                            ax.set_ylabel("Counts")

                                            st.pyplot(fig)
                                            st.caption("Values near 0 indicate accurate predictions. Positive errors indicate underprediction. Negative errors indicate overprediction.")

                                        
                                        st.divider()
                                        
                                                
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            if (result.get("status") == "success" and "model" in result):
                                                buffer = io.BytesIO()
                                                joblib.dump(result["model"], buffer)
                                                st.download_button(
                                                    label="Download Model",
                                                    data=buffer.getvalue(),
                                                    file_name=(model_name.replace(" ", "_").lower() + ".pkl"),
                                                    mime="application/octet-stream",
                                                    key=f"download_{model_name}"
                                                )
                                    
                                    
                                    
    

if __name__ == "__main__":
    main()