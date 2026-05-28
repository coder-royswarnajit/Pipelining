import os
import sys
import io
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# ==============================
# PATH SETUP
# ==============================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EDA_PATH = os.path.join(PROJECT_ROOT, "EDA")
AI_BRAIN_PATH = os.path.join(PROJECT_ROOT, "AI_Brain")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if EDA_PATH not in sys.path:
    sys.path.append(EDA_PATH)

if AI_BRAIN_PATH not in sys.path:
    sys.path.append(AI_BRAIN_PATH)

# ==============================
# IMPORT PIPELINES
# ==============================

from EDA.eda_main import run_eda_workflow
from AI_Brain.brain_pipeline import run_ai_brain_pipeline


# ==============================
# PDF REPORT FUNCTION
# ==============================
def show_eda_section(title, data):
    """
    Displays EDA output properly in Streamlit.
    """

    st.markdown(f"### {title}")

    if isinstance(data, pd.DataFrame):
        st.dataframe(data, use_container_width=True)

    elif isinstance(data, pd.Series):
        st.dataframe(data.to_frame(), use_container_width=True)

    elif isinstance(data, dict):
        try:
            df_data = make_streamlit_safe_dataframe(data)
            st.dataframe(df_data, use_container_width=True)
        except Exception:
            st.json(data)

    elif isinstance(data, list):
        if len(data) == 0:
            st.info("No items found.")
        else:
            for item in data:
                st.write(item)

    else:
        st.write(data)
def make_pdf_report(df, eda_output, ai_results): 
    """
    Creates a downloadable PDF report in memory.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Brain EDA Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Dataset overview
    story.append(Paragraph("Dataset Overview", styles["Heading2"]))
    story.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # First 5 rows
    story.append(Paragraph("First 5 Rows", styles["Heading2"]))

    head_df = df.head(5).copy().astype(str)
    table_data = [list(head_df.columns)] + head_df.values.tolist()

    table = Table(table_data, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 14))

    # AI Brain output
    story.append(Paragraph("AI Brain Analysis", styles["Heading2"]))

    if isinstance(ai_results, dict):
        for key, value in ai_results.items():
            story.append(
                Paragraph(
                    str(key).replace("_", " ").title(),
                    styles["Heading3"]
                )
            )

            safe_text = str(value).replace("\n", "<br/>")
            story.append(Paragraph(safe_text, styles["BodyText"]))
            story.append(Spacer(1, 10))
    else:
        safe_text = str(ai_results).replace("\n", "<br/>")
        story.append(Paragraph(safe_text, styles["BodyText"]))

    story.append(Spacer(1, 14))

    # Preprocessed shape
    story.append(Paragraph("Preprocessing Output", styles["Heading2"]))

    preprocessed_df = eda_output.get("preprocessed_df")

    if preprocessed_df is not None:
        story.append(
            Paragraph(
                f"Preprocessed Shape: {preprocessed_df.shape[0]} rows, "
                f"{preprocessed_df.shape[1]} columns",
                styles["Normal"]
            )
        )
    else:
        story.append(
            Paragraph(
                "Preprocessed data not available.",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 14))

    # Metadata summary
    story.append(Paragraph("Metadata Summary", styles["Heading2"]))

    metadata = eda_output.get("metadata", {})

    metadata_rows = [
        ["Column", "Dtype", "Unique", "Missing %", "Detected Type"]
    ]

    for col, info in metadata.items():
        metadata_rows.append(
            [
                str(col),
                str(info.get("dtype", "")),
                str(info.get("unique_count", "")),
                str(info.get("missing_percent", "")),
                str(info.get("detected_type", "")),
            ]
        )

    metadata_table = Table(metadata_rows, repeatRows=1)

    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 14))

    # Graphs in PDF
    story.append(Paragraph("EDA Graphs", styles["Heading2"]))

    plot_paths = eda_output.get("eda_results", {}).get("plot_paths", [])

    if len(plot_paths) > 0:
        for plot_path in plot_paths:
            if os.path.exists(plot_path):
                story.append(
                    Paragraph(
                        os.path.basename(plot_path),
                        styles["Heading3"]
                    )
                )

                try:
                    img = Image(plot_path)
                    img.drawHeight = 280
                    img.drawWidth = 450
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception:
                    story.append(
                        Paragraph(
                            f"Could not load graph: {plot_path}",
                            styles["Normal"]
                        )
                    )
    else:
        story.append(
            Paragraph(
                "No graphs were generated.",
                styles["Normal"]
            )
        )

    doc.build(story)

    buffer.seek(0)

    return buffer

def make_streamlit_safe_dataframe(data):
    """
    Converts dict/list/object-heavy data into a Streamlit-safe DataFrame.
    Prevents PyArrow conversion errors from mixed object columns.
    """

    df_safe = pd.DataFrame(data).T

    for col in df_safe.columns:
        df_safe[col] = df_safe[col].apply(
            lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x
        )

    return df_safe.astype(str)


# ==============================
# STREAMLIT APP
# ==============================

def main():

    st.set_page_config(
        page_title="AI Brain EDA Dashboard",
        layout="wide"
    )

    st.title("AI Brain EDA Dashboard")

    st.write(
        "Upload a CSV file. The app will run EDA, preprocessing, "
        "AI Brain analysis, show graphs, and generate a PDF report."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to start.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    st.success("CSV uploaded successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    run_button = st.button("Generate EDA Report")

    if run_button:

        plot_folder = os.path.join(PROJECT_ROOT, "eda_plots")

        with st.spinner("Running EDA and preprocessing..."):
            eda_output = run_eda_workflow(
                df=df,
                save_plots=True,
                plot_folder=plot_folder
            )

        st.success("EDA and preprocessing completed.")

        metadata = eda_output["metadata"]

        with st.spinner("Running AI Brain analysis..."):
            ai_results = run_ai_brain_pipeline(
                metadata=metadata,
                df=df
            )

        st.success("AI Brain analysis completed.")

        # ==============================
        # AI OUTPUT
        # ==============================

        st.subheader("AI Brain Output")

        if isinstance(ai_results, dict):
            for key, value in ai_results.items():
                st.markdown(f"### {key.replace('_', ' ').title()}")
                st.write(value)
        else:
            st.write(ai_results)
        # ==============================
        # FULL EDA RESULTS
        # ==============================

        st.subheader("Complete EDA Results")

        eda_results = eda_output.get("eda_results", {})

        eda_display_order = [
            "dataset_summary",
            "missing_report",
            "warnings",
            "column_types",
            "outlier_report",
            "skewness_report",
            "cardinality_report",
        ]

        for key in eda_display_order:
            if key in eda_results:
                section_title = key.replace("_", " ").title()

                with st.expander(section_title, expanded=False):
                    show_eda_section(section_title, eda_results[key])
        # ==============================
        # PREPROCESSED DATA
        # ==============================

        st.subheader("Preprocessed Dataset Preview")

        preprocessed_df = eda_output.get("preprocessed_df")

        if preprocessed_df is not None:
            st.dataframe(
                preprocessed_df.head(10),
                use_container_width=True
            )
        else:
            st.info("Preprocessed dataset not available.")

        # ==============================
        # METADATA
        # ==============================

        st.subheader("Metadata Preview")

        metadata_df = make_streamlit_safe_dataframe(metadata)
        st.dataframe(metadata_df, use_container_width=True)

        # ==============================
        # EDA GRAPHS
        # ==============================

        st.subheader("EDA Graphs")

        plot_paths = eda_output["eda_results"].get("plot_paths", [])

        if len(plot_paths) > 0:
            for plot_path in plot_paths:

                if os.path.exists(plot_path):
                    st.image(
                        plot_path,
                        caption=os.path.basename(plot_path),
                        use_container_width=True
                    )
                else:
                    st.warning(f"Graph file not found: {plot_path}")
        else:
            st.info("No graphs were generated.")

        # ==============================
        # PDF DOWNLOAD
        # ==============================

        pdf_buffer = make_pdf_report(
            df=df,
            eda_output=eda_output,
            ai_results=ai_results
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="ai_brain_eda_report.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()