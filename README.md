# AI Assisted Autom-EDA and Modelling Pipeline

This project is an end-to-end automated data analysis and machine learning workflow built with Streamlit. It allows a user to upload a dataset, run exploratory data analysis, use an AI-assisted analysis layer, generate visualizations, explain the results, and run baseline machine learning models.

The main purpose of the project is to make dataset understanding, EDA reporting, and initial modelling faster and more structured.

---

## Project Goals

- Accept a dataset through a simple web interface.
- Automatically profile the dataset.
- Detect column types and generate useful EDA reports.
- Use an AI layer to understand the dataset context.
- Detect the target column and problem type.
- Recommend useful visualizations.
- Generate graph summaries and AI-based explanations.
- Run classification or regression models.
- Display results in Streamlit.
- Export a downloadable EDA report.

---

## High-Level Workflow

```text
Dataset Upload
        ↓
DataFrame Creation
        ↓
EDA Profiling
        ↓
Metadata + EDA Reports
        ↓
AI Brain Analysis
        ↓
Target Column + Problem Type + Plot Plan
        ↓
Graph Generation + Graph Summaries
        ↓
EDA Explanation
        ↓
Modelling Pipeline
        ↓
Final Streamlit Output + Downloadable Report
```

---

## Project Structure

```text
Project Root
│
├── app.py
├── README.md
├── requirements.txt
│
├── .vscode
│   ├── launch.json
│   └── settings.json
│
├── AI_Brain
│   ├── .env
│   ├── brain_pipeline.py
│   ├── eda_explainer.py
│   ├── llm_client.py
│   ├── plot_recommender.py
│   ├── problem_type.py
│   ├── prompt_templates.py
│   └── target_detector.py
│
├── EDA
│   ├── eda_main.py
│   │
│   ├── Preprocess
│   │   ├── encoding.py
│   │   ├── missing_handler.py
│   │   ├── outlier_handling.py
│   │   ├── preprocess_pipeline.py
│   │   └── scaling.py
│   │
│   └── Profiling
│       ├── box_plots.py
│       ├── cardinality.py
│       ├── correlation.py
│       ├── eda_warnings.py
│       ├── metadata.py
│       ├── missing.py
│       ├── outlier.py
│       ├── profiling_pipeline.py
│       ├── skewness.py
│       ├── summary.py
│       ├── type_detection.py
│       └── visualization.py
│
└── Modelling
    ├── classification_model.py
    ├── evaluations.py
    ├── model_pipeline.py
    ├── regression_model.py
    ├── train_test_splitter.py
    └── __init__.py
```

---

## How the Dataset Moves Through the Project

This section explains how the dataset moves from one part of the project to another.

### 1. Dataset Upload

The user uploads a CSV or Excel file through the Streamlit interface.

```text
User Uploads Dataset
        ↓
app.py
```

The main application reads the uploaded file and converts it into a Pandas DataFrame.

```text
CSV / Excel File
        ↓
Pandas DataFrame
```

This DataFrame becomes the main dataset object used across the project.

---

### 2. DataFrame Sent to the EDA Profiling Module

After upload, the DataFrame is passed to the EDA profiling pipeline.

```text
app.py
        ↓
EDA Profiling Pipeline
```

The profiling pipeline sends the DataFrame to multiple EDA components.

```text
DataFrame
   ├── Dataset Summary
   ├── Missing Value Analysis
   ├── Column Type Detection
   ├── Outlier Detection
   ├── Skewness Analysis
   ├── Cardinality Analysis
   ├── Correlation Analysis
   └── Visualization Support
```

The output is stored as an `eda_results` dictionary.

```text
EDA Profiling Pipeline
        ↓
eda_results
```

The `eda_results` object contains reports such as:

- Dataset summary
- Missing value report
- Column information
- Outlier report
- Skewness report
- Cardinality report
- Metadata
- Plot paths
- Plot summaries

---

### 3. EDA Results Used to Build Metadata

The EDA results are used to create metadata.

```text
eda_results
        ↓
metadata
```

The metadata contains column-level information such as:

- Data type
- Missing count
- Missing percentage
- Unique count
- Sample values
- Detected column type
- Outlier information
- Skewness information
- Cardinality information

This metadata is much smaller than the full dataset and is used by the AI Brain.

---

### 4. Metadata and Sample Rows Sent to AI Brain

The full dataset is not required for every AI task. Instead, the AI Brain mainly receives:

```text
metadata + first few sample rows
```

The data flow is:

```text
app.py
        ↓
AI Brain Pipeline
        ↓
Target Detection
Problem Type Detection
Plot Recommendation
EDA Explanation
```

The AI Brain returns:

- Target column analysis
- Problem type analysis
- Plot recommendations
- EDA explanation

The result is stored as `ai_results`.

```text
AI Brain Pipeline
        ↓
ai_results
```

---

### 5. Plot Recommendations Sent to Graph Generation

The AI Brain recommends what type of plots should be generated.

```text
ai_results["plot_recommendations"]
        ↓
Graph Generation Module
```

The graph generation module creates the actual graphs using Python.

```text
Plot Recommendations
        ↓
Python Graph Generation
        ↓
Plot Images
```

The AI only recommends the plot type. Python validates the recommendation and generates the actual graph.

Supported plots include:

- Histogram
- Boxplot
- Line plot
- Bar chart
- Pie chart
- Scatter plot

---

### 6. Correlation Heatmap Generated Separately

The correlation heatmap is not generated from AI plot recommendations.

It is generated separately by the correlation analysis module.

```text
DataFrame + Column Information
        ↓
Correlation Analysis
        ↓
Correlation Heatmap
```

The heatmap uses continuous numerical columns and excludes target, identifier, binary, and categorical columns where required.

---

### 7. Graph Summaries Sent Back to AI Brain

After graphs are generated, Python creates graph summaries.

```text
Generated Graphs
        ↓
Graph Summaries
```

Graph summaries can include:

- Numeric distribution summary
- Mean, median, standard deviation
- Skewness
- Outlier count
- Top category values
- Scatter correlation value
- Strong correlation pairs from heatmap
- Optional graph alt-text

These summaries are sent to the AI Brain for explanation.

```text
Graph Summaries
        ↓
AI EDA Explainer
        ↓
Business-Friendly Explanation
```

This means the AI explains the graph summaries rather than blindly guessing from image files.

---

### 8. DataFrame Sent to Modelling Pipeline

After the AI Brain detects the target column and problem type, the raw DataFrame is passed to the modelling pipeline.

```text
DataFrame + Target Column + Problem Type
        ↓
Modelling Pipeline
```

The modelling pipeline handles:

```text
Train-Test Split
        ↓
Preprocessing
        ↓
Model Training
        ↓
Evaluation
```

The modelling result is returned to the Streamlit application.

```text
Modelling Pipeline
        ↓
Model Results
        ↓
Streamlit Output
```

---

## File Connection Map

This section explains which files are connected to which part of the project.

---

## Main Application Layer

### `app.py`

`app.py` is the main controller of the project.

It connects to:

```text
app.py
├── EDA Profiling Pipeline
├── AI Brain Pipeline
├── Graph Generation
├── Correlation Heatmap
├── Modelling Pipeline
└── HTML Report Generation
```

Main responsibilities:

- Loads uploaded dataset
- Converts file to DataFrame
- Calls EDA analysis
- Creates metadata
- Calls AI Brain
- Calls graph generation
- Calls correlation heatmap
- Calls modelling pipeline
- Displays results in Streamlit
- Generates downloadable report

---

## EDA Layer

### `EDA/Profiling/profiling_pipeline.py`

This is the main EDA controller.

It connects to:

```text
profiling_pipeline.py
├── summary.py
├── missing.py
├── type_detection.py
├── outlier.py
├── skewness.py
├── cardinality.py
├── correlation.py
├── visualization.py
└── box_plots.py
```

Its output goes back to:

```text
app.py
```

as:

```text
eda_results
```

---

### `EDA/Profiling/summary.py`

Connected to:

```text
profiling_pipeline.py
```

Purpose:

- Generates basic dataset summary.
- Counts rows, columns, duplicates, missing cells, memory usage, and data types.

---

### `EDA/Profiling/missing.py`

Connected to:

```text
profiling_pipeline.py
```

Purpose:

- Generates missing value report for each column.

---

### `EDA/Profiling/type_detection.py`

Connected to:

```text
profiling_pipeline.py
AI_Brain prompt templates
AI_Brain LLM client
```

Purpose:

- Detects column types such as continuous, categorical, binary, identifier, text, datetime, and unknown.
- Uses AI-assisted detection with rule-based fallback.
- Its output is stored as `column_info`.

This is important because many later components depend on `column_info`.

Connected components that use `column_info`:

```text
outlier.py
skewness.py
cardinality.py
correlation.py
visualization.py
metadata creation
AI Brain
```

---

### `EDA/Profiling/outlier.py`

Connected to:

```text
profiling_pipeline.py
type_detection.py output
```

Purpose:

- Detects outliers only for valid continuous columns.
- Skips identifier, binary, categorical, text, datetime, and target columns when configured.

---

### `EDA/Profiling/skewness.py`

Connected to:

```text
profiling_pipeline.py
type_detection.py output
```

Purpose:

- Measures skewness for continuous columns.
- Skips non-continuous columns.

---

### `EDA/Profiling/cardinality.py`

Connected to:

```text
profiling_pipeline.py
type_detection.py output
```

Purpose:

- Analyzes uniqueness and cardinality for categorical-like columns.
- Helps identify high-cardinality and identifier-like columns.

---

### `EDA/Profiling/correlation.py`

Connected to:

```text
app.py
profiling_pipeline.py
type_detection.py output
```

Purpose:

- Generates correlation heatmap for continuous numerical input columns.
- Produces graph summary for AI explanation.
- This module works separately from AI plot recommendations.

Output:

```text
correlation heatmap image
correlation plot summary
```

---

### `EDA/Profiling/visualization.py`

Connected to:

```text
app.py
AI plot recommendations
```

Purpose:

- Generates graphs recommended by the AI Brain.
- Creates graph images.
- Creates graph summaries.
- Optionally generates graph alt-text if supported.

Output:

```text
plot_paths
plot_summaries
```

---

### `EDA/Profiling/box_plots.py`

Connected to:

```text
profiling_pipeline.py
```

Purpose:

- Generates box plots for continuous columns if used by the pipeline.

---

### `EDA/Profiling/eda_warnings.py`

Connected to:

```text
profiling_pipeline.py
```

Purpose:

- Generates basic warnings about dataset quality.

---

### `EDA/Profiling/metadata.py`

Connected to:

```text
profiling_pipeline.py
AI Brain
```

Purpose:

- Builds metadata for AI analysis and reporting.

---

## AI Brain Layer

### `AI_Brain/brain_pipeline.py`

This is the main AI controller.

It connects to:

```text
brain_pipeline.py
├── target_detector.py
├── problem_type.py
├── plot_recommender.py
├── eda_explainer.py
├── llm_client.py
└── prompt_templates.py
```

Input:

```text
metadata
sample rows
eda_results
plot_summaries
```

Output:

```text
ai_results
```

`ai_results` may contain:

- Target column analysis
- Problem type analysis
- Plot recommendations
- EDA explanation

---

### `AI_Brain/target_detector.py`

Connected to:

```text
brain_pipeline.py
prompt_templates.py
llm_client.py
```

Purpose:

- Uses metadata and sample rows to identify the best target column.

---

### `AI_Brain/problem_type.py`

Connected to:

```text
brain_pipeline.py
prompt_templates.py
llm_client.py
```

Purpose:

- Detects whether the machine learning task is classification or regression.

---

### `AI_Brain/plot_recommender.py`

Connected to:

```text
brain_pipeline.py
prompt_templates.py
llm_client.py
visualization.py
```

Purpose:

- Recommends suitable plots.
- Does not generate graphs directly.
- Validates plot recommendation format.

Output goes to:

```text
visualization.py
```

---

### `AI_Brain/eda_explainer.py`

Connected to:

```text
brain_pipeline.py
prompt_templates.py
llm_client.py
eda_results
plot_summaries
```

Purpose:

- Explains EDA results in simple language.
- Explains skewness, outliers, cardinality, graphs, and modelling impact.
- Uses graph summaries instead of directly reading image files.

---

### `AI_Brain/prompt_templates.py`

Connected to:

```text
target_detector.py
problem_type.py
plot_recommender.py
eda_explainer.py
type_detection.py
```

Purpose:

- Stores prompts used by AI Brain tasks.

---

### `AI_Brain/llm_client.py`

Connected to:

```text
all AI_Brain modules
```

Purpose:

- Sends prompts to the configured LLM.
- Returns model responses.

---

## Preprocessing Layer

### `EDA/Preprocess/preprocess_pipeline.py`

This is the main preprocessing controller.

It connects to:

```text
preprocess_pipeline.py
├── missing_handler.py
├── outlier_handling.py
├── encoding.py
└── scaling.py
```

Purpose:

- Applies preprocessing steps in a structured way.

---

### `EDA/Preprocess/missing_handler.py`

Purpose:

- Handles missing values.

---

### `EDA/Preprocess/outlier_handling.py`

Purpose:

- Caps or handles outliers while protecting the target column.

---

### `EDA/Preprocess/encoding.py`

Purpose:

- Encodes categorical and binary columns.
- Drops identifier columns where required.

---

### `EDA/Preprocess/scaling.py`

Purpose:

- Scales numerical features when required.

---

## Modelling Layer

### `Modelling/model_pipeline.py`

This is the main modelling controller.

It connects to:

```text
model_pipeline.py
├── train_test_splitter.py
├── classification_model.py
├── regression_model.py
├── evaluations.py
└── preprocessing components
```

Input:

```text
raw DataFrame
target column
problem type
```

Output:

```text
model_results
```

---

### `Modelling/train_test_splitter.py`

Purpose:

- Splits dataset into training and testing sets.
- Uses stratified split for classification when possible.
- Uses normal split for regression.

---

### `Modelling/classification_model.py`

Purpose:

- Provides classification models.

Supported examples:

- Logistic Regression
- Support Vector Classifier
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors Classifier

---

### `Modelling/regression_model.py`

Purpose:

- Provides regression models.

---

### `Modelling/evaluations.py`

Purpose:

- Evaluates models based on the problem type.
- Generates classification or regression metrics.

---

## End-to-End File Flow

```text
app.py
  ↓
Uploaded dataset converted to DataFrame
  ↓
EDA/Profiling/profiling_pipeline.py
  ↓
summary.py
missing.py
type_detection.py
outlier.py
skewness.py
cardinality.py
  ↓
eda_results returned to app.py
  ↓
metadata created from eda_results
  ↓
AI_Brain/brain_pipeline.py
  ↓
target_detector.py
problem_type.py
plot_recommender.py
  ↓
plot recommendations returned to app.py
  ↓
EDA/Profiling/visualization.py
  ↓
recommended graphs + graph summaries generated
  ↓
EDA/Profiling/correlation.py
  ↓
correlation heatmap + heatmap summary generated
  ↓
AI_Brain/brain_pipeline.py
  ↓
eda_explainer.py explains EDA results and graph summaries
  ↓
app.py displays EDA report and graphs
  ↓
Modelling/model_pipeline.py
  ↓
train_test_splitter.py
classification_model.py / regression_model.py
evaluations.py
  ↓
model results returned to app.py
```

---

## Data Objects Passed Between Components

| Data Object | Created By | Used By | Purpose |
|---|---|---|---|
| `df` | `app.py` | EDA, AI Brain, Modelling, Graph Generation | Main uploaded dataset |
| `eda_results` | EDA profiling pipeline | app, AI Brain, report generator | Stores EDA reports and analysis outputs |
| `column_info` | Column type detection | outlier, skewness, cardinality, correlation, metadata | Stores detected column types |
| `metadata` | app / EDA metadata builder | AI Brain | Compact dataset summary for LLM |
| `ai_results` | AI Brain pipeline | app, modelling setup, graph generation | Stores AI outputs |
| `plot_recommendations` | AI Brain plot recommender | visualization module | Tells Python which graphs to generate |
| `plot_paths` | visualization and correlation modules | Streamlit and HTML report | Stores graph image paths |
| `plot_summaries` | visualization and correlation modules | AI EDA explainer | Helps AI explain graphs |
| `model_results` | modelling pipeline | Streamlit output | Stores model metrics and evaluation results |

---

## Data Leakage Handling

The modelling part is designed to reduce data leakage.

Correct modelling flow:

```text
Raw Dataset
        ↓
Train-Test Split
        ↓
Fit preprocessing only on training data
        ↓
Transform training and testing data separately
        ↓
Train model
        ↓
Evaluate on test data
```

This prevents information from the test set from leaking into training.

---

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Optional Graph Alt-Text Support

If graph alt-text is used:

```bash
pip install matplotalt
```

If required by your environment:

```bash
pip install torch torchvision
```

The project should still work without graph alt-text if fallback handling is implemented.

---

## Environment Variables

The AI Brain uses an LLM client. Store API keys and model configuration in:

```text
AI_Brain/.env
```

Example:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=your_model_name_here
```

Do not commit real API keys to version control.

---

## How to Run

```bash
streamlit run app.py
```

Then:

1. Upload a CSV or Excel dataset.
2. Click **Run EDA Analysis**.
3. Review EDA reports, AI insights, and graphs.
4. Download the EDA report if required.
5. Click **Run Modelling** after EDA analysis is completed.

---

## Expected Output

The Streamlit app displays:

- Dataset preview
- Dataset summary
- Missing value report
- Column information
- Skewness report
- Outlier report
- Cardinality report
- Target column
- Problem type
- AI-recommended plots
- Correlation heatmap
- Graph summaries
- AI EDA explanation
- Model evaluation results
- Downloadable HTML report

---

## Important Design Note

The AI Brain is used as an assistant, not as the only decision-maker.

```text
AI Brain recommends and explains
Python validates and executes
```

This makes the project more reliable because graph generation, preprocessing, EDA calculations, and modelling are handled using Python logic.

---

## Limitations

- AI output quality depends on prompt quality and selected model.
- Very small datasets may produce weak model results.
- Highly imbalanced classification datasets may require additional handling.
- Correlation heatmap requires at least two valid continuous numerical columns.
- Plot recommendations should always be validated by Python.
- Graph alt-text may require additional dependencies.
- The current system is designed for tabular datasets.

---

## Future Improvements

- User-selected target override.
- Better model comparison dashboard.
- Cross-validation support.
- Time-series split support.
- Feature importance plots.
- PDF report export.
- Model download option.
- Class imbalance handling.
- Automated feature engineering.
- Interactive graph controls.
- More advanced graph explanations.

---

## Summary

This project provides a modular Auto EDA and modelling workflow. The dataset starts from a Streamlit upload, moves through EDA profiling, AI Brain analysis, graph generation, explanation, and modelling, and finally returns reports and results to the user.

The system is designed to be understandable, extensible, and suitable for explaining both technical and business-level insights.
