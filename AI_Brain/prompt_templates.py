import json


def column_type_detection_prompt(column_metadata, sample_rows):
    prompt = f"""You are an expert Data Scientist.

                Your task is to detect the semantic type of each dataset column.
                    Allowed detected_type values only:
                    - empty
                    - binary
                    - categorical
                    - continuous
                    - datetime
                    - identifier
                    - text
                    - unknown

                    Use the following rules:
                    1. identifier:
                    - ID-like columns
                    - serial numbers
                    - product IDs
                    - columns with mostly unique values
                    - names or codes that identify rows

                    2. binary:
                    - columns with exactly two meaningful values
                    - yes/no, true/false, 0/1

                    3. categorical:
                    - non-numeric class/category columns
                    - low or moderate number of repeated labels

                    4. continuous:
                    - numeric measurement columns
                    - values represent quantities, sensor readings, price, age, temperature, pressure, etc.

                    5. datetime:
                    - date/time columns

                    6. text:
                    - long free-text columns or descriptions

                    7. empty:
                    - column has no non-null values

                    8. unknown:
                    - use only when unclear

                    Important:
                    - Do not classify numeric target-like columns as identifier just because they are unique.
                    - If a column is named x and numeric, it is usually continuous.
                    - If a column is named y and numeric with many unique values, it is usually continuous.
                    - If a column is named y but has few repeated classes, it may be categorical or binary.
                    - Return valid Python dictionary format only.
                    - Do not include explanations outside the dictionary.

                    Column Metadata:
                    {column_metadata}
                    
                    Sample Rows:
                    {sample_rows}

                    Return only in this exact format:

                    {{
                        "column_name": "detected_type",
                        "column_name": "detected_type"
                    }}
                    """

    return prompt

def plot_recommendation_prompt(metadata, target_column=None, problem_type=None):
    return f"""
You are an expert Data Scientist.

Your task is to recommend suitable EDA plots using only column metadata.

Do not ask for the full dataset.
Do not generate Python code.
Only recommend plot types.

Allowed plot types:
- histogram
- boxplot
- line
- bar
- pie
- scatter
- skip

Rules:
1. continuous column:
   - histogram for distribution
   - boxplot for outliers
   - line if record order trend is useful

2. binary column:
   - pie or bar

3. categorical column:
   - bar preferred
   - pie only if very low cardinality

4. identifier:
   - skip

5. text:
   - skip

6. datetime:
   - line only if paired with a numeric column

7. regression target:
   - recommend scatter plots between continuous input features and the target

8. classification target:
   - recommend bar plots or boxplots grouped by target if suitable, but not scatter plot

Important:
- Use only these plot types: histogram, boxplot, line, bar, pie, scatter, skip
- Do not recommend plots for identifier columns.
- Do not recommend the target column as a normal input feature plot unless useful for target distribution.
- Return only a valid Python list of dictionaries.
- Do not include explanation outside the list.

Metadata:
{metadata}

Target Column:
{target_column}

Problem Type:
{problem_type}

Return only in this exact format:

[
    {{
        "plot_type": "histogram",
        "columns": ["column_name"],
        "title": "Plot title"
    }},
    {{
        "plot_type": "scatter",
        "columns": ["feature_column", "target_column"],
        "title": "Plot title"
    }}
]
"""



def eda_explanation_prompt(eda_summary, target_column=None, problem_type=None):
    return f"""
        You are an expert Data Analyst and Machine Learning Engineer.

        Your task is to explain the EDA results in simple, clear language.

        The explanation should be useful for:
        - a data science intern
        - a mentor reviewing the project
        - a business user who wants to understand the dataset

        Do not write code.
        Do not mention that you are an AI.
        Do not over-explain basic definitions.

        Focus on:
        1. Dataset overview
        2. Missing values
        3. Column types
        4. Skewness interpretation
        5. Outlier interpretation
        6. Cardinality interpretation

        Important rules:
        - If skewness report is empty, say that no major skewness analysis was available or no valid continuous columns were found.
        - If outlier report is empty, say that no significant outlier information was found.
        - If cardinality report has identifier-like columns, mention that such columns should usually be removed before modelling.
        - If target column is available, explain how EDA relates to that target.
        - If problem type is available, explain what the EDA suggests for that ML task.
        - Keep the explanation practical and project-focused.
        - Use short headings.
        - Use bullet points where useful.
        - Keep it concise but informative.

        EDA Results:
        {eda_summary}

        Target Column:
        {target_column}

        Problem Type:
        {problem_type}

        Return the explanation in this format:

        Dataset Overview:
        ...

        Missing Values:
        ...

        Column Type Insights:
        ...

        Skewness Insights:
        ...

        Outlier Insights:
        ...

        Cardinality Insights:
        ...

        """

def plot_explanation_prompt(plot_payload):
    return f"""
            You are an expert Data Analyst.

            You will receive a compact batch of plot metadata and statistics.

            Your task is to explain what the data is saying in simple, natural language.

            Rules:
            - Focus on interpreting the data.
            - Explain patterns, trends, concentration, spread, variability, relationships, and unusual observations.
            - Do NOT explain what a graph type is.
            - Do NOT mention terms such as histogram, scatter plot, box plot, bar chart, or pie chart.
            - Do NOT provide business recommendations.
            - Do NOT provide action items.
            - Do NOT use phrases like "this graph shows" or "this chart shows".
            - Keep each explanation between 1 to 2 sentences.
            - Make explanations understandable to non-technical users.
            - Make the headings properly visible and clear.
            - Don't include takeaways or recommendations, just insights.
            - Don't suggest anything to do with the data, just explain what the data is saying.
            - Return ONLY valid JSON.

            Expected Format:

            {{
                "plot_key": {{
                    "summary": "Explanation here"
                }}
            }}

            Return one top-level key per plot_key in the batch.

            Visualization Information:

            {json.dumps(plot_payload, separators=(",", ":"))}
            """
            
            
def build_imputation_recommendation_prompt(metadata, sample_rows):
    """
    Generates column-wise missing value handling recommendations.

    Returns JSON only.
    """

    return f"""
You are a Senior Data Scientist responsible for designing a robust data preprocessing pipeline.

Your task is to recommend EXACTLY ONE missing-value handling strategy for EACH column in the dataset.

You must use the provided metadata to determine the most appropriate strategy.

Do NOT generate Python code.
Do NOT ask for additional information.
Return ONLY valid JSON.

--------------------------------------------------
AVAILABLE STRATEGIES
--------------------------------------------------

NUMERIC COLUMNS:
- mean
- median
- knn
- zero
- drop

CATEGORICAL COLUMNS:
- mode
- unknown
- drop

TEXT COLUMNS:
- unknown
- drop

IDENTIFIER COLUMNS:
- drop

DATETIME COLUMNS:
- forward_fill
- drop

--------------------------------------------------
DECISION RULES
--------------------------------------------------

GENERAL:

- Every column must receive exactly ONE strategy.

- Consider:
  - Missing percentage
  - Column type
  - Cardinality
  - Distribution characteristics
  - Business usefulness
  - Whether the column appears predictive

- Prefer preserving useful information whenever reasonable.

- Use "drop" only when there is strong evidence the column provides little value or contains excessive missing values.

--------------------------------------------------
NUMERIC COLUMNS
--------------------------------------------------

Use "mean" when:
- Distribution appears approximately symmetric.
- Missing percentage is low.
- Outliers do not dominate the column.

Use "median" when:
- Distribution is skewed.
- Outliers are present.
- Robust imputation is preferred.

Use "knn" when:
- Missing percentage is moderate.
- Column appears related to other numeric features.
- Preserving feature relationships is important.

Use "zero" when:
- Missing values likely represent absence.
- Counts, frequencies, quantities, or event occurrences.
- Zero is a meaningful value.

Use "drop" when:
- Missing percentage is extremely high.
- Column usefulness appears limited.

--------------------------------------------------
CATEGORICAL COLUMNS
--------------------------------------------------

Use "mode" when:
- Missing percentage is low to moderate.
- Most common category is representative.

Use "unknown" when:
- Missingness may carry information.
- Missing values could represent a distinct category.
- The dataset may benefit from retaining missing-value signals.

Use "drop" when:
- Missing percentage is extremely high.
- Column appears unlikely to contribute meaningful information.

--------------------------------------------------
TEXT COLUMNS
--------------------------------------------------

Use "unknown" when:
- Text content may still be useful after imputation.
- Missing values can be represented as a placeholder.

Use "drop" when:
- Missing percentage is extremely high.
- Column appears unusable.

--------------------------------------------------
IDENTIFIER COLUMNS
--------------------------------------------------

Use "drop" for:
- IDs
- UUIDs
- Transaction IDs
- Record IDs
- Other unique identifiers


--------------------------------------------------
IMPORTANT
--------------------------------------------------

- Recommend exactly one strategy per column.
- Every recommendation must include a concise reason.
- Base decisions ONLY on supplied metadata.
- Be consistent across similar columns.
- Return valid JSON only.
- No markdown.
- No explanations outside JSON.

Expected Output Format:

{{
    "column_name": {{
        "strategy": "median",
        "reason": "Skewed numeric distribution with moderate missing values."
    }},
    "another_column": {{
        "strategy": "unknown",
        "reason": "Missingness may represent a meaningful category."
    }}
}}

--------------------------------------------------
DATASET METADATA
--------------------------------------------------

{metadata}

 SAMPLE ROWS
 {sample_rows}
"""


def build_metric_recommendation_prompt(
    metadata,
    sample_rows,
    target_column=None,
    problem_type=None,
    target_context=None,
    target_distribution=None,
    dataset_statistics=None,
):
    if problem_type == "classification":
        allowed_metrics = "accuracy, precision, recall, f1_score"

        metric_guidance = """
Metric Definitions:

- accuracy:
  Use when classes are reasonably balanced and false positives and false negatives
  have similar business impact.

- precision:
  Use when false positives are more costly than false negatives.
  Examples: spam filtering, expensive customer outreach, manual review systems.

- recall:
  Use when false negatives are more costly than false positives.
  Examples: fraud detection, disease screening, safety monitoring,medical datasets.

- f1_score:
  Use when both precision and recall matter and a balanced tradeoff is required.
  Common choice for imbalanced classification problems.
"""

    else:
        allowed_metrics = "mae, mse, rmse, r2_score"

        metric_guidance = """
Metric Definitions:

- mae:
  Measures average absolute prediction error.
  Easy to interpret and robust to outliers.

- rmse:
  Penalizes large errors more heavily.
  Suitable when large prediction mistakes are especially costly.

- mse:
  Similar to RMSE but expressed in squared units.
  Useful for optimization-focused evaluation.

- r2_score:
  Measures how much variance in the target is explained by the model.
  Useful when explanatory power matters more than prediction error.
"""

    return f"""
You are a senior Machine Learning Engineer responsible for selecting the most appropriate
PRIMARY evaluation metric for model comparison and ranking.

Your goal is NOT to choose a mathematically popular metric.

Your goal is to choose the metric that best aligns with:

1. Dataset characteristics
2. Target distribution
3. Business objective
4. Error costs
5. Class imbalance (if classification)
6. Outliers and skewness (if regression)

You must reason like a real-world ML practitioner.

----------------------------------------
AVAILABLE METRICS
----------------------------------------

{allowed_metrics}

{metric_guidance}

----------------------------------------
DATASET INFORMATION
----------------------------------------

Metadata:
{metadata}

Dataset Statistics:
{dataset_statistics}

Sample Rows:
{sample_rows}

Target Column:
{target_column}

Problem Type:
{problem_type}

Target Distribution:
{target_distribution}

Business Context:
{target_context}

----------------------------------------
DECISION RULES
----------------------------------------

For Classification:

- If the dataset is heavily imbalanced,
  avoid accuracy unless there is strong evidence otherwise.

- If missing a positive case is more costly than a false alarm,
  prefer recall.

- If false alarms are more costly than missed positives,
  prefer precision.

- If both error types matter,
  prefer f1_score.

- Use accuracy only when classes are reasonably balanced and
  error costs appear similar.

For Regression:

- If target values contain significant outliers or skewness,
  prefer mae unless business requirements suggest otherwise.

- If large prediction errors are especially harmful,
  prefer rmse.

- Use mse only when emphasis is on optimization sensitivity
  rather than interpretability.

- Use r2_score when explaining variance is more important than
  prediction accuracy.

----------------------------------------
IMPORTANT
----------------------------------------

- Select EXACTLY ONE primary metric.
- Do not recommend multiple metrics.
- Do not generate code.
- Do not ask for additional information.
- Base your decision only on the supplied information.
- If business context is ambiguous, choose the most robust metric
  for the observed data characteristics.

Return ONLY valid JSON.

{{
    "metric": "<selected_metric>",
    "reason": "<concise explanation>",
    "confidence": "high|medium|low"
}}
"""

def shap_explanation_prompt(
    shap_analysis,
    target_column=None,
    problem_type=None,
    model_name=None,
):
    return f"""
You are a senior Data Scientist explaining model behavior to a business audience.

Your task is to convert compact SHAP analysis results into practical business context.

Do NOT write code.
Do NOT mention that you are an AI.
Do NOT explain the SHAP algorithm.
Do NOT claim causation. Use wording such as "associated with", "influenced", or
"the model relied on".

Focus on:
1. Which features had the strongest influence on the best model, using importance_pct as the contribution percentage.
2. Whether each influential feature generally pushed predictions up or down.
3. What this means in simple business language.
4. Any caution needed when interpreting transformed or encoded feature names.

Keep the explanation concise and useful for a project report. Prefer business language such as "The model relies most heavily on..." over raw SHAP terminology.

SHAP Analysis:
{json.dumps(shap_analysis, separators=(",", ":"))}

Target Column:
{target_column}

Problem Type:
{problem_type}

Best Model:
{model_name}

Return the explanation in this format:

Model Explanation:
...

Most Influential Factors:
- ...

Business Interpretation:
...

Important Caution:
...
"""


def optimization_explanation_prompt(
    model_name,
    problem_type,
    optimization_result,
    shap_analysis,
):
    return f"""
You are an expert machine learning assistant.

The best model is:

{model_name}

Problem Type:
{problem_type}

Optimization Summary:
{optimization_result}

SHAP Summary:
{shap_analysis}

Explain:

1. Why Optuna improved this model.
2. Which hyperparameters had the biggest effect.
3. What the optimization results indicate.
4. Which features are most important according to SHAP.
5. Explain everything in simple business language.

Keep the explanation concise.
"""