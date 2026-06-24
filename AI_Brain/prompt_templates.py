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
            
            
def build_imputation_recommendation_prompt(metadata):
    """
    Generates column-wise missing value handling recommendations.

    Returns JSON only.
    """

    return f"""
You are an expert Data Scientist.

Your task is to recommend the most appropriate missing-value handling strategy
for every column in a dataset.

Rules:

NUMERIC COLUMNS:
Choose exactly one:
- mean
- median
- knn
- zero
- drop

CATEGORICAL COLUMNS:
Choose exactly one:
- mode
- unknown
- drop

TEXT COLUMNS:
Choose exactly one:
- unknown
- drop


IDENTIFIER COLUMNS:
Choose exactly one:
- drop

Decision Guidelines:

Numeric:
- mean → low skewness and low missing %
- median → skewed distributions
- knn → correlated numeric features and moderate missing %
- zero → when missing likely means absence/count=0
- drop → very high missing %

Categorical:
- mode → low/moderate missing %
- unknown → missing itself may contain information
- drop → very high missing %

Text:
- unknown or drop

Datetime:
- forward_fill when appropriate
- drop if unusable

Return ONLY valid JSON.

Example:

{{
  "Age": {{
    "strategy": "median",
    "reason": "Right-skewed distribution"
  }},
  "Income": {{
    "strategy": "knn",
    "reason": "Correlated with multiple numeric features"
  }},
  "Gender": {{
    "strategy": "mode",
    "reason": "Low missing percentage"
  }}
}}

Dataset Metadata:

{metadata}
"""