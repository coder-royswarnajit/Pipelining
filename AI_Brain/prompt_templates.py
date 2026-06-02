def target_column_prompt(metadata, sample_rows):
    prompt = f"""You are an expert Machine Learning Engineer.

                    Your task is to analyze the dataset metadata and first 5 rows, then identify the most likely target column or target columns.

                    You must decide based on:
                    1. Column names
                    2. Data types
                    3. Unique values
                    4. Missing values
                    5. Meaning of first 5 rows
                    6. Whether the column looks like an output/label/result/status/failure column

                    Dataset Metadata:
                    {metadata}

                    First 5 Rows:
                    {sample_rows}

                    Return your answer in this format only:

                    Possible Target Columns:
                    - column_name: reason

                    Best Target Column:
                    column_name
                    """
    return prompt


def problem_type_prompt(metadata, sample_rows, target_column):
    prompt = f"""You are an expert Machine Learning Engineer.

                    Your task is to decide whether the machine learning problem is:
                    1. Classification
                    2. Regression
                    3. Unknown

                    Use the given metadata, first 5 rows, and selected target column.

                    Dataset Metadata:
                    {metadata}

                    First 5 Rows:
                    {sample_rows}

                    Selected Target Column:
                    {target_column}

                    Rules:
                    - If the target column has categories, labels, classes, yes/no, 0/1, failure/no failure,
                    then it is a classification problem.
                    - If the target column is continuous numerical output,
                    then it is a regression problem.
                    - If the target column is unclear, say unknown.

                    Return your answer in this format only: classification / regression / unknown

                    """
    return prompt

def column_type_detection_prompt(column_metadata):
    prompt = f"""You are an expert Data Scientist.

                Your task is to detect the semantic type of each dataset column.
                    Allowed detected_type values only:
                    - empty
                    - binary
                    - categorical
                    - categorical_numeric
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

                    4. categorical_numeric:
                    - numeric values that behave like categories/classes
                    - example: 0, 1, 2, 3 rating/category codes

                    5. continuous:
                    - numeric measurement columns
                    - values represent quantities, sensor readings, price, age, temperature, pressure, etc.

                    6. datetime:
                    - date/time columns

                    7. text:
                    - long free-text columns or descriptions

                    8. empty:
                    - column has no non-null values

                    9. unknown:
                    - use only when unclear

                    Important:
                    - Do not classify numeric target-like columns as identifier just because they are unique.
                    - If a column is named x and numeric, it is usually continuous.
                    - If a column is named y and numeric with many unique values, it is usually continuous.
                    - If a column is named y but has few repeated classes, it may be categorical_numeric or binary.
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

4. categorical_numeric:
   - bar

5. identifier:
   - skip

6. text:
   - skip

7. datetime:
   - line only if paired with a numeric column

8. regression target:
   - recommend scatter plots between continuous input features and the target

9. classification target:
   - recommend bar plots or boxplots grouped by target if suitable

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
7. Graph/plot interpretation
8. Impact on modelling
9. Recommended next steps

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

Graph Insights:
...

Modelling Impact:
...

Recommended Next Steps:
...
"""