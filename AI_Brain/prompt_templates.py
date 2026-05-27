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