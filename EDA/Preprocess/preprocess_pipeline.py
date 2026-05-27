from Preprocess.outlier_handling import handle_outliers
from Preprocess.missing_handler import handle_missing_values
from Preprocess.encoding import encode_features



def run_preprocessing_pipeline(df, outliers=True, missing=True, encoding=True):

    print("\nSTARTING PREPROCESSING PIPELINE")


    if outliers:
        print("\nHandling Outliers")
        df = handle_outliers(df)

    if missing:
        print("\nHandling Missing Values")
        df = handle_missing_values(df)

    if encoding:
        print("\nEncoding Features")
        df = encode_features(df)

    print("\nPREPROCESSING COMPLETED")

    return df