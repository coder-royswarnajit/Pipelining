import pandas as pd

'''Shows total number of missing values in each columns'''
def missing_report(df):
    missing = pd.DataFrame({'Missing Count':df.isnull().sum(),
                            'Missing %':(df.isnull().sum()/len(df))*100})

    return missing.sort_values(by='Missing %', ascending=False)
