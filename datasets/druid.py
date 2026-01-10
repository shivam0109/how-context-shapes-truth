"""
Script to get the data from DRUID dataset 
"""

import pandas as pd
from datasets import load_dataset
import re 
import os 

BASE_DIR = os.path.dirname('druid/')

def get_druid_data():
    ds_druid = load_dataset("copenlu/druid", "DRUID")
    df_druid = ds_druid['train'].to_pandas()
    return df_druid

def split_on_punctuation_take_first(text):
    # Split on any punctuation character
    parts = re.split(r'[\W_]+', text)
    return parts[0].strip() if parts else ''


def preprocess_druid_data(df_druid):
    """
    Preprocess the DRUID data
    1. Select the relevant columns. 
    2. Remove non-relevant rows.  
    3. Remove duplicate evidences.  
    4. Add factcheck website column 
    """
    df_druid = df_druid[['id','claim_id','claim','claimant','evidence',
                     'factcheck_verdict','is_gold','relevant','evidence_stance']].copy()
    print("Initial shape: ", df_druid.shape)

    df_druid = df_druid[df_druid['relevant']].copy()
    print("Shape after removing non-relevant rows: ", df_druid.shape)

    df_druid = df_druid[~df_druid.duplicated(subset='evidence', keep='first')]
    print("Shape after removing duplicate evidences: ", df_druid.shape) 

    df_druid['fc_source'] = df_druid['claim_id'].apply(lambda x: split_on_punctuation_take_first(x))
    print("Shape after adding factcheck website column: ", df_druid.shape)
    print("Factcheck website column: ", df_druid['fc_source'].unique())
    
    return df_druid

def split_by_factcheck_website(df_druid):
    """
    Split the data by factcheck website
    """
    dfs_by_fcsource = {key: group for key, group in df_druid.groupby('fc_source')} 
    df_borderlines = dfs_by_fcsource['borderlines'].copy()
    print("Shape of borderlines: ", df_borderlines.shape)
    df_checkyourfact = dfs_by_fcsource['checkyourfact'].copy()
    print("Shape of checkyourfact: ", df_checkyourfact.shape)
    df_climatefeedback = dfs_by_fcsource['climatefeedback'].copy()
    print("Shape of climatefeedback: ", df_climatefeedback.shape)
    df_factcheckni = dfs_by_fcsource['factcheckni'].copy()
    print("Shape of factcheckni: ", df_factcheckni.shape)
    df_factly = dfs_by_fcsource['factly'].copy()
    print("Shape of factly: ", df_factly.shape)
    df_healthfeedback = dfs_by_fcsource['healthfeedback'].copy()
    print("Shape of healthfeedback: ", df_healthfeedback.shape)
    df_politifact = dfs_by_fcsource['politifact'].copy()
    print("Shape of politifact: ", df_politifact.shape)
    df_sciencefeedback = dfs_by_fcsource['sciencefeedback'].copy()
    print("Shape of sciencefeedback: ", df_sciencefeedback.shape)
    df_slfact = dfs_by_fcsource['srilankafactcrescendo'].copy()
    print("Shape of srilankafactcrescendo: ", df_slfact.shape)

    for key, df in dfs_by_fcsource.items():
        df.to_csv(os.path.join(BASE_DIR, f'{key}.csv'), index=False)

    # Concatenate factchecking datasets
    # Leave srilankafactcrescendo out as it only has 'False' factcheck verdicts
    df_factchecks = pd.concat([df_checkyourfact, df_factcheckni, df_factly], ignore_index=True)
    print("Shape of factchecks: ", df_factchecks.shape)
    df_factchecks.to_csv(os.path.join(BASE_DIR, 'all_factchecks.csv'), index=False)

def run():
    df_druid = get_druid_data()
    df_druid = preprocess_druid_data(df_druid)
    split_by_factcheck_website(df_druid)

if __name__ == "__main__":
    run()



