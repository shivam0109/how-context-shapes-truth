"""
Code to create a dataset simulating randomly shuffled choices in the dataset.
"""

import argparse
import os
import random
import json
import pandas as pd
import numpy as np
from ast import literal_eval


def save_to_jsonl(df, filepath):
    """Save DataFrame to JSONL format."""
    with open(filepath, 'w') as f:
        for _, row in df.iterrows():
            # Convert numpy types to Python types for JSON serialization
            record = {}
            for col, value in row.items():
                if col in ['first_token_activations_true', 'first_token_activations_false']:
                    # Convert string representations to lists for activation fields
                    record[col] = literal_eval(value)
                elif isinstance(value, np.ndarray):
                    record[col] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = value.item()
                else:
                    record[col] = value
            f.write(json.dumps(record) + '\n')


def main(dir_path, run_with_context, context_col):
    if run_with_context:
        df_ta_fb = pd.read_csv(os.path.join(dir_path, "df_ta_fb_with_context.csv"))
        df_fa_tb = pd.read_csv(os.path.join(dir_path, "df_fa_tb_with_context.csv"))
    else:
        df_ta_fb = pd.read_csv(os.path.join(dir_path, "df_ta_fb_without_context.csv"))
        df_fa_tb = pd.read_csv(os.path.join(dir_path, "df_fa_tb_without_context.csv"))
        df_ta_fb[context_col].replace(np.nan, 'None', inplace=True)
        df_fa_tb[context_col].replace(np.nan, 'None', inplace=True)
    
    # Get common <claim, evidence> pairs  
    pairs_ta_fb = {tuple(x) for x in df_ta_fb[['claim', context_col]].values}
    pairs_fa_tb = {tuple(x) for x in df_fa_tb[['claim', context_col]].values}
    common_pairs = pairs_ta_fb.intersection(pairs_fa_tb)
    print(f"Number of common <claim, evidence> pairs: {len(common_pairs)}")
    
    # Iterate through common_pairs and create a list of selected rows
    rows_to_append = []
    count_ta_fb = 0
    count_fa_tb = 0
    for pair in common_pairs:
        if random.random() < 0.5:
            rows_to_append.append(df_ta_fb[(df_ta_fb['claim'] == pair[0]) & (df_ta_fb[context_col] == pair[1])])
            count_ta_fb += 1
        else:
            rows_to_append.append(df_fa_tb[(df_fa_tb['claim'] == pair[0]) & (df_fa_tb[context_col] == pair[1])])
            count_fa_tb += 1
    
    # Concatenate all rows into a new dataframe
    if rows_to_append:
        new_df = pd.concat(rows_to_append, ignore_index=True)
    else:
        new_df = pd.DataFrame(columns=df_ta_fb.columns)
    

    # Save the new dataframe to JSONL 
    if run_with_context:
        output_path = os.path.join(dir_path, "df_shuffled_choices_with_context.jsonl")
        save_to_jsonl(new_df, output_path)
    else:
        output_path = os.path.join(dir_path, "df_shuffled_choices_without_context.jsonl")
        save_to_jsonl(new_df, output_path)

    print(f"Saved output to: {output_path}")
    print(f"Number of rows in the new dataframe: {len(new_df)}")
    print(f"Number of rows from df_ta_fb: {count_ta_fb}")
    print(f"Number of rows from df_fa_tb: {count_fa_tb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, required=True) # path to modeling datasets 
    parser.add_argument("--context_col", type=str, required=True) # column name for context
    parser.add_argument("--run_with_context", action="store_true") # whether to run with context or without context 
    args = parser.parse_args()
    main(args.dir_path, args.run_with_context, args.context_col)

