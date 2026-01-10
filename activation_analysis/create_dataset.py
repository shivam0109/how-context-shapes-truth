"""
Code to create dataset for modeling the activations.
1. Select only instances where instruction has been followed.
2. Pair (TA, FB) and (FA, TB) with and without context.
3. Select only instances where the instruction has been followed with and without context.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import glob
import re
import sys
from ast import literal_eval
import gc

def load_data(jsonl_file):
    """
    Load data from a jsonl file, returning only rows where instruction_followed is True.
    Reads line-by-line to reduce memory and skips malformed lines.
    Input:
        jsonl_file: path to the jsonl file
    Output:
        data: list of dictionaries (filtered)
    """
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue
            
            # Filtering by instruction_followed
            instruction_followed = record.get('instruction_followed', False)
            if instruction_followed in [True, "true", "yes", "1"]:
                data.append(record)
    return data


def extract_claim_context(prompt, context_col):
    claim_match = re.search(r"\[claim\]: <(.*?)>", prompt, re.DOTALL)
    context_match = re.search(rf"\[{context_col}\]: <(.*?)>", prompt, re.DOTALL)
    claim = claim_match.group(1).strip() if claim_match else None
    context = context_match.group(1).strip() if context_match else None
    return claim, context


# def convert_activations_to_list(df):
#     for col in df.columns:
#         if 'activations' in col:
#             def try_convert(x):
#                 if isinstance(x, str):
#                     try:
#                         return literal_eval(x)
#                     except Exception as e:
#                         print(f"Could not convert value in column '{col}': {x}")
#                         return x
#                 else:
#                     return x
#             df[col] = df[col].apply(try_convert)
#     return df

def _process_jsonl(jsonl_file_path, context_col, merge_col=None, tag="true", context=False):
    """
    Basic preprocessing of the jsonl file.
    Input: jsonl_file_path, tag="true" or "false"
    Output: df
    """
    # Load data - already filtered by instruction_followed in load_data
    data = load_data(jsonl_file_path)
    print(f"Number of rows in {jsonl_file_path}: {len(data)}")
    print(f"Tag: {tag}, Context: {context}")
    
    # Convert to DF 
    df = pd.DataFrame(data)

    # Remove whitespace from the prompt and completion 
    df['prompt'] = df['prompt'].str.strip()
    df['completion'] = df['completion'].str.strip()
    
    # Rename the columns - Add _true or _false to the column names 
    if merge_col is not None:
        # Exclude merge_col from renaming
        df.columns = [f'{col}_{tag}' if col != merge_col else col for col in list(df.columns)]
    else:
        df.columns = [f'{col}_{tag}' for col in list(df.columns)]

    # Get claim and context columns from prompt
    df[['claim', context_col]] = df[f'prompt_{tag}'].apply(lambda x: extract_claim_context(x, context_col)).apply(pd.Series)

    return df

def _merge_dataframes_in_chunks(df1: pd.DataFrame,
    df2: pd.DataFrame,
    chunk_size: int,
    on_columns,  # Can be string or list
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merges two pandas DataFrames in chunks to handle large datasets efficiently.

    Args:
        df1 (pd.DataFrame): The first DataFrame (will be processed in chunks).
        df2 (pd.DataFrame): The second DataFrame (will be loaded entirely into memory,
                            or could be adapted for chunking if also very large).
        chunk_size (int): The number of rows to process from df1 in each chunk.
        on_columns: The common column name(s) to merge on. Can be string or list.
        how (str): Type of merge to be performed. Options include 'left', 'right',
                   'outer', 'inner' (default).

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    merged_chunks = []
    num_chunks = int(np.ceil(len(df1) / chunk_size))

    print(f"Starting chunked merge. Total rows in df1: {len(df1)}, Chunk size: {chunk_size}, Num chunks: {num_chunks}")

    for i in range(num_chunks):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, len(df1))
        
        # Select the chunk from df1
        df1_chunk = df1.iloc[start_row:end_row]
        
        print(f"Processing chunk {i+1}/{num_chunks} (rows {start_row}-{end_row-1})")

        # Perform the merge operation for the current chunk
        merged_chunk = pd.merge(df1_chunk, df2, on=on_columns, how=how)
        merged_chunks.append(merged_chunk)

    # Concatenate all merged chunks into a single DataFrame
    final_merged_df = pd.concat(merged_chunks, ignore_index=True)
    print("Chunked merge completed successfully.")
    return final_merged_df   


def _filter_and_merge_data_helper(input_json_files, context_col, context=False, merge_col=None):
    true_path = [x for x in input_json_files if x.split('/')[-1].startswith('ta') or x.split('/')[-1].startswith('tb')]
    false_path = [x for x in input_json_files if x.split('/')[-1].startswith('fa') or x.split('/')[-1].startswith('fb')]
    assert len(true_path) == 1 and len(false_path) == 1
    # Process the jsonl files 
    df_true_filtered = _process_jsonl(true_path[0], context_col=context_col, merge_col=merge_col, tag="true", context=context)
    df_false_filtered = _process_jsonl(false_path[0], context_col=context_col, merge_col=merge_col, tag="false", context=context)
    # Merge the dataframes
    if merge_col is not None:
        on_columns = ['claim', context_col, merge_col]
    else:
        on_columns = ['claim', context_col]
    df = _merge_dataframes_in_chunks(df_true_filtered, df_false_filtered, chunk_size=100, on_columns=on_columns, how='inner')
    # Print the number of rows in each dataframe 
    print(f"Shape of df: {df.shape}")
    return df


def filter_and_merge_data(input_json_files_without_context, input_json_files_with_context, context_col, merge_col=None):
    """
    Input: json_files_without_context, json_files_with_context 
    Output: df_without_context, df_with_context

    Filter and merge the data from the jsonl files. 
    Consider only instances where the instruction has been followed both with and without context.
    """
    df_with_context = _filter_and_merge_data_helper(input_json_files_with_context, context_col=context_col, context=True, merge_col=merge_col)
    if len(input_json_files_without_context) > 0:
        df_without_context = _filter_and_merge_data_helper(input_json_files_without_context, context_col=context_col, context=False, merge_col=merge_col)
        # Filter out rows from df_with_context where claim is not present in df_without_context
        df_with_context = df_with_context[df_with_context['claim'].isin(df_without_context['claim'])]
    else:
        df_without_context = pd.DataFrame()

    
    # Print the number of rows in each dataframe 
    print(f"Shape of df_without_context: {df_without_context.shape}")
    print(f"Shape of df_with_context after filtering claims not present in df_without_context: {df_with_context.shape}")

    return df_without_context, df_with_context


def main(input_dir, output_dir, context_col, merge_col=None):
    """
    Main function to create dataset for modeling the activations.
    Input:
        input_dir: path to activations - implicit or explicit folder 
    Output:
        None 
    """
    
    # Get all the jsonl files in the directory 
    jsonl_files = glob.glob(os.path.join(input_dir, '**', '*.jsonl'), recursive=True)

    # Separate jsonl files with and without context 
    json_files_with_context = [file for file in jsonl_files if 'with_context' in file]
    json_files_without_context = [file for file in jsonl_files if 'wo_context' in file]

    print("JSON files with context: \n", json_files_with_context)
    print("JSON files without context: \n", json_files_without_context)
    
    # Create the output directory 
    os.makedirs(output_dir, exist_ok=True)

    # Pair (TA, FB) and (FA, TB)
    ta_fb_with_context = [file for file in json_files_with_context if file.split('/')[-1].startswith('ta') or file.split('/')[-1].startswith('fb')]
    fa_tb_with_context = [file for file in json_files_with_context if file.split('/')[-1].startswith('fa') or file.split('/')[-1].startswith('tb')]
    if len(json_files_without_context) > 0:
        ta_fb_without_context = [file for file in json_files_without_context if file.split('/')[-1].startswith('ta') or file.split('/')[-1].startswith('fb')]
        fa_tb_without_context = [file for file in json_files_without_context if file.split('/')[-1].startswith('fa') or file.split('/')[-1].startswith('tb')]
    else:
        ta_fb_without_context = []
        fa_tb_without_context = []
    
    # Select only instances where the instruction has been followed 
    print("Creating dataset for TA-FB")
    df_ta_fb_without_context, df_ta_fb_with_context = filter_and_merge_data(ta_fb_without_context, ta_fb_with_context, context_col=context_col, merge_col=merge_col)
    
    # Save the dataframes to csv 
    if df_ta_fb_without_context.shape[0] > 0: 
        df_ta_fb_without_context.to_csv(os.path.join(output_dir, 'df_ta_fb_without_context.csv'), index=False)
    df_ta_fb_with_context.to_csv(os.path.join(output_dir, 'df_ta_fb_with_context.csv'), index=False)
    # Free memory 
    del df_ta_fb_without_context, df_ta_fb_with_context
    gc.collect()
    
    print("Creating dataset for FA-TB")
    
    df_fa_tb_without_context, df_fa_tb_with_context = filter_and_merge_data(fa_tb_without_context, fa_tb_with_context, context_col=context_col, merge_col=merge_col)
    # Save the dataframes to csv 
    if df_fa_tb_without_context.shape[0] > 0:
        df_fa_tb_without_context.to_csv(os.path.join(output_dir, 'df_fa_tb_without_context.csv'), index=False)
    df_fa_tb_with_context.to_csv(os.path.join(output_dir, 'df_fa_tb_with_context.csv'), index=False)
    # Free memory 
    del df_fa_tb_without_context, df_fa_tb_with_context
    gc.collect()
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True) # path to the preprocessed activations - implicit or explicit folder 
    parser.add_argument("--output_dir", type=str, required=True) # path to the output directory 
    parser.add_argument("--context_col", type=str, required=True) # context column name
    parser.add_argument("--merge_col", type=str) # column to merge on
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.context_col, args.merge_col)

