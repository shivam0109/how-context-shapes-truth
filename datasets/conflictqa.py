"""
Script to get data from ConflictQA dataset. Create 2 datasets: 
(1) With parametric evidence as context and (2) With counter evidence as context.  
Fields to use: index, claim, ground_truth, parametric_memory_aligned_evidence, counter_memory_aligned_evidence
"""

import json 
import os 
import pandas as pd
import argparse

USECOLS = ['index', 'claim', 'ground_truth', 'parametric_memory_aligned_evidence', 'counter_memory_aligned_evidence']

def load_jsonl(file_path, usecols=None):
    """
    Load data with usecols if provided.
    """
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if usecols:
        # Filter to only specified columns
        data = [{col: item[col] for col in usecols if col in item} for item in data]
    
    return data


def main(input_path, output_path):
    """
    Main function to create datasets.
    """
    data = load_jsonl(input_path, usecols=USECOLS)
    df = pd.DataFrame(data)
    df_parametric = df[['index', 'claim', 'ground_truth', 'parametric_memory_aligned_evidence']].copy()
    df_counter = df[['index', 'claim', 'ground_truth', 'counter_memory_aligned_evidence']].copy()
    df_parametric.rename(columns={'parametric_memory_aligned_evidence': 'evidence'}, inplace=True)
    df_counter.rename(columns={'counter_memory_aligned_evidence': 'evidence'}, inplace=True)
    df_parametric.to_csv(os.path.join(output_path, 'conflictqa_parametric.csv'), index=False)
    df_counter.to_csv(os.path.join(output_path, 'conflictqa_counter.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='conflictqa/conflictqa_rephrased_claims.jsonl', type=str)
    parser.add_argument('--output_path', default='conflictqa', type=str)
    args = parser.parse_args()
    main(args.input_path, args.output_path)