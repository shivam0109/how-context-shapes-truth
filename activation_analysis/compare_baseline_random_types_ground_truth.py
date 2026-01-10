"""
Script to compare theta and magnitude metrics for the last layer across baseline and random types.
Uses thetas_ground_truth_extended.json files (list of dictionaries) instead of averaged metrics.

This script compares baseline context activations with 5 different random context types:
- baseline: Original context
- random_char: Character-level random text
- random_word: Word-level random text (no semantic meaning)
- random_salad: Syntactically correct but semantically meaningless sentences
- random_wiki: Real sentences from Wikipedia
- random_shuffle: Random shuffle of the context

Generates CSV dataframes comparing theta and magnitude_rel_with_true_with_false for last layer:
1. Common datapoints between baseline and each random type
2. Common datapoints across all 6 random types
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# List of random types to compare
RANDOM_TYPES = ['baseline', 'random_char', 'random_word', 'random_salad', 'random_wiki', 'random_shuffle']

# Mapping from random types to display labels
TYPE_LABELS = {
    'baseline': 'Baseline',
    'random_char': 'Random Char',
    'random_word': 'Random Word',
    'random_salad': 'Random Salad',
    'random_wiki': 'Random Wiki',
    'random_shuffle': 'Random Shuffle'
}


def load_ground_truth_json(base_dir, dataset, random_type, prompt_type):
    """
    Load thetas_ground_truth_extended.json for a specific random type.
    Returns: list of dictionaries with keys: ['claim', 'ground_truth', 'layer_dict', 'evidence', 'index', 'ground_truth_true', 'compound_id']
    """
    # Construct the path to the metrics file
    if random_type == 'baseline':
        middle_path = f"modeling/{prompt_type}/all_data/shuffled/baseline"
    else:
        middle_path = f"modeling/{prompt_type}/all_data/shuffled/{random_type}"
    
    json_file = os.path.join(
        base_dir,
        dataset,
        middle_path,
        'model_outputs_truthfulness',
        'vector_diff',
        'thetas_ground_truth_extended.json'
    )
    
    if not os.path.exists(json_file):
        print(f"WARNING: JSON file not found for {random_type}: {json_file}")
        return None
    
    print(f"Loading JSON from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"ERROR: Expected list, got {type(data)}")
        return None
    
    print(f"Loaded {len(data)} records for {random_type}")
    
    # Check structure of first record
    if len(data) > 0:
        sample_keys = list(data[0].keys())
        print(f"Sample record keys: {sample_keys}")
    
    return data


def get_merge_key(dataset, data_list):
    """
    Determine the merge key for a dataset.
    For corporate_lobbying: use 'compound_id'
    Else: use 'index' if present, otherwise 'claim'
    
    Args:
        dataset: Dataset name
        data_list: List of dictionaries to check for available keys
        
    Returns:
        str: The merge key to use
    """
    if 'corporate_lobbying' in dataset:
        return 'compound_id'
    
    # Check if 'index' field exists in the data
    # Check multiple records to be sure
    if len(data_list) > 0:
        # Check first few records to see if 'index' is consistently present
        sample_size = min(10, len(data_list))
        has_index = all('index' in data_list[i] and data_list[i].get('index') is not None 
                       for i in range(sample_size))
        if has_index:
            return 'index'
    
    return 'claim'


def extract_last_layer_metrics(data, merge_key):
    """
    Extract theta and magnitude_rel_with_true_with_false from the last layer.
    
    Args:
        data: List of dictionaries from thetas_ground_truth_extended.json
        merge_key: Key to use for merging (e.g., 'compound_id', 'index', 'claim')
        
    Returns:
        pd.DataFrame: DataFrame with merge_key, theta, and magnitude_rel_with_true_with_false
    """
    if data is None or len(data) == 0:
        return pd.DataFrame()
    
    records = []
    
    for record in data:
        if 'layer_dict' not in record:
            continue
        
        layer_dict = record['layer_dict']
        if not isinstance(layer_dict, dict):
            continue
        
        # Find the last layer
        layer_keys = [k for k in layer_dict.keys() if k.startswith('layer_')]
        if not layer_keys:
            continue
        
        # Extract layer numbers and find the maximum
        layer_nums = []
        for k in layer_keys:
            try:
                layer_num = int(k.split('_')[1])
                layer_nums.append(layer_num)
            except (ValueError, IndexError):
                continue
        
        if not layer_nums:
            continue
        
        last_layer_num = max(layer_nums)
        last_layer_key = f'layer_{last_layer_num}'
        
        if last_layer_key not in layer_dict:
            continue
        
        last_layer_data = layer_dict[last_layer_key]
        if not isinstance(last_layer_data, dict):
            continue
        
        # Extract theta and magnitude
        theta = last_layer_data.get('theta', None)
        magnitude = last_layer_data.get('magnitude_rel_with_true_with_false', None)
        
        # Get merge key value
        merge_value = record.get(merge_key, None)
        
        if merge_value is not None:
            records.append({
                merge_key: merge_value,
                'theta': theta,
                'magnitude_rel_with_true_with_false': magnitude
            })
    
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} records with last layer metrics")
    
    # Check for duplicates in merge key and average if needed
    if len(df) > 0:
        duplicates = df[merge_key].duplicated().sum()
        unique_count = df[merge_key].nunique()
        if duplicates > 0:
            print(f"WARNING: Found {duplicates} duplicate values in merge key '{merge_key}'")
            print(f"  Total records: {len(df)}, Unique {merge_key} values: {unique_count}")
            print(f"  This will cause many-to-many matches during merge!")
            # Deduplicate by averaging theta and magnitude for duplicate merge_key values
            print(f"  Averaging theta and magnitude for duplicate {merge_key} values...")
            df = df.groupby(merge_key).agg({
                'theta': 'mean',
                'magnitude_rel_with_true_with_false': 'mean'
            }).reset_index()
            print(f"  After averaging: {len(df)} records (one per unique {merge_key})")
    
    return df


def perform_statistical_tests(baseline_vals, other_vals, metric_name='metric'):
    """
    Perform Wilcoxon signed-rank test (non-parametric) to test if baseline > other.
    
    Args:
        baseline_vals: Array of baseline values
        other_vals: Array of other values (paired with baseline)
        metric_name: Name of the metric (for error messages)
        
    Returns:
        tuple: (statistic, p_value) or (np.nan, np.nan) if test fails
    """
    try:
        statistic, p_value = stats.wilcoxon(
            baseline_vals,
            other_vals,
            alternative='greater'  # H1: baseline > other
        )
        return statistic, p_value
    except Exception as e:
        print(f"Warning: Could not perform Wilcoxon test for {metric_name}: {e}")
        return np.nan, np.nan


def find_common_datapoints_baseline_vs_other(baseline_df, other_df, merge_key, random_type):
    """
    Find common datapoints between baseline and another random type.
    
    Args:
        baseline_df: DataFrame with baseline data
        other_df: DataFrame with other random type data
        merge_key: Key to use for merging
        random_type: The random type name (e.g., 'random_char', 'random_word')
        
    Returns:
        pd.DataFrame: DataFrame with common datapoints
    """
    if baseline_df.empty or other_df.empty:
        return pd.DataFrame()
    
    # Extract suffix from random_type (e.g., 'random_char' -> 'char', 'random_word' -> 'word')
    if random_type.startswith('random_'):
        suffix = random_type.replace('random_', '')
    else:
        suffix = random_type
    
    # Merge on the merge key
    merged = pd.merge(
        baseline_df,
        other_df,
        on=merge_key,
        how='inner',
        suffixes=('_baseline', f'_{suffix}')
    )
    
    return merged


def find_common_datapoints_all_types(dataframes_dict, merge_key):
    """
    Find common datapoints across all random types.
    
    Args:
        dataframes_dict: Dictionary mapping random_type -> DataFrame (already with renamed columns)
        merge_key: Key to use for merging
        
    Returns:
        pd.DataFrame: DataFrame with common datapoints across all types
    """
    if not dataframes_dict:
        return pd.DataFrame()
    
    # Start with the first dataframe
    first_type = list(dataframes_dict.keys())[0]
    common_df = dataframes_dict[first_type].copy()
    
    # Iteratively merge with all other dataframes
    for random_type, df in list(dataframes_dict.items())[1:]:
        if df.empty:
            return pd.DataFrame()
        
        # Merge on the merge key
        common_df = pd.merge(
            common_df,
            df,
            on=merge_key,
            how='inner'
        )
    
    return common_df


def get_merge_key_for_comparison(dataset, data1, data2):
    """
    Determine the merge key for comparing two datasets.
    Checks both datasets to ensure the key exists in both.
    
    Args:
        dataset: Dataset name
        data1: First list of dictionaries
        data2: Second list of dictionaries
        
    Returns:
        str: The merge key to use
    """
    if 'corporate_lobbying' in dataset:
        return 'compound_id'
    
    # Check if 'index' field exists in both datasets
    if len(data1) > 0 and len(data2) > 0:
        sample_size1 = min(10, len(data1))
        sample_size2 = min(10, len(data2))
        has_index1 = all('index' in data1[i] and data1[i].get('index') is not None 
                        for i in range(sample_size1))
        has_index2 = all('index' in data2[i] and data2[i].get('index') is not None 
                        for i in range(sample_size2))
        if has_index1 and has_index2:
            return 'index'
    
    return 'claim'


def create_comparison_dataframe_baseline_vs_other(base_dir, dataset, prompt_type, output_dir, dataset_slug):
    """
    Create dataframes comparing baseline vs each random type (common datapoints only).
    """
    print("\n" + "="*80)
    print("CREATING BASELINE VS OTHER RANDOM TYPES COMPARISON")
    print("="*80)
    
    # Load baseline data
    baseline_data = load_ground_truth_json(base_dir, dataset, 'baseline', prompt_type)
    baseline_df = None
    merge_key = None
    
    if baseline_data is not None:
        # Try to determine merge key (will be refined per comparison)
        merge_key = get_merge_key(dataset, baseline_data)
        baseline_df = extract_last_layer_metrics(baseline_data, merge_key)
    
    results = []
    
    # Compare baseline with each other random type
    for random_type in RANDOM_TYPES:
        if random_type == 'baseline':
            continue
        
        print(f"\nProcessing {random_type}...")
        other_data = load_ground_truth_json(base_dir, dataset, random_type, prompt_type)
        
        # If baseline or other data is missing, add -1 values
        if baseline_data is None or other_data is None:
            print(f"Data unavailable for {random_type} or baseline - adding -1 values")
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Baseline_Avg': -1,
                'Theta_Other_Avg': -1,
                'Theta_Wilcoxon_Statistic': -1,
                'Theta_Wilcoxon_PValue': -1,
                'Magnitude_Baseline_Avg': -1,
                'Magnitude_Other_Avg': -1,
                'Magnitude_Wilcoxon_Statistic': -1,
                'Magnitude_Wilcoxon_PValue': -1
            })
            continue
        
        # Determine merge key for this comparison (check both baseline and other)
        merge_key = get_merge_key_for_comparison(dataset, baseline_data, other_data)
        print(f"Using merge key: {merge_key}")
        
        # Extract metrics for both baseline and other
        baseline_df = extract_last_layer_metrics(baseline_data, merge_key)
        if baseline_df.empty:
            print(f"Warning: No baseline metrics extracted for comparison with {random_type} - adding -1 values")
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Baseline_Avg': -1,
                'Theta_Other_Avg': -1,
                'Theta_Wilcoxon_Statistic': -1,
                'Theta_Wilcoxon_PValue': -1,
                'Magnitude_Baseline_Avg': -1,
                'Magnitude_Other_Avg': -1,
                'Magnitude_Wilcoxon_Statistic': -1,
                'Magnitude_Wilcoxon_PValue': -1
            })
            continue
            
        other_df = extract_last_layer_metrics(other_data, merge_key)
        if other_df.empty:
            print(f"No metrics extracted for {random_type} - adding -1 values")
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Baseline_Avg': -1,
                'Theta_Other_Avg': -1,
                'Theta_Wilcoxon_Statistic': -1,
                'Theta_Wilcoxon_PValue': -1,
                'Magnitude_Baseline_Avg': -1,
                'Magnitude_Other_Avg': -1,
                'Magnitude_Wilcoxon_Statistic': -1,
                'Magnitude_Wilcoxon_PValue': -1
            })
            continue
        
        # Find common datapoints
        common_df = find_common_datapoints_baseline_vs_other(baseline_df, other_df, merge_key, random_type)
        
        if common_df.empty:
            print(f"No common datapoints between baseline and {random_type} - adding -1 values")
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Baseline_Avg': -1,
                'Theta_Other_Avg': -1,
                'Theta_Wilcoxon_Statistic': -1,
                'Theta_Wilcoxon_PValue': -1,
                'Magnitude_Baseline_Avg': -1,
                'Magnitude_Other_Avg': -1,
                'Magnitude_Wilcoxon_Statistic': -1,
                'Magnitude_Wilcoxon_PValue': -1
            })
            continue
        
        print(f"Found {len(common_df)} common datapoints")
        
        # Extract suffix from random_type for column names
        if random_type.startswith('random_'):
            suffix = random_type.replace('random_', '')
        else:
            suffix = random_type
        
        # Calculate averages for common datapoints
        theta_baseline_avg = common_df['theta_baseline'].mean()
        theta_other_avg = common_df[f'theta_{suffix}'].mean()
        magnitude_baseline_avg = common_df['magnitude_rel_with_true_with_false_baseline'].mean()
        magnitude_other_avg = common_df[f'magnitude_rel_with_true_with_false_{suffix}'].mean()
        
        # Perform statistical tests (one-sided: baseline > random)
        theta_baseline_vals = common_df['theta_baseline'].values
        theta_other_vals = common_df[f'theta_{suffix}'].values
        theta_stat, theta_pvalue = perform_statistical_tests(
            theta_baseline_vals, 
            theta_other_vals, 
            metric_name='theta'
        )
        
        magnitude_baseline_vals = common_df['magnitude_rel_with_true_with_false_baseline'].values
        magnitude_other_vals = common_df[f'magnitude_rel_with_true_with_false_{suffix}'].values
        magnitude_stat, magnitude_pvalue = perform_statistical_tests(
            magnitude_baseline_vals,
            magnitude_other_vals,
            metric_name='magnitude'
        )
        
        results.append({
            'Type': TYPE_LABELS.get(random_type, random_type),
            'Num_Common_Datapoints': len(common_df),
            'Theta_Baseline_Avg': theta_baseline_avg,
            'Theta_Other_Avg': theta_other_avg,
            'Theta_Wilcoxon_Statistic': theta_stat,
            'Theta_Wilcoxon_PValue': theta_pvalue,
            'Magnitude_Baseline_Avg': magnitude_baseline_avg,
            'Magnitude_Other_Avg': magnitude_other_avg,
            'Magnitude_Wilcoxon_Statistic': magnitude_stat,
            'Magnitude_Wilcoxon_PValue': magnitude_pvalue
        })
    
    # Always create dataframe, even if all values are -1
    if not results:
        print("Warning: No results collected - creating dataframe with -1 values for all types")
        # Create dataframe with all random types and -1 values
        for random_type in RANDOM_TYPES:
            if random_type != 'baseline':
                results.append({
                    'Type': TYPE_LABELS.get(random_type, random_type),
                    'Num_Common_Datapoints': -1,
                    'Theta_Baseline_Avg': -1,
                    'Theta_Other_Avg': -1,
                    'Theta_Wilcoxon_Statistic': -1,
                    'Theta_Wilcoxon_PValue': -1,
                    'Magnitude_Baseline_Avg': -1,
                    'Magnitude_Other_Avg': -1,
                    'Magnitude_Wilcoxon_Statistic': -1,
                    'Magnitude_Wilcoxon_PValue': -1
                })
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'df_baseline_vs_other_{dataset_slug}_{prompt_type}.csv')
    df.to_csv(out_file, index=False)
    print(f"\nSaved baseline vs other comparison to: {out_file}")
    print("\nDataFrame:")
    print(df.to_string(index=False))
    
    return df


def create_comparison_dataframe_all_types(base_dir, dataset, prompt_type, output_dir, dataset_slug):
    """
    Create dataframe comparing all 6 random types (common datapoints only).
    """
    print("\n" + "="*80)
    print("CREATING ALL TYPES COMPARISON (COMMON DATAPOINTS)")
    print("="*80)
    
    # Load all data
    all_data = {}
    for random_type in RANDOM_TYPES:
        print(f"\nLoading {random_type}...")
        data = load_ground_truth_json(base_dir, dataset, random_type, prompt_type)
        if data is None:
            print(f"Data unavailable for {random_type} - will use -1 values")
        else:
            all_data[random_type] = data
    
    if not all_data:
        print("ERROR: No data loaded - creating dataframe with -1 values for all types")
        results = []
        for random_type in RANDOM_TYPES:
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Avg': -1,
                'Magnitude_Avg': -1
            })
        df = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'df_all_types_common_{dataset_slug}_{prompt_type}.csv')
        df.to_csv(out_file, index=False)
        print(f"\nSaved all types comparison to: {out_file}")
        return df
    
    # Determine merge key - check all loaded data to find common key
    # For corporate_lobbying: use 'compound_id'
    # Else: check if 'index' exists in all types, otherwise use 'claim'
    if 'corporate_lobbying' in dataset:
        merge_key = 'compound_id'
    else:
        # Check if 'index' exists in all loaded data types
        all_have_index = True
        for random_type, data in all_data.items():
            if len(data) > 0:
                sample_size = min(10, len(data))
                has_index = all('index' in data[i] and data[i].get('index') is not None 
                               for i in range(sample_size))
                if not has_index:
                    all_have_index = False
                    break
            else:
                all_have_index = False
                break
        
        if all_have_index:
            merge_key = 'index'
        else:
            merge_key = 'claim'
    
    print(f"Using merge key: {merge_key}")
    
    # Extract metrics for all types
    all_dfs = {}
    for random_type, data in all_data.items():
        df = extract_last_layer_metrics(data, merge_key)
        if not df.empty:
            # Rename columns to include type suffix
            df = df.rename(columns={
                'theta': f'theta_{random_type}',
                'magnitude_rel_with_true_with_false': f'magnitude_{random_type}'
            })
            all_dfs[random_type] = df
        else:
            print(f"Warning: No metrics extracted for {random_type}")
    
    if len(all_dfs) < 2:
        print("ERROR: Need at least 2 types with data - creating dataframe with -1 values")
        results = []
        for random_type in RANDOM_TYPES:
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Avg': -1,
                'Magnitude_Avg': -1
            })
        df = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'df_all_types_common_{dataset_slug}_{prompt_type}.csv')
        df.to_csv(out_file, index=False)
        print(f"\nSaved all types comparison to: {out_file}")
        return df
    
    # Find common datapoints across all types
    print(f"\nFinding common datapoints across {len(all_dfs)} types...")
    common_df = find_common_datapoints_all_types(all_dfs, merge_key)
    
    if common_df.empty:
        print("ERROR: No common datapoints found across all types - creating dataframe with -1 values")
        results = []
        for random_type in RANDOM_TYPES:
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Avg': -1,
                'Magnitude_Avg': -1
            })
        df = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'df_all_types_common_{dataset_slug}_{prompt_type}.csv')
        df.to_csv(out_file, index=False)
        print(f"\nSaved all types comparison to: {out_file}")
        return df
    
    print(f"Found {len(common_df)} common datapoints across all types")
    
    # Calculate averages for each type
    results = []
    for random_type in RANDOM_TYPES:
        theta_col = f'theta_{random_type}'
        magnitude_col = f'magnitude_{random_type}'
        
        if theta_col in common_df.columns and magnitude_col in common_df.columns:
            theta_avg = common_df[theta_col].mean()
            magnitude_avg = common_df[magnitude_col].mean()
            
            # For all types comparison, we can compare each type to baseline
            # Store the values for potential statistical tests
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': len(common_df),
                'Theta_Avg': theta_avg,
                'Magnitude_Avg': magnitude_avg
            })
        else:
            # Type was not in the common datapoints (missing data)
            print(f"Warning: {random_type} not in common datapoints - adding -1 values")
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Avg': -1,
                'Magnitude_Avg': -1
            })
    
    if not results:
        print("Warning: No results collected - creating dataframe with -1 values for all types")
        for random_type in RANDOM_TYPES:
            results.append({
                'Type': TYPE_LABELS.get(random_type, random_type),
                'Num_Common_Datapoints': -1,
                'Theta_Avg': -1,
                'Magnitude_Avg': -1
            })
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'df_all_types_common_{dataset_slug}_{prompt_type}.csv')
    df.to_csv(out_file, index=False)
    print(f"\nSaved all types comparison to: {out_file}")
    print("\nDataFrame:")
    print(df.to_string(index=False))
    
    return df


def main(base_dir, dataset, prompt_type, output_dir):
    """
    Main function to generate comparisons.
    
    Args:
        base_dir: Base directory containing dataset subdirectories
        dataset: Dataset name (e.g., 'mf2', 'druid/borderlines')
        prompt_type: Prompt type to use (implicit/explicit)
        output_dir: Directory to save outputs
    """
    print("="*80)
    print("COMPARING BASELINE AND RANDOM TYPES (GROUND TRUTH)")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Dataset: {dataset}")
    print(f"Prompt type: {prompt_type}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    # Create dataset slug for unique filenames
    dataset_slug = dataset.replace('/', '_')
    
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(output_dir, dataset_slug)
    
    # Create baseline vs other comparison
    print("\n" + "="*80)
    print("SETTING 1: BASELINE VS OTHER RANDOM TYPES")
    print("="*80)
    create_comparison_dataframe_baseline_vs_other(base_dir, dataset, prompt_type, dataset_output_dir, dataset_slug)
    
    # Create all types comparison
    print("\n" + "="*80)
    print("SETTING 2: ALL TYPES (COMMON DATAPOINTS)")
    print("="*80)
    create_comparison_dataframe_all_types(base_dir, dataset, prompt_type, dataset_output_dir, dataset_slug)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare theta and magnitude metrics for baseline and random types using ground truth JSON"
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="Base directory containing dataset subdirectories"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'mf2', 'druid/borderlines')"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="implicit",
        help="Prompt type to use (implicit or explicit)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save outputs (default: 'comparisons_ground_truth' subdirectory in base_dir)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, 'comparisons_ground_truth')
    
    main(args.base_dir, args.dataset, args.prompt_type, args.output_dir)

