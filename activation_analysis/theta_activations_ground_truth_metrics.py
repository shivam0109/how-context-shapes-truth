"""
Code to get theta using vector difference from JSONL files with ground truth annotations and extended metrics
1. Get the vector difference between two activations - True vs False - without context using ground truth
2. Get the vector difference between two activations - True vs False - with context using ground truth
3. Get theta across all layers by comparing 1 and 2: (at an individual level and across the dataset)
4. Calculate extended metrics including magnitude and theta relationships
"""

import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

# Import utility functions
from ground_truth_utils import (
    get_ground_truth_vector_difference,
    calculate_extended_metrics,
    calculate_theta_metrics,
    calculate_basic_metrics,
    normalize_claims,
    create_record_with_metadata,
    extract_ground_truth,
    extract_bill_identifier,
    EPSILON,
    ZERO_VECTOR_THETA,
    COSINE_MIN,
    COSINE_MAX
)

# Configuration constants
DEFAULT_CHUNK_SIZE = 100
DEFAULT_OUTPUT_DIR = 'model_outputs_truthfulness'
DEFAULT_VECTOR_DIFF_DIR = 'vector_diff'
DEFAULT_FILENAME_WO_CONTEXT = "df_shuffled_choices_without_context.jsonl"
DEFAULT_FILENAME_WITH_CONTEXT = "df_shuffled_choices_with_context.jsonl"

USECOLS_DRUID = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'evidence', 
        'factcheck_verdict_true', 'factcheck_verdict_false', 'evidence_stance_true', 'evidence_stance_false']

USECOLS_MF2 = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'synopsis', 
            'movie_id_true', 'granularity_true', 'category_true', 'claim_type_true']

USECOLS_CL_BILL = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'bill summary', 
            'answer_true', 'index_true', 'prompt_true']

USECOLS_CL_COMPANY = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'company description', 
            'answer_true', 'index_true', 'prompt_true']

USECOLS_CONFLICTQA = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'evidence', 'index', 'ground_truth_true']


def load_jsonl_to_dataframe(filepath, usecols=None, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Load JSONL file into a pandas DataFrame with memory-efficient chunking.
    
    Args:
        filepath: Path to the JSONL file
        usecols: List of columns to keep (optional)
        chunk_size: Number of records to process at a time (default: 100)
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    data_chunks = []
    total_records = 0
    chunk_count = 0
    
    with open(filepath, 'r') as f:
        chunk_data = []
        
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                
                # Filter columns early if specified
                if usecols:
                    record = {k: v for k, v in record.items() if k in usecols}
                
                chunk_data.append(record)
                total_records += 1
                
                # Process chunk when it reaches chunk_size
                if len(chunk_data) >= chunk_size:
                    chunk_df = pd.DataFrame(chunk_data)
                    data_chunks.append(chunk_df)
                    chunk_count += 1
                    print(f"Processed chunk {chunk_count} ({len(chunk_data)} records)")
                    chunk_data = []  # Reset chunk
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
        
        # Process remaining records in final chunk
        if chunk_data:
            chunk_df = pd.DataFrame(chunk_data)
            data_chunks.append(chunk_df)
            chunk_count += 1
    
    # Combine all chunks
    df = pd.concat(data_chunks, ignore_index=True)
    
    # Verify column filtering
    if usecols:
        available_cols = [col for col in usecols if col in df.columns]
        missing_cols = [col for col in usecols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
        df = df[available_cols]
    
    print(f"Successfully loaded {len(df)} records from JSONL file")
    print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")
    
    return df


def get_cos_and_theta(vec_diff_wo_context, vec_diff_with_context):
    """
    Calculate cosine similarity and theta angle between two vectors.
    
    Args:
        vec_diff_wo_context: Vector difference without context
        vec_diff_with_context: Vector difference with context
    
    Returns:
        tuple: (cosine_similarity, theta_angle_in_degrees)
    """
    norm_wo = np.linalg.norm(vec_diff_wo_context)
    norm_with = np.linalg.norm(vec_diff_with_context)
    
    # Check for zero vectors to avoid division by zero
    if norm_wo == 0 or norm_with == 0:
        return 0.0, ZERO_VECTOR_THETA  # Return configured theta for zero vectors
    
    cos = np.dot(vec_diff_wo_context, vec_diff_with_context) / (norm_wo * norm_with)
    
    # Clamp cosine to avoid numerical errors
    cos = np.clip(cos, COSINE_MIN, COSINE_MAX)
    theta_radians = np.arccos(cos)
    
    # Convert radians to degrees
    theta_degrees = np.degrees(theta_radians)
    
    return cos, theta_degrees


def create_compound_identifier(row, dataset):
    """
    Create compound identifier for datasets that require it.
    
    Args:
        row: DataFrame row
        dataset: Dataset name
    
    Returns:
        str: Compound identifier (claim + bill info for CL datasets, claim for others)
    """
    claim = row['claim'].lower().strip()
    
    if dataset in ['cl_bill', 'cl_company']:
        # Extract bill identifier from prompt_true
        bill_identifier = extract_bill_identifier(row['prompt_true'], dataset)
        return f"{claim}|||{bill_identifier.lower().strip()}"
    else:
        return claim


def process_claim_group(df_wo_context_claim, df_with_context_claim, claim, dataset, thetas, NUM_LAYERS):
    """
    Process a group of claims (either single claim or compound_id group).
    
    Args:
        df_wo_context_claim: DataFrame subset for without context
        df_with_context_claim: DataFrame subset for with context
        claim: Original claim text
        dataset: Dataset name
        thetas: List to append results to
        NUM_LAYERS: Number of layers
    """
    print(f"Found {len(df_with_context_claim)} matching claims")

    if df_wo_context_claim.empty or df_with_context_claim.empty:
        print(f"Missing data for claim: {claim}")
        return
    
    # Get ground truth for this claim (should be same for all rows with same claim)
    ground_truth = extract_ground_truth(df_wo_context_claim.iloc[0], dataset)
    print(f"Ground truth for claim '{claim}': {ground_truth}")
    
    for _, row in df_with_context_claim.iterrows():
        layer_dict = {}
        for layer in range(NUM_LAYERS):
            # Extract the specific layer from both datasets
            true_wo = np.array(df_wo_context_claim['first_token_activations_true'].iloc[0][layer])
            false_wo = np.array(df_wo_context_claim['first_token_activations_false'].iloc[0][layer])
            true_with = np.array(row['first_token_activations_true'][layer])
            false_with = np.array(row['first_token_activations_false'][layer])
            
            # Calculate basic metrics (cos, theta, magnitude)
            basic_metrics = calculate_basic_metrics(
                true_with, false_with, true_wo, false_wo, ground_truth, get_cos_and_theta
            )
            
            # Calculate extended magnitude metrics
            extended_magnitude_metrics = calculate_extended_metrics(
                true_with, false_with, true_wo, false_wo, ground_truth
            )
            
            # Calculate extended theta metrics
            extended_theta_metrics = calculate_theta_metrics(
                true_with, false_with, true_wo, false_wo, ground_truth, get_cos_and_theta
            )
            
            # Combine all metrics
            layer_metrics = {
                **basic_metrics,
                **extended_magnitude_metrics,
                **extended_theta_metrics
            }
            
            layer_dict[f'layer_{layer}'] = layer_metrics
        
        # Create record with metadata
        record = create_record_with_metadata(claim, ground_truth, layer_dict, row)
        thetas.append(record)


def get_theta_extended(df_wo_context, df_with_context, dataset):
    """
    Calculate extended theta metrics for all layers using ground truth.
    
    Args:
        df_wo_context: DataFrame without context
        df_with_context: DataFrame with context
        dataset: Dataset name for ground truth extraction
    
    Returns:
        tuple: (individual_metrics, averaged_metrics)
    """
    thetas = []
    NUM_LAYERS = len(df_wo_context['first_token_activations_true'].iloc[0])
    print(f"Processing {NUM_LAYERS} layers with extended metrics")

    # Normalize and validate claims (skip for CL datasets)
    if dataset not in ['cl_bill', 'cl_company']:
        claims = normalize_claims(df_wo_context, df_with_context)
        print(f"Processing {len(claims)} unique claims")
    
    if dataset in ['cl_bill', 'cl_company']:
        # For CL datasets, process by compound_id directly
        # Validate compound_id uniqueness
        if len(df_wo_context['compound_id'].unique()) != df_wo_context.shape[0]:
            raise ValueError("Duplicate compound_ids found in without_context dataset")
        compound_ids = df_wo_context['compound_id'].unique()
        print(f"Processing {len(compound_ids)} unique compound_ids")
        
        for compound_id in compound_ids:
            df_wo_context_claim = df_wo_context[df_wo_context['compound_id'] == compound_id]
            df_with_context_claim = df_with_context[df_with_context['compound_id'] == compound_id]
            
            if df_wo_context_claim.empty or df_with_context_claim.empty:
                print(f"Missing data for compound_id: {compound_id}")
                continue
            
            # Extract claim from the first row for metadata
            claim = df_wo_context_claim['claim'].iloc[0]
            
            # Process this compound_id
            process_claim_group(df_wo_context_claim, df_with_context_claim, claim, dataset, thetas, NUM_LAYERS)
    else:
        # Use regular claim matching for other datasets
        for claim in claims:
            df_wo_context_claim = df_wo_context[df_wo_context['claim'] == claim]
            df_with_context_claim = df_with_context[df_with_context['claim'] == claim]
            
            if df_wo_context_claim.empty or df_with_context_claim.empty:
                print(f"Missing data for claim: {claim}")
                continue
            
            # Process this claim
            process_claim_group(df_wo_context_claim, df_with_context_claim, claim, dataset, thetas, NUM_LAYERS)
    
    print(f"Calculating averages across {len(thetas)} total samples...")
    
    # Calculate averages across all data points
    thetas_avg = {}
    for layer in range(NUM_LAYERS):
        layer_key = f'layer_{layer}'
        
        # Get all metric names from the first record
        if thetas:
            first_layer_metrics = thetas[0]['layer_dict'][layer_key]
            metric_names = list(first_layer_metrics.keys())
            
            layer_metrics = {}
            for metric_name in metric_names:
                metric_values = [
                    thetas[i]['layer_dict'][layer_key][metric_name] 
                    for i in range(len(thetas))
                ]
                
                layer_metrics[f'{metric_name}_avg'] = float(np.mean(metric_values))
                layer_metrics[f'{metric_name}_std'] = float(np.std(metric_values))
                layer_metrics[f'{metric_name}_median'] = float(np.median(metric_values))
                layer_metrics[f'{metric_name}_min'] = float(np.min(metric_values))
                layer_metrics[f'{metric_name}_max'] = float(np.max(metric_values))
            
            thetas_avg[layer_key] = layer_metrics
    
    return thetas, thetas_avg


def get_subspace_angles(df_wo_context, df_with_context, dataset):
    """ 
    Calculate subspace angles between w/o and w/ context using ground truth
    """
    NUM_LAYERS = len(df_wo_context['first_token_activations_true'].iloc[0])
    print(f"Calculating subspace angles for {NUM_LAYERS} layers")
    
    subspace_angles_dict = {'layer_{layer}'.format(layer=layer): {} for layer in range(NUM_LAYERS)}
    
    for layer in range(NUM_LAYERS):
        print(f"Processing layer {layer}/{NUM_LAYERS-1}")
        
        # Calculate ground truth vector differences for this layer
        wo_context_activations_layer = []
        with_context_activations_layer = []
        
        if dataset in ['cl_bill', 'cl_company']:
            # Use compound_id matching for CL datasets
            wo_compound_ids = set(df_wo_context['compound_id'].unique())
            with_compound_ids = set(df_with_context['compound_id'].unique())
            matching_compound_ids = wo_compound_ids.intersection(with_compound_ids)
            
            print(f"Found {len(matching_compound_ids)} matching compound_ids for layer {layer}")
            
            for compound_id in matching_compound_ids:
                # Get rows for this compound_id
                wo_rows = df_wo_context[df_wo_context['compound_id'] == compound_id]
                with_rows = df_with_context[df_with_context['compound_id'] == compound_id]
                
                # Process each matching pair
                for _, wo_row in wo_rows.iterrows():
                    for _, with_row in with_rows.iterrows():
                        ground_truth = extract_ground_truth(wo_row, dataset)
                        
                        true_wo = np.array(wo_row['first_token_activations_true'][layer])
                        false_wo = np.array(wo_row['first_token_activations_false'][layer])
                        true_with = np.array(with_row['first_token_activations_true'][layer])
                        false_with = np.array(with_row['first_token_activations_false'][layer])
                        
                        wo_vec_diff = get_ground_truth_vector_difference(true_wo, false_wo, ground_truth)
                        with_vec_diff = get_ground_truth_vector_difference(true_with, false_with, ground_truth)
                        
                        wo_context_activations_layer.append(wo_vec_diff)
                        with_context_activations_layer.append(with_vec_diff)
        else:
            # Use claim matching for other datasets
            wo_claims = set(df_wo_context['claim'].unique())
            with_claims = set(df_with_context['claim'].unique())
            matching_claims = wo_claims.intersection(with_claims)
            
            print(f"Found {len(matching_claims)} matching claims for layer {layer}")
            
            for claim in matching_claims:
                # Get rows for this claim
                wo_rows = df_wo_context[df_wo_context['claim'] == claim]
                with_rows = df_with_context[df_with_context['claim'] == claim]
                
                # Process each matching pair
                for _, wo_row in wo_rows.iterrows():
                    for _, with_row in with_rows.iterrows():
                        ground_truth = extract_ground_truth(wo_row, dataset)
                        
                        true_wo = np.array(wo_row['first_token_activations_true'][layer])
                        false_wo = np.array(wo_row['first_token_activations_false'][layer])
                        true_with = np.array(with_row['first_token_activations_true'][layer])
                        false_with = np.array(with_row['first_token_activations_false'][layer])
                        
                        wo_vec_diff = get_ground_truth_vector_difference(true_wo, false_wo, ground_truth)
                        with_vec_diff = get_ground_truth_vector_difference(true_with, false_with, ground_truth)
                        
                        wo_context_activations_layer.append(wo_vec_diff)
                        with_context_activations_layer.append(with_vec_diff)
        
        wo_context_activations_layer = np.array(wo_context_activations_layer).T
        with_context_activations_layer = np.array(with_context_activations_layer).T
        
        if wo_context_activations_layer.shape[0] != with_context_activations_layer.shape[0]:
            print(f"Shape mismatch at layer {layer}: {wo_context_activations_layer.shape} vs {with_context_activations_layer.shape}")
            raise AssertionError(f"Shape mismatch at layer {layer}")
        
        subspace_angle = subspace_angles(wo_context_activations_layer, with_context_activations_layer)
        subspace_angles_dict['layer_{layer}'.format(layer=layer)] = np.degrees(subspace_angle).tolist()
        # print(f"Layer {layer} subspace angle: {np.degrees(subspace_angle)} degrees")
    
    return subspace_angles_dict


def main(wo_context_dir, with_context_dir, dataset, chunk_size=DEFAULT_CHUNK_SIZE, 
         output_dir=DEFAULT_OUTPUT_DIR, vector_diff_dir=DEFAULT_VECTOR_DIFF_DIR,
         filename_wo_context=DEFAULT_FILENAME_WO_CONTEXT, 
         filename_with_context=DEFAULT_FILENAME_WITH_CONTEXT):
    """Main function to process theta calculations with ground truth and extended metrics."""
    print(f"Starting extended theta calculations for {dataset} dataset with ground truth")
    print(f"Input directories: wo_context={wo_context_dir}, with_context={with_context_dir}")
    
    # Check if input directory exists
    if not os.path.exists(wo_context_dir):
        print(f"Input directory not found: {wo_context_dir}")
        raise FileNotFoundError(f"Input directory not found: {wo_context_dir}")
    
    if not os.path.exists(with_context_dir):
        print(f"Input directory not found: {with_context_dir}")
        raise FileNotFoundError(f"Input directory not found: {with_context_dir}")
    
    # Define file paths - using configurable filenames
    wo_context_path = os.path.join(wo_context_dir, filename_wo_context)
    with_context_path = os.path.join(with_context_dir, filename_with_context)
    
    # Check if input files exist
    if not os.path.exists(wo_context_path):
        print(f"Input file not found: {wo_context_path}")
        raise FileNotFoundError(f"Input file not found: {wo_context_path}")
    if not os.path.exists(with_context_path):
        print(f"Input file not found: {with_context_path}")
        raise FileNotFoundError(f"Input file not found: {with_context_path}")
    
    print(f"Loading data from {wo_context_dir} and {with_context_dir}...")
    
    # Load and process data
    if dataset == 'druid':
        USECOLS = USECOLS_DRUID
        CONTEXT_COL = 'evidence'
        print("Using DRUID column configuration")
    elif dataset == 'mf2':
        USECOLS = USECOLS_MF2
        CONTEXT_COL = 'synopsis'
        print("Using MF2 column configuration")
    elif dataset == 'cl_bill':
        USECOLS = USECOLS_CL_BILL
        CONTEXT_COL = 'bill summary'
        print("Using CL bill column configuration")
    elif dataset == 'cl_company':
        USECOLS = USECOLS_CL_COMPANY
        CONTEXT_COL = 'company description'
        print("Using CL company column configuration")
    elif dataset == 'conflictqa':
        USECOLS = USECOLS_CONFLICTQA
        CONTEXT_COL = 'evidence'
        print("Using ConflictQA column configuration")
    else:
        print(f"Invalid dataset: {dataset}")
        raise ValueError(f"Invalid dataset: {dataset}. Supported datasets: druid, mf2, cl_bill, cl_company, conflictqa")

    print("Loading JSONL files...")
    start_time = time.time()
    
    df_wo_context = load_jsonl_to_dataframe(wo_context_path, usecols=USECOLS, chunk_size=chunk_size)
    df_with_context = load_jsonl_to_dataframe(with_context_path, usecols=USECOLS, chunk_size=chunk_size)
    
    # Create compound identifiers for deduplication
    df_wo_context['compound_id'] = df_wo_context.apply(lambda row: create_compound_identifier(row, dataset), axis=1)
    df_with_context['compound_id'] = df_with_context.apply(lambda row: create_compound_identifier(row, dataset), axis=1)
    
    # Deduplicate based on compound_id
    df_wo_context = df_wo_context.drop_duplicates(subset=['compound_id'])
    print(f"Loaded {len(df_wo_context)} without-context samples")

    df_with_context = df_with_context.drop_duplicates(subset=['compound_id', CONTEXT_COL])
    print(f"Loaded {len(df_with_context)} with-context samples")
    
    load_time = time.time() - start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    print(f"Processing {len(df_wo_context)} without-context samples and {len(df_with_context)} with-context samples...")
    
    # Calculate extended thetas
    print("Starting extended theta calculations with ground truth...")
    theta_start_time = time.time()
    
    thetas, thetas_avg = get_theta_extended(df_wo_context, df_with_context, dataset)
    theta_time = time.time() - theta_start_time
    print(f"Extended theta calculations completed in {theta_time:.2f} seconds")
    
    print("Starting subspace angles calculations with ground truth...")
    subspace_start_time = time.time()
    subspace_angles_dict = get_subspace_angles(df_wo_context, df_with_context, dataset)
    subspace_time = time.time() - subspace_start_time
    print(f"Subspace angles calculations completed in {subspace_time:.2f} seconds")
    
    # Save results
    output_files = {
        "thetas_ground_truth_extended.json": thetas,
        "thetas_avg_ground_truth_extended.json": thetas_avg,
        "subspace_angles_ground_truth.json": subspace_angles_dict
    }
    
    print("Saving results...")
    save_start_time = time.time()
    
    for filename, data in output_files.items():
        output_path = os.path.join(with_context_dir, output_dir, vector_diff_dir, filename)
        
        # Create output directory if it doesn't exist
        output_dir_path = os.path.dirname(output_path)
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Created output directory: {output_dir_path}")
        
        print(f"Saving {filename}...")
        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Successfully saved {filename} to {output_path}")
        except Exception as e:
            print(f"Failed to save {filename}: {str(e)}")
            raise
    
    save_time = time.time() - save_start_time
    print(f"All files saved in {save_time:.2f} seconds")
    print(f"Processing complete. Calculated extended theta for {len(thetas)} samples using ground truth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate extended theta angles and metrics between activation vector differences from JSONL files using ground truth")
    parser.add_argument("--wo_context_dir", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("--with_context_dir", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset (druid, mf2, cl_bill, cl_company, conflictqa)")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, 
                       help=f"Chunk size for processing large JSONL files (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory name (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--vector_diff_dir", type=str, default=DEFAULT_VECTOR_DIFF_DIR,
                       help=f"Vector difference subdirectory name (default: {DEFAULT_VECTOR_DIFF_DIR})")
    parser.add_argument("--filename_wo_context", type=str, default=DEFAULT_FILENAME_WO_CONTEXT,
                       help=f"Without context filename (default: {DEFAULT_FILENAME_WO_CONTEXT})")
    parser.add_argument("--filename_with_context", type=str, default=DEFAULT_FILENAME_WITH_CONTEXT,
                       help=f"With context filename (default: {DEFAULT_FILENAME_WITH_CONTEXT})")
    args = parser.parse_args()
    
    main(args.wo_context_dir, args.with_context_dir, args.dataset, args.chunk_size,
         args.output_dir, args.vector_diff_dir, args.filename_wo_context, args.filename_with_context)
