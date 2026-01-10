"""
Code to get theta using vector difference 
1. Get the vector difference between two activations - True vs False - without context 
2. Get the vector difference between two activations - True vs False - with context 
3. Get theta across all layers by comparing 1 and 2: (at an individual level and across the dataset)
"""

import argparse
import os
import json
import logging
import time
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.linalg import subspace_angles

USECOLS_DRUID = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'evidence', 
        'factcheck_verdict_true', 'factcheck_verdict_false', 'evidence_stance_true', 'evidence_stance_false']

USECOLS_MF2 = ['first_token_activations_true', 'first_token_activations_false', 'claim', 'synopsis', 
            'movie_id_true', 'granularity_true', 'category_true', 'claim_type_true']

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the script.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path (default: None, logs to console only)
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def convert_activations_to_list(df):
    """Convert string representations of activations to lists."""
    logger = logging.getLogger()
    logger.info("Converting string activations to lists...")
    
    try:
        df['first_token_activations_true'] = df['first_token_activations_true'].apply(literal_eval)
        df['first_token_activations_false'] = df['first_token_activations_false'].apply(literal_eval)
        logger.info("Successfully converted activations to lists")
    except Exception as e:
        logger.info(f"Failed to convert activations to lists: {str(e)}")
        raise
    
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
    logger = logging.getLogger()
    norm_wo = np.linalg.norm(vec_diff_wo_context)
    norm_with = np.linalg.norm(vec_diff_with_context)
    
    # Check for zero vectors to avoid division by zero
    if norm_wo == 0 or norm_with == 0:
        logger.info("Zero vector detected, returning default values")
        return 0.0, 90.0  # Return 90 degrees for zero vectors
    
    cos = np.dot(vec_diff_wo_context, vec_diff_with_context) / (norm_wo * norm_with)
    
    # Clamp cosine to [-1, 1] to avoid numerical errors
    cos = np.clip(cos, -1.0, 1.0)
    theta_radians = np.arccos(cos)
    
    # Convert radians to degrees
    theta_degrees = np.degrees(theta_radians)
    
    logger.debug(f"Calculated cos={cos:.4f}, theta={theta_degrees:.2f}°")
    
    return cos, theta_degrees

def get_theta(df_wo_context, df_with_context):
    """
    Calculate theta angles between vector differences for all layers.
    
    Args:
        df_wo_context: DataFrame without context
        df_with_context: DataFrame with context
    
    Returns:
        tuple: (individual_thetas, averaged_thetas)
    """
    logger = logging.getLogger()
    thetas = []
    NUM_LAYERS = len(df_wo_context['first_token_activations_true'].iloc[0])
    logger.info(f"Processing {NUM_LAYERS} layers")

    # Normalize claims for matching
    df_wo_context['claim'] = df_wo_context['claim'].apply(lambda x: x.lower().strip())
    df_with_context['claim'] = df_with_context['claim'].apply(lambda x: x.lower().strip())
    
    # Verify data integrity
    if len(df_wo_context['claim'].unique()) != df_wo_context.shape[0]:
        logger.info("Duplicate claims found in without_context dataset")
        raise ValueError("Duplicate claims found in without_context dataset")
    
    claims = df_wo_context['claim'].unique().tolist()
    logger.info(f"Processing {len(claims)} unique claims")
    
    for claim in claims:
        df_wo_context_claim = df_wo_context[df_wo_context['claim'] == claim]
        df_with_context_claim = df_with_context[df_with_context['claim'] == claim]
        
        logger.info(f"Found {len(df_with_context_claim)} matching claims")

        if df_wo_context_claim.empty or df_with_context_claim.empty:
            logger.info(f"Missing data for claim: {claim}")
            continue
        
        # Get vector difference without context (should be same for all rows with same claim)
        vec_diff_wo_context_all_layers = (np.array(df_wo_context_claim['first_token_activations_true'].iloc[0]) - 
                                   np.array(df_wo_context_claim['first_token_activations_false'].iloc[0]))
        
        for _, row in df_with_context_claim.iterrows():
            layer_dict = {}
            for layer in range(NUM_LAYERS):
                # Extract the specific layer from both datasets
                vec_diff_wo_context = vec_diff_wo_context_all_layers[layer]
                vec_diff_with_context = (np.array(row['first_token_activations_true'][layer]) - 
                                       np.array(row['first_token_activations_false'][layer]))
                
                cos, theta = get_cos_and_theta(vec_diff_wo_context, vec_diff_with_context)
                magnitude = np.linalg.norm(vec_diff_with_context - vec_diff_wo_context)
                magnitude_diff = np.linalg.norm(vec_diff_with_context) - np.linalg.norm(vec_diff_wo_context)

                layer_dict[f'layer_{layer}'] = {
                    'cos': float(cos),  # Convert numpy types to Python types for JSON serialization
                    'theta': float(theta),
                    'magnitude': float(magnitude),
                    'magnitude_diff': float(magnitude_diff)
                }
            
            # Create record with claim, layer_dict, and all other available columns
            record = {
                'claim': claim,
                'layer_dict': layer_dict
            }
            
            # Add all other columns from the row (excluding claim and layer_dict)
            for col in row.index:
                if col not in ['claim', 'layer_dict']:
                    record[col] = row[col]
            
            thetas.append(record)
    
    logger.info(f"Calculating averages across {len(thetas)} total samples...")
    
    # Calculate averages across all data points
    thetas_avg = {}
    for layer in range(NUM_LAYERS):
        layer_key = f'layer_{layer}'
        layer_thetas = [thetas[i]['layer_dict'][layer_key]['theta'] for i in range(len(thetas))]
        layer_cos = [thetas[i]['layer_dict'][layer_key]['cos'] for i in range(len(thetas))]
        layer_magnitudes = [thetas[i]['layer_dict'][layer_key]['magnitude'] for i in range(len(thetas))]
        layer_magnitude_diffs = [thetas[i]['layer_dict'][layer_key]['magnitude_diff'] for i in range(len(thetas))]
        
        thetas_avg[layer_key] = {
            'theta_avg': float(np.mean(layer_thetas)),
            'cos_avg': float(np.mean(layer_cos)),
            'magnitude_avg': float(np.mean(layer_magnitudes)),
            'magnitude_diff_avg': float(np.mean(layer_magnitude_diffs)),
            'theta_std': float(np.std(layer_thetas)),
            'cos_std': float(np.std(layer_cos)),
            'magnitude_std': float(np.std(layer_magnitudes)),
            'magnitude_diff_std': float(np.std(layer_magnitude_diffs))
        }
    
    return thetas, thetas_avg

def get_subspace_angles(df_wo_context, df_with_context):
    """ 
    Calculate subspace angles between w/o and w/ context
    """
    logger = logging.getLogger()
    NUM_LAYERS = len(df_wo_context['first_token_activations_true'].iloc[0])
    logger.info(f"Calculating subspace angles for {NUM_LAYERS} layers")
    
    wo_context_activations = df_wo_context['first_token_activations_true'].tolist()
    with_context_activations = df_with_context['first_token_activations_true'].tolist()
    subspace_angles_dict = {'layer_{layer}'.format(layer=layer): {} for layer in range(NUM_LAYERS)}
    
    for layer in range(NUM_LAYERS):
        logger.debug(f"Processing layer {layer}/{NUM_LAYERS-1}")
        wo_context_activations_layer = [x[layer] for x in wo_context_activations]
        with_context_activations_layer = [x[layer] for x in with_context_activations]
        wo_context_activations_layer = np.array(wo_context_activations_layer).T
        with_context_activations_layer = np.array(with_context_activations_layer).T
        
        if wo_context_activations_layer.shape[0] != with_context_activations_layer.shape[0]:
            logger.info(f"Shape mismatch at layer {layer}: {wo_context_activations_layer.shape} vs {with_context_activations_layer.shape}")
            raise AssertionError(f"Shape mismatch at layer {layer}")
        
        subspace_angle = subspace_angles(wo_context_activations_layer, with_context_activations_layer)
        subspace_angles_dict['layer_{layer}'.format(layer=layer)] = np.degrees(subspace_angle).tolist()
        logger.debug(f"Layer {layer} subspace angle: {np.degrees(subspace_angle)} degrees")
    
    return subspace_angles_dict
    

def main(wo_context_dir, with_context_dir, dataset, log_file=None, log_level=logging.INFO):
    """Main function to process theta calculations."""
    # Create log directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_level=log_level, log_file=log_file)
    
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Starting theta calculations for {dataset} dataset")
    logger.info(f"Input directories: wo_context={wo_context_dir}, with_context={with_context_dir}")
    
    # Check if input directory exists
    if not os.path.exists(wo_context_dir):
        logger.info(f"Input directory not found: {wo_context_dir}")
        raise FileNotFoundError(f"Input directory not found: {wo_context_dir}")
    
    if not os.path.exists(with_context_dir):
        logger.info(f"Input directory not found: {with_context_dir}")
        raise FileNotFoundError(f"Input directory not found: {with_context_dir}")
    
    # Define file paths
    wo_context_path = os.path.join(wo_context_dir, "df_shuffled_choices_without_context.csv")
    with_context_path = os.path.join(with_context_dir, "df_shuffled_choices_with_context.csv")
    
    # Check if input files exist
    if not os.path.exists(wo_context_path):
        logger.info(f"Input file not found: {wo_context_path}")
        raise FileNotFoundError(f"Input file not found: {wo_context_path}")
    if not os.path.exists(with_context_path):
        logger.info(f"Input file not found: {with_context_path}")
        raise FileNotFoundError(f"Input file not found: {with_context_path}")
    
    logger.info(f"Loading data from {wo_context_dir} and {with_context_dir}...")
    
    # Load and process data
    if dataset == 'druid':
        USECOLS = USECOLS_DRUID
        logger.info("Using DRUID column configuration")
    elif dataset == 'mf2':
        USECOLS = USECOLS_MF2
        logger.info("Using MF2 column configuration")
    else:
        logger.info(f"Invalid dataset: {dataset}")
        raise ValueError(f"Invalid dataset: {dataset}")

    logger.info("Loading CSV files...")
    start_time = time.time()
    
    df_wo_context = pd.read_csv(wo_context_path, usecols=USECOLS)
    df_wo_context = convert_activations_to_list(df_wo_context)
    logger.info(f"Loaded {len(df_wo_context)} without-context samples")

    df_with_context = pd.read_csv(with_context_path, usecols=USECOLS)
    df_with_context = convert_activations_to_list(df_with_context)
    logger.info(f"Loaded {len(df_with_context)} with-context samples")
    
    load_time = time.time() - start_time
    logger.info(f"Data loading completed in {load_time:.2f} seconds")
    logger.info(f"Processing {len(df_wo_context)} without-context samples and {len(df_with_context)} with-context samples...")
    
    # Calculate thetas
    logger.info("Starting theta calculations...")
    theta_start_time = time.time()
    
    thetas, thetas_avg = get_theta(df_wo_context, df_with_context)
    theta_time = time.time() - theta_start_time
    logger.info(f"Theta calculations completed in {theta_time:.2f} seconds")
    
    logger.info("Starting subspace angles calculations...")
    subspace_start_time = time.time()
    subspace_angles_dict = get_subspace_angles(df_wo_context, df_with_context)
    subspace_time = time.time() - subspace_start_time
    logger.info(f"Subspace angles calculations completed in {subspace_time:.2f} seconds")
    
    # Save results
    output_files = {
        "thetas.json": thetas,
        "thetas_avg.json": thetas_avg,
        "subspace_angles.json": subspace_angles_dict
    }
    
    logger.info("Saving results...")
    save_start_time = time.time()
    
    for filename, data in output_files.items():
        output_path = os.path.join(with_context_dir, 'model_outputs', 'vector_diff', filename)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        logger.info(f"Saving {filename}...")
        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully saved {filename} to {output_path}")
        except Exception as e:
            logger.info(f"Failed to save {filename}: {str(e)}")
            raise
    
    save_time = time.time() - save_start_time
    logger.info(f"All files saved in {save_time:.2f} seconds")
    logger.info(f"Processing complete. Calculated theta for {len(thetas)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate theta angles between activation vector differences")
    parser.add_argument("--wo_context_dir", type=str, required=True, help="Input directory containing CSV files")
    parser.add_argument("--with_context_dir", type=str, required=True, help="Input directory containing CSV files")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument("--log_file", type=str, default="custom_logs/theta_vd_logs.txt", help="Log file path (default: custom_logs/theta_vd_logs.txt)")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level (default: INFO)")
    args = parser.parse_args()

    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level.upper())
    
    main(args.wo_context_dir, args.with_context_dir, args.dataset, args.log_file, log_level)