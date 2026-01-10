"""
Utility functions for ground truth-based activation analysis.
This module provides clean, reusable functions for processing ground truth annotations
and calculating various metrics with proper normalization.
"""

import numpy as np
import pandas as pd 

# Configuration constants
EPSILON = 1e-10  # Small value to avoid division by zero
ZERO_VECTOR_THETA = 90.0  # Default theta angle for zero vectors
COSINE_MIN = -1.0  # Minimum cosine value for clamping
COSINE_MAX = 1.0   # Maximum cosine value for clamping
druid_gt = pd.read_csv('/home/theta-hypothesis/get_datasets/druid/druid_per_claim_ground_truth.csv')

def get_ground_truth_vector_difference(true_activation, false_activation, ground_truth):
    """
    Calculate vector difference based on ground truth.
    
    Args:
        true_activation: True activation vector
        false_activation: False activation vector  
        ground_truth: Boolean indicating which activation represents the true case
    
    Returns:
        numpy.array: Vector difference (ground_truth_true - ground_truth_false)
    """
    if ground_truth:
        # If ground truth is True, then true_activation is correct
        return np.array(true_activation) - np.array(false_activation)
    else:
        # If ground truth is False, then false_activation is correct, so flip the difference
        return np.array(false_activation) - np.array(true_activation)


def calculate_extended_metrics(
    true_with_context,
    false_with_context,
    true_wo_context,
    false_wo_context,
    ground_truth
):
    """
    Calculate extended metrics between activation vectors using ground truth.
    
    Args:
        true_with_context: True activation vector with context
        false_with_context: False activation vector with context
        true_wo_context: True activation vector without context
        false_wo_context: False activation vector without context
        ground_truth: Boolean indicating which activation represents the true case
    
    Returns:
        dict: Dictionary containing magnitude and theta metrics
    """
    # Calculate ground truth-based vector differences
    vec_diff_wo_context = get_ground_truth_vector_difference(
        true_wo_context, false_wo_context, ground_truth
    )
    vec_diff_with_context = get_ground_truth_vector_difference(
        true_with_context, false_with_context, ground_truth
    )
    
    # Normalize by the without-context difference magnitude
    norm_wo = np.linalg.norm(vec_diff_wo_context)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Magnitude metrics (all normalized by norm_wo)
    # 1) true_with vs false_with
    metrics['magnitude_rel_with_true_with_false'] = float(
        np.linalg.norm(vec_diff_with_context) / (norm_wo + EPSILON)
    )
    
    # 2) with vs wo (difference between the two vector differences)
    metrics['magnitude_rel_with_wo'] = float(
        np.linalg.norm(vec_diff_with_context - vec_diff_wo_context) / (norm_wo + EPSILON)
    )
    
    # 3) true_with vs false_wo
    # This should be: true_with - false_wo if ground_truth is True, else false_wo - true_with
    if ground_truth:
        true_with_false_wo_diff = np.asarray(true_with_context) - np.asarray(false_wo_context)
    else:
        true_with_false_wo_diff = np.asarray(false_with_context) - np.asarray(true_wo_context)
    metrics['magnitude_rel_with_true_wo_false'] = float(
        np.linalg.norm(true_with_false_wo_diff) / (norm_wo + EPSILON)
    )
    
    # 4) true_wo vs false_with
    # This should be: true_wo - false_with if ground_truth is True, else false_with - true_wo
    if ground_truth:
        true_wo_false_with_diff = np.asarray(true_wo_context) - np.asarray(false_with_context)
    else:
        true_wo_false_with_diff = np.asarray(false_wo_context) - np.asarray(true_with_context)
    metrics['magnitude_rel_wo_true_with_false'] = float(
        np.linalg.norm(true_wo_false_with_diff) / (norm_wo + EPSILON)
    )
    
    # 5) true_with vs true_wo
    if ground_truth:
        true_with_true_wo_diff = np.asarray(true_with_context) - np.asarray(true_wo_context)
    else:
        true_with_true_wo_diff = np.asarray(false_with_context) - np.asarray(false_wo_context)
    metrics['magnitude_rel_with_true_wo_true'] = float(
        np.linalg.norm(true_with_true_wo_diff) / (norm_wo + EPSILON)
    )
    
    # 6) false_with vs false_wo
    if ground_truth:
        false_with_false_wo_diff = np.asarray(false_with_context) - np.asarray(false_wo_context)
    else:
        false_with_false_wo_diff = np.asarray(true_with_context) - np.asarray(true_wo_context)
    metrics['magnitude_rel_with_false_wo_false'] = float(
        np.linalg.norm(false_with_false_wo_diff) / (norm_wo + EPSILON)
    )
    
    return metrics


def calculate_theta_metrics(
    true_with_context,
    false_with_context,
    true_wo_context,
    false_wo_context,
    ground_truth,
    get_cos_and_theta_func
):
    """
    Calculate theta (angle) metrics between activation vectors using ground truth.
    
    Args:
        true_with_context: True activation vector with context
        false_with_context: False activation vector with context
        true_wo_context: True activation vector without context
        false_wo_context: False activation vector without context
        ground_truth: Boolean indicating which activation represents the true case
        get_cos_and_theta_func: Function to calculate cosine similarity and theta angle
    
    Returns:
        dict: Dictionary containing theta metrics
    """
    # Calculate ground truth-based vector differences
    vec_diff_wo_context = get_ground_truth_vector_difference(
        true_wo_context, false_wo_context, ground_truth
    )
    vec_diff_with_context = get_ground_truth_vector_difference(
        true_with_context, false_with_context, ground_truth
    )
    
    metrics = {}
    
    # Theta metrics (angles in degrees)
    # 1) (with vs without) - main theta metric
    _, theta_with_wo = get_cos_and_theta_func(vec_diff_wo_context, vec_diff_with_context)
    metrics['theta_with_wo'] = float(theta_with_wo)
    
    # 2) (true_with, true_wo)
    _, theta_twith_two = get_cos_and_theta_func(
        np.asarray(true_with_context), np.asarray(true_wo_context)
    )
    metrics['theta_true_with_true_wo'] = float(theta_twith_two)

    # 3) (false_with, false_wo)
    _, theta_fwith_fwo = get_cos_and_theta_func(
        np.asarray(false_with_context), np.asarray(false_wo_context)
    )
    metrics['theta_false_with_false_wo'] = float(theta_fwith_fwo)

    # 4) (true_with, false_wo)
    _, theta_twith_fwo = get_cos_and_theta_func(
        np.asarray(true_with_context), np.asarray(false_wo_context)
    )
    metrics['theta_true_with_false_wo'] = float(theta_twith_fwo)

    # 5) (true_wo, false_with)
    _, theta_two_fwith = get_cos_and_theta_func(
        np.asarray(true_wo_context), np.asarray(false_with_context)
    )
    metrics['theta_true_wo_false_with'] = float(theta_two_fwith)
    
    # 6) (true_with, false_with)
    _, theta_twith_fwith = get_cos_and_theta_func(
        np.asarray(true_with_context), np.asarray(false_with_context)
    )
    metrics['theta_true_with_false_with'] = float(theta_twith_fwith)

    # 7) (true_wo, false_wo)
    _, theta_two_fwo = get_cos_and_theta_func(
        np.asarray(true_wo_context), np.asarray(false_wo_context)
    )
    metrics['theta_true_wo_false_wo'] = float(theta_two_fwo)
    
    return metrics


def calculate_basic_metrics(
    true_with_context,
    false_with_context,
    true_wo_context,
    false_wo_context,
    ground_truth,
    get_cos_and_theta_func
):
    """
    Calculate basic metrics (cosine similarity, theta, magnitude) using ground truth.
    
    Args:
        true_with_context: True activation vector with context
        false_with_context: False activation vector with context
        true_wo_context: True activation vector without context
        false_wo_context: False activation vector without context
        ground_truth: Boolean indicating which activation represents the true case
        get_cos_and_theta_func: Function to calculate cosine similarity and theta angle
    
    Returns:
        dict: Dictionary containing basic metrics
    """
    # Calculate ground truth-based vector differences
    vec_diff_wo_context = get_ground_truth_vector_difference(
        true_wo_context, false_wo_context, ground_truth
    )
    vec_diff_with_context = get_ground_truth_vector_difference(
        true_with_context, false_with_context, ground_truth
    )
    
    # Calculate cosine similarity and theta angle
    cos, theta = get_cos_and_theta_func(vec_diff_wo_context, vec_diff_with_context)
    
    # Calculate magnitude metrics
    magnitude = np.linalg.norm(vec_diff_with_context - vec_diff_wo_context)
    magnitude_diff = np.linalg.norm(vec_diff_with_context) - np.linalg.norm(vec_diff_wo_context)
    
    return {
        'cos': float(cos),
        'theta': float(theta),
        'magnitude': float(magnitude),
        'magnitude_diff': float(magnitude_diff)
    }


def normalize_claims(df_wo_context, df_with_context):
    """
    Normalize claims for matching and verify data integrity.
    
    Args:
        df_wo_context: DataFrame without context
        df_with_context: DataFrame with context
    
    Returns:
        list: List of unique normalized claims
    """
    # Normalize claims for matching
    df_wo_context['claim'] = df_wo_context['claim'].apply(lambda x: x.lower().strip())
    df_with_context['claim'] = df_with_context['claim'].apply(lambda x: x.lower().strip())
    
    # Verify data integrity
    if len(df_wo_context['claim'].unique()) != df_wo_context.shape[0]:
        raise ValueError("Duplicate claims found in without_context dataset")
    
    claims = df_wo_context['claim'].unique().tolist()
    return claims


def extract_ground_truth(row, dataset):
    """
    Extract ground truth boolean value from dataset-specific columns.
    
    Args:
        row: DataFrame row containing the data
        dataset: Dataset name ('druid', 'mf2', 'cl_bill', 'cl_company', 'conflictqa')
    
    Returns:
        bool: Ground truth value (True/False)
    
    Raises:
        ValueError: If dataset is not supported or ground truth value is invalid
    """
    # if dataset == 'druid':
    #     # DRUID: factcheck_verdict_true -> Boolean True, Boolean False, string True, string False, string Half-True
    #     verdict = row['factcheck_verdict_true']
    #     if isinstance(verdict, bool):
    #         return verdict
    #     elif isinstance(verdict, str):
    #         verdict_lower = verdict.lower().strip()
    #         if verdict_lower in ['true', 'half-true', 'half true']:
    #             return True
    #         elif verdict_lower == 'false':
    #             return False
    #         else:
    #             raise ValueError(f"Invalid DRUID verdict value: {verdict}")
    #     else:
    #         raise ValueError(f"Invalid DRUID verdict type: {type(verdict)}")
    
    if dataset == 'druid':
        # DRUID: per_claim_ground_truth.csv
        claim = row['claim']
        if claim in druid_gt['claim'].values:
            return bool(druid_gt[druid_gt['claim'] == claim]['ground_truth'].iloc[0])
        else:
            raise ValueError(f"Claim {claim} not found in DRUID ground truth")

    elif dataset == 'mf2':
        # MF2: claim_type_true -> string: true_claim, false_claim
        claim_type = row['claim_type_true']
        if isinstance(claim_type, str):
            claim_type_lower = claim_type.lower().strip()
            if claim_type_lower == 'true_claim':
                return True
            elif claim_type_lower == 'false_claim':
                return False
            else:
                raise ValueError(f"Invalid MF2 claim_type value: {claim_type}")
        else:
            raise ValueError(f"Invalid MF2 claim_type type: {type(claim_type)}")
    
    elif dataset in ['cl_bill', 'cl_company']:
        # CL_BILL and CL_COMPANY: answer_true -> 'Yes' and 'No'
        answer = row['answer_true']
        if isinstance(answer, str):
            answer_lower = answer.lower().strip()
            if answer_lower == 'yes':
                return True
            elif answer_lower == 'no':
                return False
            else:
                raise ValueError(f"Invalid {dataset} answer value: {answer}")
        else:
            raise ValueError(f"Invalid {dataset} answer type: {type(answer)}")
    
    elif dataset == 'conflictqa':
        # ConflictQA: ground_truth -> boolean True and False
        ground_truth = row['ground_truth_true']
        if isinstance(ground_truth, (bool, np.bool_)):
            return bool(ground_truth)  # Convert numpy.bool_ to Python bool
        else:
            raise ValueError(f"Invalid ConflictQA ground_truth type: {type(ground_truth)}")
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def extract_bill_identifier(prompt_true, dataset):
    """
    Extract bill identifier from prompt_true field for CL datasets.
    
    Args:
        prompt_true: The prompt_true field containing the full prompt
        dataset: Dataset name ('cl_bill' or 'cl_company')
    
    Returns:
        str: Extracted bill title or bill identifier
    """
    import re
    
    if dataset == 'cl_bill':
        # Extract content between [bill title]: <...>
        pattern = r'\[bill title\]:\s*<([^>]+)>'
    elif dataset == 'cl_company':
        # Extract content between [bill]: <...>
        pattern = r'\[bill\]:\s*<([^>]+)>'
    else:
        raise ValueError(f"Unsupported dataset for bill extraction: {dataset}")
    
    match = re.search(pattern, prompt_true, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Could not extract bill identifier from prompt for {dataset}")


def create_record_with_metadata(claim, ground_truth, layer_dict, row):
    """
    Create record with claim, ground_truth, layer_dict, and all other available columns.
    
    Args:
        claim: Claim text
        ground_truth: Ground truth boolean value
        layer_dict: Dictionary containing layer-wise metrics
        row: DataFrame row with additional metadata
    
    Returns:
        dict: Complete record with all metadata
    """
    record = {
        'claim': claim,
        'ground_truth': ground_truth,
        'layer_dict': layer_dict
    }
    
    # Add all other columns from the row (excluding activation columns and metadata)
    for col in row.index:
        if col not in ['claim', 'ground_truth', 'layer_dict', 'first_token_activations_true', 'first_token_activations_false']:
            record[col] = row[col]
    
    return record
