"""
Theta Analysis Pipeline - Step 1: Dataset Creation
Calls create_dataset.py functionality to create paired datasets from activation data.
"""

import argparse
import os
import logging
from create_dataset import main as create_dataset_main
from random_shuffle_choices import main as random_shuffle_choices_main
from theta_activations_vector_difference_jsonl import main as theta_analysis_main

def run_create_dataset_step(input_dir, output_dir, context_col, merge_col=None):
    """
    Step 1: Create dataset by calling create_dataset.py functionality
    """
    print("=" * 60)
    print("STEP 1: Creating Dataset")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Call the main function from create_dataset.py
    create_dataset_main(input_dir, output_dir, context_col, merge_col)
    print(f"✓ Dataset creation completed. Output saved to: {output_dir}")


def run_random_shuffle_choices_step(input_dir, context_col):
    """
    Step 2: Create random shuffled choices
    """
    print("=" * 60)
    print("STEP 2: Creating Random Shuffled Choices")
    print("=" * 60)
    
    list_of_files = os.listdir(input_dir)
    
    # Run without context
    if ('df_fa_tb_without_context.csv' in list_of_files) or ('df_ta_fb_without_context.csv' in list_of_files):
        random_shuffle_choices_main(input_dir, False, context_col)
    
    # Run with context 
    random_shuffle_choices_main(input_dir, True, context_col)
    
    print(f"✓ Random shuffle completed. Output saved to: {input_dir}")


def run_theta_analysis_step(wo_context_dir, with_context_dir, dataset):
    """
    Step 3: Calculate theta angles
    """
    print("=" * 60)
    print("STEP 3: Calculating Theta Angles")
    print("=" * 60)
    
    # Set up logging for theta analysis
    log_file = os.path.join(with_context_dir, "theta_analysis.log")
    log_level = logging.INFO
    chunk_size = 100
    
    theta_analysis_main(wo_context_dir, with_context_dir, dataset, log_file, log_level, chunk_size)
    print(f"✓ Theta analysis completed. Output saved to: {with_context_dir}")


def main(input_dir, output_dir, context_col, dataset, wo_context_dir, merge_col=None):
    """
    Main pipeline function for complete theta analysis pipeline
    """
    print("Starting Theta Analysis Pipeline")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Context column: {context_col}")
    print(f"Dataset: {dataset}")
    print(f"Without context directory: {wo_context_dir}")
    
    # Step 1: Create dataset
    run_create_dataset_step(input_dir, output_dir, context_col, merge_col)
    
    # Step 2: Create random shuffled choices
    run_random_shuffle_choices_step(output_dir, context_col)

    # Step 3: Calculate theta angles
    run_theta_analysis_step(wo_context_dir, output_dir, dataset)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theta Analysis Pipeline - Complete Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Path to activations directory (implicit or explicit folder)")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Path to output directory for all pipeline results")
    parser.add_argument("--context_col", type=str, required=True, 
                       help="Context column name (e.g., 'evidence' for DRUID, 'synopsis' for MF2)")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset type (e.g., 'druid' or 'mf2')")
    parser.add_argument("--wo_context_dir", type=str, required=True, 
                       help="Path to without context directory")
    parser.add_argument("--merge_col", type=str, 
                       help="Merge column name - if different from (claim, context_col)")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.context_col, args.dataset, args.wo_context_dir, args.merge_col)
