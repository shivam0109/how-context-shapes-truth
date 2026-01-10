"""
Script to concatenate baseline vs random comparisons from ground truth analysis across models and datasets.
Creates summary dataframes from both comparison types:
1. Baseline vs other random types (common datapoints)
2. All types common (common datapoints across all 6 types)
"""

import argparse
import os
import pandas as pd

# Models and datasets from the bash script
MODELS = ['llama3-8b', 'mistral-nemo-12b', 'qwen3-4b', 'smollm3-3b']

DATASETS = [
    'mf2',
    'druid/borderlines',
    'druid/politifact',
    'druid/sciencefeedback_cluster1',
    'corporate_lobbying/bill',
    'corporate_lobbying/company',
    'conflictqa/counter',
    'conflictqa/parametric',
    'privacyqa'
]

# Mapping from CSV Type labels to column names
TYPE_TO_COLUMN = {
    'Baseline': 'Baseline',
    'Random Char': 'random_char',
    'Random Word': 'random_word',
    'Random Salad': 'random_salad',
    'Random Wiki': 'random_wiki',
    'Random Shuffle': 'random_shuffle'
}

PROMPT_TYPE = 'implicit'


def load_baseline_vs_other(plots_base_dir, model, dataset, prompt_type):
    """
    Load baseline vs other comparison CSV file.
    
    Returns:
        dict: Dictionary with extracted values, or None if file not found
    """
    dataset_slug = dataset.replace('/', '_')
    
    csv_file = os.path.join(
        plots_base_dir,
        model,
        'baseline_random_comparisons_ground_truth',
        dataset_slug,
        f'df_baseline_vs_other_{dataset_slug}_{prompt_type}.csv'
    )
    
    if not os.path.exists(csv_file):
        print(f"Warning: File not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Extract values into dictionaries
    # Structure: one row per random type with baseline and other values
    theta_baseline_dict = {}
    theta_other_dict = {}
    magnitude_baseline_dict = {}
    magnitude_other_dict = {}
    num_common_dict = {}
    theta_wilcoxon_stat_dict = {}
    theta_wilcoxon_pvalue_dict = {}
    magnitude_wilcoxon_stat_dict = {}
    magnitude_wilcoxon_pvalue_dict = {}
    
    for _, row in df.iterrows():
        type_label = row['Type']
        column_name = TYPE_TO_COLUMN.get(type_label)
        
        if column_name:
            theta_baseline_dict[column_name] = row.get('Theta_Baseline_Avg', -1)
            theta_other_dict[column_name] = row.get('Theta_Other_Avg', -1)
            magnitude_baseline_dict[column_name] = row.get('Magnitude_Baseline_Avg', -1)
            magnitude_other_dict[column_name] = row.get('Magnitude_Other_Avg', -1)
            num_common_dict[column_name] = row.get('Num_Common_Datapoints', -1)
            theta_wilcoxon_stat_dict[column_name] = row.get('Theta_Wilcoxon_Statistic', -1)
            theta_wilcoxon_pvalue_dict[column_name] = row.get('Theta_Wilcoxon_PValue', -1)
            magnitude_wilcoxon_stat_dict[column_name] = row.get('Magnitude_Wilcoxon_Statistic', -1)
            magnitude_wilcoxon_pvalue_dict[column_name] = row.get('Magnitude_Wilcoxon_PValue', -1)
    
    return {
        'theta_baseline': theta_baseline_dict,
        'theta_other': theta_other_dict,
        'magnitude_baseline': magnitude_baseline_dict,
        'magnitude_other': magnitude_other_dict,
        'num_common': num_common_dict,
        'theta_wilcoxon_stat': theta_wilcoxon_stat_dict,
        'theta_wilcoxon_pvalue': theta_wilcoxon_pvalue_dict,
        'magnitude_wilcoxon_stat': magnitude_wilcoxon_stat_dict,
        'magnitude_wilcoxon_pvalue': magnitude_wilcoxon_pvalue_dict
    }


def load_all_types_common(plots_base_dir, model, dataset, prompt_type):
    """
    Load all types common comparison CSV file.
    
    Returns:
        dict: Dictionary with extracted values, or None if file not found
    """
    dataset_slug = dataset.replace('/', '_')
    
    csv_file = os.path.join(
        plots_base_dir,
        model,
        'baseline_random_comparisons_ground_truth',
        dataset_slug,
        f'df_all_types_common_{dataset_slug}_{prompt_type}.csv'
    )
    
    if not os.path.exists(csv_file):
        print(f"Warning: File not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Extract values into dictionaries
    # Structure: one row per type (including baseline) with averaged values
    theta_dict = {}
    magnitude_dict = {}
    num_common = None  # Same for all types
    
    for _, row in df.iterrows():
        type_label = row['Type']
        column_name = TYPE_TO_COLUMN.get(type_label, type_label)
        
        theta_dict[column_name] = row.get('Theta_Avg', -1)
        magnitude_dict[column_name] = row.get('Magnitude_Avg', -1)
        
        # Get num_common (should be same for all rows)
        if num_common is None:
            num_common = row.get('Num_Common_Datapoints', -1)
    
    return {
        'theta': theta_dict,
        'magnitude': magnitude_dict,
        'num_common': num_common
    }


def create_summary_dataframes(plots_base_dir, prompt_type, output_dir):
    """
    Create summary dataframes from both comparison types.
    
    Args:
        plots_base_dir: Base directory containing plots/{model}/baseline_random_comparisons_ground_truth/
        prompt_type: Prompt type (implicit/explicit)
        output_dir: Directory to save output dataframes
    """
    print("="*80)
    print("CONCATENATING BASELINE VS RANDOM COMPARISONS (GROUND TRUTH)")
    print("="*80)
    print(f"Plots base directory: {plots_base_dir}")
    print(f"Prompt type: {prompt_type}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # For baseline vs other comparisons (combined theta and magnitude)
    baseline_vs_other_rows = []
    
    # For all types common comparisons (combined theta and magnitude)
    all_types_rows = []
    
    # Process each model-dataset combination
    for model in MODELS:
        for dataset in DATASETS:
            print(f"\nProcessing {model} / {dataset}...")
            
            # Load baseline vs other comparison
            baseline_vs_other = load_baseline_vs_other(plots_base_dir, model, dataset, prompt_type)
            
            if baseline_vs_other is not None:
                # Create one row per random type to preserve Num_Common_Datapoints
                for col in ['random_char', 'random_word', 'random_salad', 'random_wiki', 'random_shuffle']:
                    baseline_theta = baseline_vs_other['theta_baseline'].get(col)
                    baseline_magnitude = baseline_vs_other['magnitude_baseline'].get(col)
                    other_theta = baseline_vs_other['theta_other'].get(col)
                    other_magnitude = baseline_vs_other['magnitude_other'].get(col)
                    num_common = baseline_vs_other['num_common'].get(col, -1)
                    theta_wilcoxon_stat = baseline_vs_other['theta_wilcoxon_stat'].get(col, -1)
                    theta_wilcoxon_pvalue = baseline_vs_other['theta_wilcoxon_pvalue'].get(col, -1)
                    magnitude_wilcoxon_stat = baseline_vs_other['magnitude_wilcoxon_stat'].get(col, -1)
                    magnitude_wilcoxon_pvalue = baseline_vs_other['magnitude_wilcoxon_pvalue'].get(col, -1)
                    
                    # Create combined row with both theta and magnitude, plus statistical tests
                    row = {
                        'Model': model,
                        'Dataset': dataset,
                        'Random_Type': col,
                        'Num_Common_Datapoints': num_common,
                        'Theta_Baseline_Avg': baseline_theta,
                        'Theta_Other_Avg': other_theta,
                        'Theta_Wilcoxon_Statistic': theta_wilcoxon_stat,
                        'Theta_Wilcoxon_PValue': theta_wilcoxon_pvalue,
                        'Magnitude_Baseline_Avg': baseline_magnitude,
                        'Magnitude_Other_Avg': other_magnitude,
                        'Magnitude_Wilcoxon_Statistic': magnitude_wilcoxon_stat,
                        'Magnitude_Wilcoxon_PValue': magnitude_wilcoxon_pvalue,
                    }
                    baseline_vs_other_rows.append(row)
                print(f"  Added baseline vs other data for {model}/{dataset}")
            else:
                print(f"  Skipping baseline vs other for {model}/{dataset} - file not found")
            
            # Load all types common comparison
            all_types = load_all_types_common(plots_base_dir, model, dataset, prompt_type)
            
            if all_types is not None:
                # Create combined row with both theta and magnitude for all types
                row = {
                    'Model': model,
                    'Dataset': dataset,
                    'Num_Common_Datapoints': all_types['num_common'],
                    'Theta_Baseline': all_types['theta'].get('Baseline'),
                    'Theta_random_char': all_types['theta'].get('random_char'),
                    'Theta_random_word': all_types['theta'].get('random_word'),
                    'Theta_random_salad': all_types['theta'].get('random_salad'),
                    'Theta_random_wiki': all_types['theta'].get('random_wiki'),
                    'Theta_random_shuffle': all_types['theta'].get('random_shuffle'),
                    'Magnitude_Baseline': all_types['magnitude'].get('Baseline'),
                    'Magnitude_random_char': all_types['magnitude'].get('random_char'),
                    'Magnitude_random_word': all_types['magnitude'].get('random_word'),
                    'Magnitude_random_salad': all_types['magnitude'].get('random_salad'),
                    'Magnitude_random_wiki': all_types['magnitude'].get('random_wiki'),
                    'Magnitude_random_shuffle': all_types['magnitude'].get('random_shuffle'),
                }
                all_types_rows.append(row)
                print(f"  Added all types common data for {model}/{dataset}")
            else:
                print(f"  Skipping all types common for {model}/{dataset} - file not found")
    
    # Create dataframes
    baseline_vs_other_df = pd.DataFrame(baseline_vs_other_rows)
    all_types_df = pd.DataFrame(all_types_rows)
    
    print("\n" + "="*80)
    print("DATAFRAME SHAPES")
    print("="*80)
    print(f"Baseline vs other (combined): {baseline_vs_other_df.shape}")
    print(f"All types common (combined): {all_types_df.shape}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataframes
    baseline_vs_other_output = os.path.join(
        output_dir, 
        f'df_baseline_vs_other_summary_{prompt_type}.csv'
    )
    all_types_output = os.path.join(
        output_dir, 
        f'df_all_types_common_summary_{prompt_type}.csv'
    )
    
    baseline_vs_other_df.to_csv(baseline_vs_other_output, index=False)
    all_types_df.to_csv(all_types_output, index=False)
    
    print(f"\nSaved baseline vs other summary (combined) to: {baseline_vs_other_output}")
    print(f"Saved all types common summary (combined) to: {all_types_output}")
    
    create_bonferroni_corrected_tables(baseline_vs_other_df, output_dir, prompt_type)
    
    # Print previews
    print("\n" + "="*80)
    print("BASELINE VS OTHER SUMMARY PREVIEW (THETA + MAGNITUDE):")
    print("="*80)
    if not baseline_vs_other_df.empty:
        print(baseline_vs_other_df.head(15).to_string(index=False))
    else:
        print("(No data)")
    
    print("\n" + "="*80)
    print("ALL TYPES COMMON SUMMARY PREVIEW (THETA + MAGNITUDE):")
    print("="*80)
    if not all_types_df.empty:
        print(all_types_df.head(10).to_string(index=False))
    else:
        print("(No data)")
    print("="*80)
    
    return baseline_vs_other_df, all_types_df


def create_bonferroni_corrected_tables(baseline_vs_other_df, output_dir, prompt_type):
    theta_columns = ['Model', 'Dataset', 'Random_Type', 'Num_Common_Datapoints',
                     'Theta_Baseline_Avg', 'Theta_Other_Avg', 'Theta_Wilcoxon_Statistic', 'Theta_Wilcoxon_PValue']
    magnitude_columns = ['Model', 'Dataset', 'Random_Type', 'Num_Common_Datapoints',
                         'Magnitude_Baseline_Avg', 'Magnitude_Other_Avg', 'Magnitude_Wilcoxon_Statistic', 'Magnitude_Wilcoxon_PValue']
    
    theta_df = baseline_vs_other_df[theta_columns].copy()
    theta_df['Theta_Wilcoxon_PValue_Bonferroni'] = theta_df['Theta_Wilcoxon_PValue'] * 160
    theta_df['Theta_Wilcoxon_PValue_Bonferroni'] = theta_df['Theta_Wilcoxon_PValue_Bonferroni'].clip(upper=1.0)
    
    magnitude_df = baseline_vs_other_df[magnitude_columns].copy()
    magnitude_df['Magnitude_Wilcoxon_PValue_Bonferroni'] = magnitude_df['Magnitude_Wilcoxon_PValue'] * 160
    magnitude_df['Magnitude_Wilcoxon_PValue_Bonferroni'] = magnitude_df['Magnitude_Wilcoxon_PValue_Bonferroni'].clip(upper=1.0)
    
    combined_df = baseline_vs_other_df.copy()
    combined_df['Theta_Wilcoxon_PValue_Bonferroni'] = combined_df['Theta_Wilcoxon_PValue'] * 320
    combined_df['Theta_Wilcoxon_PValue_Bonferroni'] = combined_df['Theta_Wilcoxon_PValue_Bonferroni'].clip(upper=1.0)
    combined_df['Magnitude_Wilcoxon_PValue_Bonferroni'] = combined_df['Magnitude_Wilcoxon_PValue'] * 320
    combined_df['Magnitude_Wilcoxon_PValue_Bonferroni'] = combined_df['Magnitude_Wilcoxon_PValue_Bonferroni'].clip(upper=1.0)
    
    theta_output = os.path.join(output_dir, f'df_baseline_vs_other_theta_bonferroni_{prompt_type}.csv')
    magnitude_output = os.path.join(output_dir, f'df_baseline_vs_other_magnitude_bonferroni_{prompt_type}.csv')
    combined_output = os.path.join(output_dir, f'df_baseline_vs_other_combined_bonferroni_{prompt_type}.csv')
    
    theta_df.to_csv(theta_output, index=False)
    magnitude_df.to_csv(magnitude_output, index=False)
    combined_df.to_csv(combined_output, index=False)
    
    print(f"\nSaved theta Bonferroni corrected table to: {theta_output}")
    print(f"Saved magnitude Bonferroni corrected table to: {magnitude_output}")
    print(f"Saved combined Bonferroni corrected table to: {combined_output}")


def main(plots_base_dir, prompt_type, output_dir):
    """
    Main function.
    
    Args:
        plots_base_dir: Base directory containing plots/{model}/baseline_random_comparisons_ground_truth/
        prompt_type: Prompt type (implicit/explicit)
        output_dir: Directory to save output dataframes
    """
    create_summary_dataframes(plots_base_dir, prompt_type, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate baseline vs random comparisons from ground truth analysis across models and datasets"
    )
    parser.add_argument(
        "--plots_base_dir",
        type=str,
        default="/Users/Documents/ucph/theta-hypothesis/plots",
        help="Base directory containing plots/{model}/baseline_random_comparisons_ground_truth/"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="implicit",
        help="Prompt type (implicit or explicit)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/Users/Documents/ucph/theta-hypothesis/baseline_vs_random_summary_ground_truth',
        help="Directory to save output dataframes"
    )
    
    args = parser.parse_args()
    
    main(args.plots_base_dir, args.prompt_type, args.output_dir)

