"""
Plot logit lens probabilities across datasets.
Creates 4 plots (one per model), each with 4 subplots for different probability types.
"""

import os
import json
import matplotlib.pyplot as plt
import argparse

MODELS = ['llama3-8b', 'mistral-nemo-12b', 'qwen3-4b', 'smollm3-3b']
DATASETS = ['druid/borderlines', 'druid/politifact', 'druid/sciencefeedback_cluster1', 'mf2',
            'corporate_lobbying/bill', 'corporate_lobbying/company', 'conflictqa/counter', 'conflictqa/parametric']

BASE_DIR = "/home//theta-hypothesis/data-generation/outputs"
OUTPUT_BASE_DIR = "/home//theta-hypothesis/plots"

DATASET_COLORS = {
    'druid/borderlines': '#1f77b4',
    'druid/politifact': '#ff7f0e',
    'druid/sciencefeedback_cluster1': '#2ca02c',
    'mf2': '#d62728',
    'corporate_lobbying/bill': '#9467bd',
    'corporate_lobbying/company': '#8c564b',
    'conflictqa/counter': '#7f7f7f',
    'conflictqa/parametric': '#bcbd22',
}

DATASET_LABELS = {
    'druid/borderlines': 'Borderlines',
    'druid/politifact': 'Politifact',
    'druid/sciencefeedback_cluster1': 'ScienceFeedback',
    'mf2': 'MF2',
    'corporate_lobbying/bill': 'Legalbench - Corporate Lobbying (Bill)',
    'corporate_lobbying/company': 'Legalbench - Corporate Lobbying (Company)',
    'conflictqa/counter': 'ConflictQA - Counter',
    'conflictqa/parametric': 'ConflictQA - Parametric',
}

MODEL_NAMES = {
    'llama3-8b': 'LLaMA-3.1-8B-Instruct',
    'mistral-nemo-12b': 'Mistral-Nemo-12B-Instruct',
    'qwen3-4b': 'Qwen3-4B-Instruct',
    'smollm3-3b': 'SmolLM3-3B',
}


def get_logit_lens_json_path(model, dataset):
    return os.path.join(BASE_DIR, model, dataset, 'modeling', 'implicit', 'all_data', 
                        'shuffled', 'baseline', 'logitlens', 'logit_lens_results.json')


def load_logit_lens_data(model, dataset):
    json_path = get_logit_lens_json_path(model, dataset)
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_average_probability_values(data, probability_type):
    all_layer_probs = {}
    
    for record in data:
        if 'layers' not in record:
            continue
        
        for layer_key in sorted(record['layers'].keys(), key=lambda x: int(x.split('_')[1])):
            if layer_key.startswith('layer_0'):
                continue
            layer_num = int(layer_key.split('_')[1])
            layer_data = record['layers'][layer_key]
            
            if probability_type == 'p_true_with_ctx':
                prob = layer_data['p_true_from_true_completion_with_ctx']
            elif probability_type == 'p_true_no_ctx':
                prob = layer_data['p_true_from_true_completion_no_ctx']
            elif probability_type == 'p_false_with_ctx':
                prob = layer_data['p_false_from_false_completion_with_ctx']
            elif probability_type == 'p_false_no_ctx':
                prob = layer_data['p_false_from_false_completion_no_ctx']
            else:
                continue
            
            if layer_num not in all_layer_probs:
                all_layer_probs[layer_num] = []
            all_layer_probs[layer_num].append(prob)
    
    layer_nums = sorted(all_layer_probs.keys())
    prob_values = [sum(all_layer_probs[layer_num]) / len(all_layer_probs[layer_num]) 
                   for layer_num in layer_nums]
    
    return layer_nums, prob_values


def plot_model_probabilities(model, all_data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{MODEL_NAMES[model]} - Logit Lens Probabilities', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    subplot_configs = [
        (0, 0, 'p_true_with_ctx', 'P(True, with context)'),
        (0, 1, 'p_true_no_ctx', 'P(True, without context)'),
        (1, 0, 'p_false_with_ctx', 'P(False, with context)'),
        (1, 1, 'p_false_no_ctx', 'P(False, without context)'),
    ]
    
    for row, col, prob_type, title in subplot_configs:
        ax = axes[row, col]
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        all_layer_nums = []
        for dataset in DATASETS:
            if dataset not in all_data or all_data[dataset] is None:
                continue
            
            data = all_data[dataset]
            layer_nums, prob_values = extract_average_probability_values(data, prob_type)
            all_layer_nums.extend(layer_nums)
            
            color = DATASET_COLORS.get(dataset, '#000000')
            label = DATASET_LABELS.get(dataset, dataset)
            ax.plot(layer_nums, prob_values, marker='o', linestyle='-', 
                   label=label, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if all_layer_nums:
            ax.set_xlim(left=1, right=max(all_layer_nums) + 0.5)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
              title='Dataset', title_fontsize=11, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def extract_probability_difference_values(data, diff_type):
    all_layer_vals = {}
    
    for record in data:
        if 'layers' not in record:
            continue
        
        for layer_key in sorted(record['layers'].keys(), key=lambda x: int(x.split('_')[1])):
            if layer_key.startswith('layer_0'):
                continue
            layer_num = int(layer_key.split('_')[1])
            layer_data = record['layers'][layer_key]
            
            if diff_type == 'diff_no_ctx':
                val = layer_data['p_true_from_true_completion_no_ctx'] - layer_data['p_false_from_false_completion_no_ctx']
            elif diff_type == 'diff_with_ctx':
                val = layer_data['p_true_from_true_completion_with_ctx'] - layer_data['p_false_from_false_completion_with_ctx']
            elif diff_type == 'ratio':
                numerator = abs(layer_data['p_true_from_true_completion_with_ctx'] - layer_data['p_false_from_false_completion_with_ctx'])
                denominator = abs(layer_data['p_true_from_true_completion_no_ctx'] - layer_data['p_false_from_false_completion_no_ctx'])
                if denominator == 0:
                    continue
                val = numerator / denominator
            else:
                continue
            
            if layer_num not in all_layer_vals:
                all_layer_vals[layer_num] = []
            all_layer_vals[layer_num].append(val)
    
    layer_nums = sorted(all_layer_vals.keys())
    avg_values = [sum(all_layer_vals[layer_num]) / len(all_layer_vals[layer_num]) 
                  for layer_num in layer_nums]
    
    return layer_nums, avg_values


def plot_model_probability_differences(model, all_data, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{MODEL_NAMES[model]} - Probability Differences', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    subplot_configs = [
        (0, 'diff_no_ctx', 'P(True) - P(False) without context'),
        (1, 'diff_with_ctx', 'P(True) - P(False) with context'),
        (2, 'ratio', 'Ratio (with/without)'),
    ]
    
    for idx, diff_type, title in subplot_configs:
        ax = axes[idx]
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        all_layer_nums = []
        for dataset in DATASETS:
            if dataset not in all_data or all_data[dataset] is None:
                continue
            
            data = all_data[dataset]
            layer_nums, values = extract_probability_difference_values(data, diff_type)
            all_layer_nums.extend(layer_nums)
            
            color = DATASET_COLORS.get(dataset, '#000000')
            label = DATASET_LABELS.get(dataset, dataset)
            ax.plot(layer_nums, values, marker='o', linestyle='-', 
                   label=label, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel('Layer', fontsize=10)
        if diff_type == 'ratio':
            ax.set_ylabel('Ratio', fontsize=10)
            ax.set_ylim(-10, 10)
        else:
            ax.set_ylabel('Probability Difference', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if diff_type == 'ratio':
            ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
        else:
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if all_layer_nums:
            ax.set_xlim(left=1, right=max(all_layer_nums) + 0.5)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
              title='Dataset', title_fontsize=11, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    for model in MODELS:
        print(f"\nProcessing {model}...")
        all_data = {}
        
        for dataset in DATASETS:
            print(f"  Loading logit lens data for {dataset}...")
            data = load_logit_lens_data(model, dataset)
            if data is None:
                print(f"    Warning: Logit lens results file not found for {dataset}")
                continue
            all_data[dataset] = data
        
        output_filename = f"logit_lens_probabilities_{model}.png"
        output_path = os.path.join(OUTPUT_BASE_DIR, 'logitlens', output_filename)
        plot_model_probabilities(model, all_data, output_path)
        
        output_filename_diff = f"logit_lens_probability_differences_{model}.png"
        output_path_diff = os.path.join(OUTPUT_BASE_DIR, 'logitlens', output_filename_diff)
        plot_model_probability_differences(model, all_data, output_path_diff)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Output directory: {os.path.join(OUTPUT_BASE_DIR, 'logitlens')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

