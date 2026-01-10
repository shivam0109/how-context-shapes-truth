"""
Plot correlations between theta/magnitude and logit lens probabilities across datasets.
Creates 4 plots (one per model), each with 4 subplots for different correlation types.
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


def get_correlation_json_path(model, dataset):
    return os.path.join(BASE_DIR, model, dataset, 'modeling', 'implicit', 'all_data', 
                        'shuffled', 'baseline', 'logitlens', 'correlations.json')


def load_correlations(model, dataset):
    json_path = get_correlation_json_path(model, dataset)
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_correlation_values(correlations, correlation_type):
    layer_nums = []
    corr_values = []
    
    for layer_key in sorted(correlations.keys(), key=lambda x: int(x.split('_')[1])):
        if layer_key.startswith('layer_0'):
            continue
        layer_num = int(layer_key.split('_')[1])
        layer_nums.append(layer_num)
        corr_values.append(correlations[layer_key][correlation_type])
    
    return layer_nums, corr_values


def plot_model_correlations(model, all_correlations, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{MODEL_NAMES[model]} - Correlations with Normalized Probability Difference', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    subplot_configs = [
        (0, 'theta_p_correlation', 'Theta vs p'),
        (1, 'magnitude_p_correlation', 'Relative Magnitude vs p'),
    ]
    
    for idx, correlation_type, title in subplot_configs:
        ax = axes[idx]
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        all_layer_nums = []
        for dataset in DATASETS:
            if dataset not in all_correlations or all_correlations[dataset] is None:
                continue
            
            correlations = all_correlations[dataset]
            layer_nums, corr_values = extract_correlation_values(correlations, correlation_type)
            all_layer_nums.extend(layer_nums)
            
            color = DATASET_COLORS.get(dataset, '#000000')
            label = DATASET_LABELS.get(dataset, dataset)
            ax.plot(layer_nums, corr_values, marker='o', linestyle='-', 
                   label=label, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Correlation', fontsize=10)
        ax.set_ylim(-1, 1)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
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
        all_correlations = {}
        
        for dataset in DATASETS:
            print(f"  Loading correlations for {dataset}...")
            correlations = load_correlations(model, dataset)
            if correlations is None:
                print(f"    Warning: Correlation file not found for {dataset}")
                continue
            all_correlations[dataset] = correlations
        
        output_filename = f"logit_lens_correlations_{model}.png"
        output_path = os.path.join(OUTPUT_BASE_DIR, 'logitlens', output_filename)
        plot_model_correlations(model, all_correlations, output_path)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Output directory: {os.path.join(OUTPUT_BASE_DIR, 'logitlens')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

