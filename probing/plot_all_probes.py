import os
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ['llama3-8b', 'mistral-nemo-12b', 'qwen3-4b', 'smollm3-3b']
DATASETS = ['druid/borderlines', 'druid/politifact', 'druid/sciencefeedback_cluster1', 'mf2', 
            'corporate_lobbying/bill', 'corporate_lobbying/company', 'conflictqa/counter', 'conflictqa/parametric']
PROBE_TYPES = ['lr', 'mlp', 'mm', 'svm']
CONTEXT_TYPES = ['with_context', 'without_context']

OUTPUT_BASE_DIR = "/home/theta-hypothesis/plots"

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

PROBE_NAMES = {
    'lr': 'Logistic Regression',
    'mlp': 'Multi-Layer Perceptron',
    'mm': 'Mean-Mean Probe',
    'svm': 'Linear SVM',
}


def get_accuracy_csv_path(model, dataset, context_type, probe_type):
    probe_dir_map = {
        'lr': 'logistic_regression',
        'mlp': 'mlp',
        'mm': 'mm',
        'svm': 'svm'
    }
    
    context_filename = 'with_context' if context_type == 'with_context' else 'without_context'
    csv_filename = f'{context_filename}.csv'
    
    return os.path.join(OUTPUT_BASE_DIR, 'probes', probe_dir_map[probe_type], model, dataset, csv_filename)


def load_accuracies_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    accuracies = {int(row['layer']): float(row['accuracy']) for _, row in df.iterrows()}
    return accuracies


def plot_probe_accuracies(probe_type, all_accuracies, output_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{PROBE_NAMES[probe_type]} - Layerwise Accuracy', fontsize=16, fontweight='bold', y=0.995)
    
    model_order = ['llama3-8b', 'mistral-nemo-12b', 'qwen3-4b', 'smollm3-3b']
    context_order = ['without_context', 'with_context']
    
    for context_idx, context_type in enumerate(context_order):
        for model_idx, model in enumerate(model_order):
            ax = axes[context_idx, model_idx]
            
            model_name = MODEL_NAMES[model]
            context_label = 'Without Context' if context_type == 'without_context' else 'With Context'
            ax.set_title(f'{model_name}\n{context_label}', fontsize=11, fontweight='bold')
            
            for dataset in DATASETS:
                if model in all_accuracies and dataset in all_accuracies[model] and context_type in all_accuracies[model][dataset]:
                    accuracies = all_accuracies[model][dataset][context_type]
                    layers = sorted([l for l in accuracies.keys() if l >= 1])
                    acc_values = [accuracies[l] for l in layers]
                    
                    color = DATASET_COLORS.get(dataset, '#000000')
                    label = DATASET_LABELS.get(dataset, dataset)
                    ax.plot(layers, acc_values, marker='o', linestyle='-', label=label, color=color, linewidth=2, markersize=4)
            
            ax.set_xlabel('Layer', fontsize=10)
            if model_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.set_ylim(0.4, 1.0)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=15, 
              title='Dataset', title_fontsize=13, framealpha=0.9, 
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    for probe_type in PROBE_TYPES:
        print(f"\n{'='*60}")
        print(f"Processing {PROBE_NAMES[probe_type]}")
        print(f"{'='*60}")
        
        all_accuracies = {}
        
        for model in MODELS:
            print(f"\nProcessing {model}...")
            all_accuracies[model] = {}
            
            for dataset in DATASETS:
                print(f"  Processing {dataset}...")
                all_accuracies[model][dataset] = {}
                
                for context_type in CONTEXT_TYPES:
                    print(f"    Processing {context_type}...")
                    csv_path = get_accuracy_csv_path(model, dataset, context_type, probe_type)
                    
                    if not os.path.exists(csv_path):
                        print(f"      Warning: CSV not found: {csv_path}")
                        continue
                    
                    accuracies = load_accuracies_from_csv(csv_path)
                    all_accuracies[model][dataset][context_type] = accuracies
                    print(f"      Loaded accuracies for {len(accuracies)} layers")
        
        output_filename = f"all_probes_{probe_type}.png"
        output_path = os.path.join(OUTPUT_BASE_DIR, 'probes', output_filename)
        plot_probe_accuracies(probe_type, all_accuracies, output_path)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Output directory: {os.path.join(OUTPUT_BASE_DIR, 'probes')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
