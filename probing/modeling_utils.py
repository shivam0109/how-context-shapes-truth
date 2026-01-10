import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'activation-analysis'))
from ground_truth_utils import extract_ground_truth # type: ignore


def get_dataset_name(filepath):
    path_parts = filepath.split('/')
    for i, part in enumerate(path_parts):
        if part == 'mf2':
            return 'mf2'
        elif part == 'druid':
            return 'druid'
        elif part == 'corporate_lobbying':
            if i + 1 < len(path_parts) and path_parts[i + 1] == 'bill':
                return 'cl_bill'
            elif i + 1 < len(path_parts) and path_parts[i + 1] == 'company':
                return 'cl_company'
        elif part == 'conflictqa':
            return 'conflictqa'
    raise ValueError(f"Could not determine dataset from path: {filepath}")


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    return data


def extract_activations_and_labels(data, dataset):
    activations_true = []
    activations_false = []
    ground_truths_true = []
    ground_truths_false = []
    
    for row in data:
        activations_true.append(row['first_token_activations_true'])
        activations_false.append(row['first_token_activations_false'])
        if 'claim' in row:
            row['claim'] = row['claim'].lower()
        ground_truth = int(extract_ground_truth(row, dataset))
        ground_truths_true.append(ground_truth)
        ground_truths_false.append(1 - ground_truth)
    
    return activations_true, activations_false, ground_truths_true, ground_truths_false


def create_layer_dataframe(activations_true, activations_false, ground_truths_true, ground_truths_false, layer):
    rows = []
    
    for i in range(len(activations_true)):
        rows.append({
            f'activations_layer_{layer}': activations_true[i][layer],
            'ground_truth': ground_truths_true[i]
        })
        rows.append({
            f'activations_layer_{layer}': activations_false[i][layer],
            'ground_truth': ground_truths_false[i]
        })
    
    return pd.DataFrame(rows)


def plot_accuracies(accuracies, output_path, model_name):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    layers = sorted(accuracies.keys())
    acc_values = [accuracies[l] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, acc_values, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy by Layer')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

