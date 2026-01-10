import numpy as np
import pandas as pd
import json
from scipy.special import expit
from sklearn.model_selection import train_test_split
import argparse

from modeling_utils import (
    get_dataset_name,
    load_jsonl,
    extract_activations_and_labels,
    create_layer_dataframe,
    plot_accuracies
)


class MMProbe:
    def __init__(self, acts, labels):
        pos_activations = acts[labels == 1]
        neg_activations = acts[labels == 0]
        pos_mean = pos_activations.mean(axis=0)
        neg_mean = neg_activations.mean(axis=0)
        self.theta_mm = pos_mean - neg_mean
        self.theta_mm = self.theta_mm / np.linalg.norm(self.theta_mm)
    
    def project(self, x):
        return x @ self.theta_mm
    
    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x)[:, 1] > threshold).astype(np.int32)
    
    def predict_proba(self, x):
        pos_prob = expit(self.project(x))
        neg_prob = 1 - pos_prob
        return np.stack([neg_prob, pos_prob]).T


def train_and_evaluate(df, layer):
    activations_list = df[f'activations_layer_{layer}'].tolist()
    X = np.array([np.array(activation) for activation in activations_list])
    y = df['ground_truth'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    mm = MMProbe(X_train, y_train)
    y_pred = mm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    theta_mm = mm.theta_mm.tolist()
    
    return accuracy, theta_mm


def main(input_jsonl, output_plot):
    dataset = get_dataset_name(input_jsonl)
    data = load_jsonl(input_jsonl)
    activations_true, activations_false, ground_truths_true, ground_truths_false = extract_activations_and_labels(data, dataset)
    
    num_layers = len(activations_true[0])
    accuracies = {}
    weights = {}
    
    for layer in range(1, num_layers):
        print(f"Processing layer {layer}...")
        df = create_layer_dataframe(activations_true, activations_false, ground_truths_true, ground_truths_false, layer)
        accuracy, theta_mm = train_and_evaluate(df, layer)
        accuracies[layer] = accuracy
        weights[f'layer_{layer}'] = {
            'theta_mm': theta_mm
        }
        print(f"Layer {layer} accuracy: {accuracy:.4f}")
    
    plot_accuracies(accuracies, output_plot, 'Mean-Mean Probe')
    print(f"Plot saved to {output_plot}")
    
    output_csv = output_plot.replace('.png', '.csv')
    accuracy_df = pd.DataFrame([{'layer': layer, 'accuracy': acc} for layer, acc in accuracies.items()])
    accuracy_df.to_csv(output_csv, index=False)
    print(f"Accuracies saved to {output_csv}")
    
    output_weights = output_plot.replace('.png', '_weights.json')
    with open(output_weights, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"Weights saved to {output_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mean-Mean Probe on activations')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output plot file path')
    args = parser.parse_args()
    
    main(args.input, args.output)

