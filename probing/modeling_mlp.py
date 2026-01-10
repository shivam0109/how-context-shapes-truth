import numpy as np
import pandas as pd
import json
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import argparse

from modeling_utils import (
    get_dataset_name,
    load_jsonl,
    extract_activations_and_labels,
    create_layer_dataframe,
    plot_accuracies
)


def train_and_evaluate(df, layer):
    activations_list = df[f'activations_layer_{layer}'].tolist()
    X = np.array([np.array(activation) for activation in activations_list])
    y = df['ground_truth'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(512, 128, 64),
        solver='adam',
        activation='tanh',
        learning_rate='constant',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    coefs = [coef.tolist() for coef in model.coefs_]
    intercepts = [intercept.tolist() for intercept in model.intercepts_]
    
    return accuracy, coefs, intercepts


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
        accuracy, coefs, intercepts = train_and_evaluate(df, layer)
        accuracies[layer] = accuracy
        weights[f'layer_{layer}'] = {
            'coefs': coefs,
            'intercepts': intercepts
        }
        print(f"Layer {layer} accuracy: {accuracy:.4f}")
    
    plot_accuracies(accuracies, output_plot, 'Multi-Layer Perceptron')
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
    parser = argparse.ArgumentParser(description='Run MLP on activations')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output plot file path')
    args = parser.parse_args()
    
    main(args.input, args.output)

