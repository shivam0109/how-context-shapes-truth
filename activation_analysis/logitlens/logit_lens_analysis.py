"""
Extract probabilities of True and False across layer activations using Logit Lens.
Stores all probabilities for later analysis.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'activation-analysis'))
from ground_truth_utils import extract_ground_truth, extract_bill_identifier # type: ignore


class LogitLens:
    def __init__(self, model_name, device="cpu"):
        print(f"Loading model components: {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()
        
        self.unembedding_matrix = self.model.lm_head.weight
        
        self.true_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in ["True", " True"]]
        self.false_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in ["False", " False"]]
        
        print(f"Tracking IDs -> True: {self.true_ids} | False: {self.false_ids}")

    def analyze(self, activations_list):
        results = []
        with torch.inference_mode():
            for i, layer_act in enumerate(activations_list):
                act_tensor = torch.tensor(layer_act, device=self.device).unsqueeze(0)
                logits = torch.matmul(act_tensor, self.unembedding_matrix.T)
                probs = F.softmax(logits, dim=-1)
                
                p_true = sum(probs[0, tid].item() for tid in self.true_ids)
                p_false = sum(probs[0, fid].item() for fid in self.false_ids)
                
                results.append((i, p_true, p_false))
        return results


def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _extract_bill_identifier(prompt_true, dataset):
    if dataset == 'cl_bill':
        pattern = r'\[bill title\]:\s*<([^>]+)>'
    elif dataset == 'cl_company':
        pattern = r'\[bill\]:\s*<([^>]+)>'
    else:
        return ''
    match = re.search(pattern, prompt_true or '', re.DOTALL)
    return match.group(1).strip() if match else ''


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


def get_context_column(dataset):
    context_map = {
        'druid': 'evidence',
        'mf2': 'synopsis',
        'cl_bill': 'bill summary',
        'cl_company': 'company description',
        'conflictqa': 'evidence'
    }
    return context_map.get(dataset)


def add_compound_ids(data, dataset):
    for record in data:
        claim = str(record.get('claim', '')).lower().strip()
        prompt_true = record.get('prompt_true', '')
        bill_id = _extract_bill_identifier(prompt_true, dataset).lower().strip()
        record['compound_id'] = f"{claim}|||{bill_id}"


def normalize_claims(data):
    for record in data:
        if 'claim' in record:
            record['claim'] = record['claim'].lower().strip()


def deduplicate_by_key(data, key_field, context_col=None):
    seen = set()
    unique_data = []
    for record in data:
        key = record.get(key_field)
        if context_col:
            context = record.get(context_col, '')
            key = (key, context) if key else None
        if key and key not in seen:
            seen.add(key)
            unique_data.append(record)
    return unique_data


def group_by_key(data, key_field):
    grouped = {}
    for record in data:
        key = record.get(key_field)
        if key:
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
    return grouped


def extract_metadata_fields(record, dataset):
    metadata = {}
    
    if 'prompt_true' in record:
        metadata['prompt_true'] = record['prompt_true']
    
    if 'index' in record:
        metadata['index'] = record['index']
    elif 'index_true' in record:
        metadata['index'] = record['index_true']
    
    if 'claim' in record:
        metadata['claim'] = record['claim']
    
    if dataset == 'druid' and 'evidence' in record:
        metadata['evidence'] = record['evidence']
    elif dataset == 'mf2' and 'synopsis' in record:
        metadata['synopsis'] = record['synopsis']
    elif dataset == 'cl_bill':
        if 'bill summary' in record:
            metadata['bill_summary'] = record['bill summary']
        if 'compound_id' in record:
            metadata['compound_id'] = record['compound_id']
    elif dataset == 'cl_company':
        if 'company description' in record:
            metadata['company_description'] = record['company description']
        if 'compound_id' in record:
            metadata['compound_id'] = record['compound_id']
    
    return metadata


def build_layers_dict(stats_true_with, stats_false_with, stats_true_without, stats_false_without, num_layers):
    layers = {}
    for layer_idx in range(num_layers):
        _, p_true_from_true_with, p_false_from_true_with = stats_true_with[layer_idx]
        _, p_true_from_false_with, p_false_from_false_with = stats_false_with[layer_idx]
        _, p_true_from_true_no, p_false_from_true_no = stats_true_without[layer_idx]
        _, p_true_from_false_no, p_false_from_false_no = stats_false_without[layer_idx]
        
        layers[f'layer_{layer_idx}'] = {
            'p_true_from_true_completion_with_ctx': p_true_from_true_with,
            'p_false_from_true_completion_with_ctx': p_false_from_true_with,
            'p_true_from_false_completion_with_ctx': p_true_from_false_with,
            'p_false_from_false_completion_with_ctx': p_false_from_false_with,
            'p_true_from_true_completion_no_ctx': p_true_from_true_no,
            'p_false_from_true_completion_no_ctx': p_false_from_true_no,
            'p_true_from_false_completion_no_ctx': p_true_from_false_no,
            'p_false_from_false_completion_no_ctx': p_false_from_false_no,
        }
    return layers


def process_record_pair(lens, record_with, record_without, dataset, key_value, key_field):
    metadata = extract_metadata_fields(record_with, dataset)
    metadata[key_field] = key_value
    
    record_for_gt = record_with.copy()
    if 'claim' in record_for_gt:
        record_for_gt['claim'] = record_for_gt['claim'].lower()
    ground_truth = int(extract_ground_truth(record_for_gt, dataset))
    metadata['ground_truth'] = ground_truth
    
    acts_true_with = record_with['first_token_activations_true']
    acts_false_with = record_with['first_token_activations_false']
    acts_true_without = record_without['first_token_activations_true']
    acts_false_without = record_without['first_token_activations_false']
    
    stats_true_with = lens.analyze(acts_true_with)
    stats_false_with = lens.analyze(acts_false_with)
    stats_true_without = lens.analyze(acts_true_without)
    stats_false_without = lens.analyze(acts_false_without)
    
    num_layers = len(stats_true_with)
    if not (len(stats_false_with) == num_layers and 
            len(stats_true_without) == num_layers and 
            len(stats_false_without) == num_layers):
        raise ValueError(f"Mismatched layer counts for {key_field} {key_value}")
    
    layers = build_layers_dict(stats_true_with, stats_false_with, stats_true_without, stats_false_without, num_layers)
    
    return {
        **metadata,
        'layers': layers
    }


def process_matching_records(lens, wo_grouped, with_grouped, dataset, key_field):
    results = []
    missing_keys = []
    for key in wo_grouped.keys():
        if key not in with_grouped:
            missing_keys.append(key)
            continue
        wo_records = wo_grouped[key]
        with_records = with_grouped[key]
        record_without = wo_records[0]
        for record_with in with_records:
            result = process_record_pair(lens, record_with, record_without, dataset, key, key_field)
            results.append(result)
    
    if missing_keys:
        print(f"Found {len(missing_keys)} {key_field}s in without-context but not in with-context")
    
    return results


def prepare_data(data, dataset, context_col, is_with_context):
    if dataset in ['cl_bill', 'cl_company']:
        add_compound_ids(data, dataset)
        key_field = 'compound_id'
        data = deduplicate_by_key(data, key_field, context_col if is_with_context else None)
    else:
        normalize_claims(data)
        key_field = 'claim'
        data = deduplicate_by_key(data, key_field, context_col if is_with_context else None)
    
    return data, key_field


def process_and_save(model_name, input_dir, device="cpu"):
    lens = LogitLens(model_name, device)
    
    jsonl_with_context = os.path.join(input_dir, "df_shuffled_choices_with_context.jsonl")
    jsonl_without_context = os.path.join(input_dir, "df_shuffled_choices_without_context.jsonl")
    
    if not os.path.exists(jsonl_with_context):
        raise FileNotFoundError(f"File not found: {jsonl_with_context}")
    if not os.path.exists(jsonl_without_context):
        raise FileNotFoundError(f"File not found: {jsonl_without_context}")
    
    dataset = get_dataset_name(jsonl_with_context)
    context_col = get_context_column(dataset)
    
    data_with = load_jsonl(jsonl_with_context)
    data_without = load_jsonl(jsonl_without_context)
    
    data_without, _ = prepare_data(data_without, dataset, context_col, is_with_context=False)
    print(f"Loaded {len(data_without)} without-context samples")
    
    data_with, key_field = prepare_data(data_with, dataset, context_col, is_with_context=True)
    print(f"Loaded {len(data_with)} with-context samples")
    
    wo_grouped = group_by_key(data_without, key_field)
    with_grouped = group_by_key(data_with, key_field)
    
    results = process_matching_records(lens, wo_grouped, with_grouped, dataset, key_field)
    
    output_dir = os.path.join(input_dir, 'logitlens')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'logit_lens_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract True/False probabilities using Logit Lens')
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing JSONL files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu/cuda)')
    args = parser.parse_args()
    
    process_and_save(args.model_name, args.input_dir, args.device)


if __name__ == "__main__":
    main()
