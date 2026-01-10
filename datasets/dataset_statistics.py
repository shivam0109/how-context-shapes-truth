#!/usr/bin/env python3
"""
Dataset Statistics Script

This script generates comprehensive statistics for the following datasets:
1. Druid Borderlines
2. Druid Politifact 
3. Druid Sciencefeedback cluster 1
4. MF2
5. Legalbench - Corporate lobbying
6. Legalbench - PrivacyQA (privacyqa_claims_postprocessed.csv)
7. Conflictqa - Counter
8. Conflictqa - Parametric

For each dataset, it calculates:
- Basic dataset statistics (number of rows, columns)
- Text length statistics for context columns
- Character count, word count, and sentence count statistics
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import textstat

def count_sentences(text: str) -> int:
    """Count sentences in text using simple sentence boundary detection."""
    if pd.isna(text) or text == '':
        return 0
    # Simple sentence counting - split by common sentence endings
    sentences = re.split(r'[.!?]+', str(text))
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def calculate_readability_scores(text: str) -> Dict[str, float]:
    """Calculate specific readability scores for a text."""
    if pd.isna(text) or text == '':
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'gunning_fog': 0.0,
            'dale_chall_readability_score': 0.0
        }
    
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text)
        }
    except:
        # Return zeros if textstat fails
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'gunning_fog': 0.0,
            'dale_chall_readability_score': 0.0
        }

def calculate_text_stats(text_series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Calculate comprehensive text statistics for a series."""
    # Remove NaN values
    clean_texts = text_series.dropna()
    
    if len(clean_texts) == 0:
        return {
            f"{column_name}_count": 0,
            f"{column_name}_char_count_mean": 0,
            f"{column_name}_char_count_std": 0,
            f"{column_name}_char_count_min": 0,
            f"{column_name}_char_count_max": 0,
            f"{column_name}_word_count_mean": 0,
            f"{column_name}_word_count_std": 0,
            f"{column_name}_word_count_min": 0,
            f"{column_name}_word_count_max": 0,
            f"{column_name}_sentence_count_mean": 0,
            f"{column_name}_sentence_count_std": 0,
            f"{column_name}_sentence_count_min": 0,
            f"{column_name}_sentence_count_max": 0,
            f"{column_name}_flesch_reading_ease_mean": 0,
            f"{column_name}_flesch_reading_ease_std": 0,
            f"{column_name}_flesch_kincaid_grade_mean": 0,
            f"{column_name}_flesch_kincaid_grade_std": 0,
            f"{column_name}_gunning_fog_mean": 0,
            f"{column_name}_gunning_fog_std": 0,
            f"{column_name}_dale_chall_readability_score_mean": 0,
            f"{column_name}_dale_chall_readability_score_std": 0
        }
    
    # Convert to string and calculate statistics
    texts = clean_texts.astype(str)
    
    # Character counts
    char_counts = texts.str.len()
    
    # Word counts
    word_counts = texts.str.split().str.len()
    
    # Sentence counts
    sentence_counts = texts.apply(count_sentences)
    
    # Readability scores
    readability_scores = texts.apply(calculate_readability_scores)
    flesch_reading_ease = [score['flesch_reading_ease'] for score in readability_scores]
    flesch_kincaid_grade = [score['flesch_kincaid_grade'] for score in readability_scores]
    gunning_fog = [score['gunning_fog'] for score in readability_scores]
    dale_chall_readability_score = [score['dale_chall_readability_score'] for score in readability_scores]
    
    return {
        f"{column_name}_count": len(clean_texts),
        f"{column_name}_char_count_mean": char_counts.mean(),
        f"{column_name}_char_count_std": char_counts.std(),
        f"{column_name}_char_count_min": char_counts.min(),
        f"{column_name}_char_count_max": char_counts.max(),
        f"{column_name}_word_count_mean": word_counts.mean(),
        f"{column_name}_word_count_std": word_counts.std(),
        f"{column_name}_word_count_min": word_counts.min(),
        f"{column_name}_word_count_max": word_counts.max(),
        f"{column_name}_sentence_count_mean": sentence_counts.mean(),
        f"{column_name}_sentence_count_std": sentence_counts.std(),
        f"{column_name}_sentence_count_min": sentence_counts.min(),
        f"{column_name}_sentence_count_max": sentence_counts.max(),
        f"{column_name}_flesch_reading_ease_mean": np.mean(flesch_reading_ease),
        f"{column_name}_flesch_reading_ease_std": np.std(flesch_reading_ease),
        f"{column_name}_flesch_kincaid_grade_mean": np.mean(flesch_kincaid_grade),
        f"{column_name}_flesch_kincaid_grade_std": np.std(flesch_kincaid_grade),
        f"{column_name}_gunning_fog_mean": np.mean(gunning_fog),
        f"{column_name}_gunning_fog_std": np.std(gunning_fog),
        f"{column_name}_dale_chall_readability_score_mean": np.mean(dale_chall_readability_score),
        f"{column_name}_dale_chall_readability_score_std": np.std(dale_chall_readability_score)
    }

def analyze_dataset(file_path: str, context_columns: List[str], dataset_name: str) -> Dict[str, Any]:
    """Analyze a single dataset and return statistics."""
    print(f"Analyzing {dataset_name}...")
    
    try:
        # Load dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            print(f"Unsupported file format: {file_path}")
            return {}
        
        # Basic dataset statistics
        stats = {
            'dataset_name': dataset_name,
            'file_path': file_path,
            'total_rows': len(df)
        }
        
        # Analyze each context column
        for col in context_columns:
            if col in df.columns:
                col_stats = calculate_text_stats(df[col], col)
                stats.update(col_stats)
            else:
                print(f"Warning: Column '{col}' not found in {dataset_name}")
                # Add zero stats for missing columns
                col_stats = calculate_text_stats(pd.Series([], dtype=str), col)
                stats.update(col_stats)
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {str(e)}")
        return {
            'dataset_name': dataset_name,
            'file_path': file_path,
            'error': str(e)
        }

def get_dataset_configs():
    """Return the dataset configurations."""
    return [
        {
            'name': 'Druid Borderlines',
            'path': 'druid/borderlines.csv',
            'context_columns': ['evidence']
        },
        {
            'name': 'Druid Politifact',
            'path': 'druid/politifact.csv', 
            'context_columns': ['evidence']
        },
        {
            'name': 'Druid Sciencefeedback Cluster 1',
            'path': 'druid/sciencefeedback_cluster1.csv',
            'context_columns': ['evidence']
        },
        {
            'name': 'MF2',
            'path': 'mf2/mf2.csv',
            'context_columns': ['synopsis']
        },
        {
            'name': 'Legalbench Corporate Lobbying - Bill Summary',
            'path': 'legalbench/corporate_lobbying/corporate_lobbying.csv',
            'context_columns': ['bill_summary']
        },
        {
            'name': 'Legalbench Corporate Lobbying - Company Description',
            'path': 'legalbench/corporate_lobbying/corporate_lobbying.csv',
            'context_columns': ['company_description']
        },
        {
            'name': 'Legalbench PrivacyQA',
            'path': 'legalbench/privacyqa/privacyqa_claims_postprocessed.csv',
            'context_columns': ['policy_text']
        },
        {
            'name': 'ConflictQA Counter',
            'path': 'conflictqa/conflictqa_counter.csv',
            'context_columns': ['evidence']
        },
        {
            'name': 'ConflictQA Parametric',
            'path': 'conflictqa/conflictqa_parametric.csv',
            'context_columns': ['evidence']
        }
    ]

def print_dataset_summary(dataset_name, stats, context_columns):
    """Print summary for a single dataset."""
    print(f"\n{dataset_name}:")
    print(f"  Rows: {stats.get('total_rows', 'N/A')}")
    
    for col in context_columns:
        if f"{col}_count" in stats:
            print(f"  {col} entries: {stats[f'{col}_count']}")
            if stats[f'{col}_count'] > 0:
                print(f"    Avg chars: {stats.get(f'{col}_char_count_mean', 0):.1f}")
                print(f"    Avg words: {stats.get(f'{col}_word_count_mean', 0):.1f}")
                print(f"    Avg sentences: {stats.get(f'{col}_sentence_count_mean', 0):.1f}")
                print(f"    Flesch Reading Ease: {stats.get(f'{col}_flesch_reading_ease_mean', 0):.1f}")
                print(f"    Flesch-Kincaid Grade: {stats.get(f'{col}_flesch_kincaid_grade_mean', 0):.1f}")
                print(f"    Gunning Fog: {stats.get(f'{col}_gunning_fog_mean', 0):.1f}")
                print(f"    Dale-Chall Score: {stats.get(f'{col}_dale_chall_readability_score_mean', 0):.1f}")

def create_summary_dataframe(all_stats):
    """Create a clean dataframe with essential statistics."""
    df_rows = []
    for stats in all_stats:
        if 'error' in stats:
            continue
            
        row = {
            'Dataset': stats['dataset_name'],
            'Rows': stats.get('total_rows', 0)
        }
        
        # Add context column statistics with standardized column names
        for key, value in stats.items():
            if key.endswith('_char_count_mean') and value > 0:
                col_name = key.replace('_char_count_mean', '')
                char_mean = stats.get(f'{col_name}_char_count_mean', 0)
                word_mean = stats.get(f'{col_name}_word_count_mean', 0)
                sent_mean = stats.get(f'{col_name}_sentence_count_mean', 0)
                char_std = stats.get(f'{col_name}_char_count_std', 0)
                word_std = stats.get(f'{col_name}_word_count_std', 0)
                sent_std = stats.get(f'{col_name}_sentence_count_std', 0)
                
                # Readability scores
                flesch_reading_ease_mean = stats.get(f'{col_name}_flesch_reading_ease_mean', 0)
                flesch_reading_ease_std = stats.get(f'{col_name}_flesch_reading_ease_std', 0)
                flesch_kincaid_grade_mean = stats.get(f'{col_name}_flesch_kincaid_grade_mean', 0)
                flesch_kincaid_grade_std = stats.get(f'{col_name}_flesch_kincaid_grade_std', 0)
                gunning_fog_mean = stats.get(f'{col_name}_gunning_fog_mean', 0)
                gunning_fog_std = stats.get(f'{col_name}_gunning_fog_std', 0)
                dale_chall_readability_score_mean = stats.get(f'{col_name}_dale_chall_readability_score_mean', 0)
                dale_chall_readability_score_std = stats.get(f'{col_name}_dale_chall_readability_score_std', 0)
                
                # Use standardized column names
                row['context_col_avg_chars'] = round(char_mean, 1)
                row['context_col_std_chars'] = round(char_std, 1)
                row['context_col_avg_words'] = round(word_mean, 1)
                row['context_col_std_words'] = round(word_std, 1)
                row['context_col_avg_sentences'] = round(sent_mean, 1)
                row['context_col_std_sentences'] = round(sent_std, 1)
                row['context_col_flesch_reading_ease'] = round(flesch_reading_ease_mean, 1)
                row['context_col_flesch_reading_ease_std'] = round(flesch_reading_ease_std, 1)
                row['context_col_flesch_kincaid_grade'] = round(flesch_kincaid_grade_mean, 1)
                row['context_col_flesch_kincaid_grade_std'] = round(flesch_kincaid_grade_std, 1)
                row['context_col_gunning_fog'] = round(gunning_fog_mean, 1)
                row['context_col_gunning_fog_std'] = round(gunning_fog_std, 1)
                row['context_col_dale_chall_readability_score'] = round(dale_chall_readability_score_mean, 1)
                row['context_col_dale_chall_readability_score_std'] = round(dale_chall_readability_score_std, 1)
        
        df_rows.append(row)
    
    return pd.DataFrame(df_rows)

def save_results(summary_df, base_path):
    """Save results to CSV file."""
    # Save summary dataframe to CSV
    csv_output_file = base_path / 'dataset_statistics_summary.csv'
    summary_df.to_csv(csv_output_file, index=False)
    
    return csv_output_file

def main():
    """Main function to analyze all datasets."""
    # Get dataset configurations
    datasets = get_dataset_configs()
    
    # Get the script directory (get_datasets folder)
    base_path = Path(__file__).parent
    all_stats = []
    
    print("=" * 80)
    print("DATASET STATISTICS ANALYSIS")
    print("=" * 80)
    
    # Analyze each dataset
    for dataset in datasets:
        file_path = base_path / dataset['path']
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        stats = analyze_dataset(str(file_path), dataset['context_columns'], dataset['name'])
        all_stats.append(stats)
        
        # Print summary for this dataset
        print_dataset_summary(dataset['name'], stats, dataset['context_columns'])
    
    # Create summary dataframe and save results
    summary_df = create_summary_dataframe(all_stats)
    csv_output_file = save_results(summary_df, base_path)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Summary dataframe saved to: {csv_output_file}")
    print("=" * 80)
    
    print("\nSUMMARY DATAFRAME:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
