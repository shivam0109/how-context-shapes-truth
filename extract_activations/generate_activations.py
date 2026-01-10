"""
This script is used to generate 
1. counterfactual data. 
2. Activations 
3. Aggregate activations to get : 
    1) First token activations for all layers
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import time
import os
import yaml 
import json
import argparse
import huggingface_hub
import re 
# Suppresses all warnings and informational messages
logging.set_verbosity_error()


# Set device to cuda if available, otherwise mps, otherwise cpu
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}") 
HF_TOKEN = os.getenv("HF_TOKEN")

def login_to_huggingface(token=HF_TOKEN):
    """
    Login to Hugging Face Hub. If a token is provided, use it; otherwise, will prompt or use environment variable.
    Args:
        token (str, optional): Hugging Face access token. 
    Returns:
        None
    """
    if token is not None:
        huggingface_hub.login(token=token)
    else:
        huggingface_hub.login()


def read_config(config_path):
    """
    Reads a YAML configuration file and returns its contents as a dictionary.
    Args:
        config_path (str): Path to the YAML config file.
    Returns:
        dict: Configuration data.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_prompt_template(prompt_template_path):
    """
    Get the prompt from the config
    Args:
        prompt_template_path (str): Path to the prompt template file.
    Returns:
        str: Prompt template.
    """
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"Prompt template file not found: {prompt_template_path}")
    
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()
    print(f"Loaded prompt template from: {prompt_template_path}")
    return prompt_template

# Remove examples from the dataframe that are already used in the prompt
def get_claims_used_in_prompt(prompt_template):
    """
    Extract claims used in the prompt template.
    Args:
        prompt_template (str): The prompt template string.
    Returns:
        List[str]: List of claims found in the prompt.
    """
    pattern = re.compile(r"\[claim\]: <(.*?)>")
    # Use findall() to extract all matching groups from the text
    extracted_claims = pattern.findall(prompt_template)
    return extracted_claims

def remove_prompt_claims(df, prompt_template):
    """
    Remove claims from dataframe that are already used in the prompt template.
    Args:
        df (pd.DataFrame): Input dataframe.
        prompt_template (str): Prompt template string.
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    print("Original DF Shape: ", df.shape)
    extracted_claims = get_claims_used_in_prompt(prompt_template)
    df_new = df[~df['claim'].isin(extracted_claims)].copy()
    print("New DF Shape after removing prompt examples: ", df_new.shape)
    return df_new


def create_prompt_wo_context_druid(df, prompt_template, tokenizer):
    """
    Create prompt without context for Druid dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    unique_claims = list(df['claim'].unique())
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for claim in unique_claims:
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[evidence]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[evidence]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[evidence]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[evidence]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'factcheck_verdict': 'None', 'evidence_stance': 'None'})
        prompts_tb.append({'prompt': prompt_tb, 'factcheck_verdict': 'None', 'evidence_stance': 'None'})
        prompts_fa.append({'prompt': prompt_fa, 'factcheck_verdict': 'None', 'evidence_stance': 'None'})
        prompts_fb.append({'prompt': prompt_fb, 'factcheck_verdict': 'None', 'evidence_stance': 'None'})
    
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_with_context_druid(df, colname, prompt_template, tokenizer):
    """
    Create prompt with context for Druid dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt':prompt_ta, 'factcheck_verdict':row['factcheck_verdict'], 'evidence_stance':row['evidence_stance']})
        prompts_tb.append({'prompt':prompt_tb, 'factcheck_verdict':row['factcheck_verdict'], 'evidence_stance':row['evidence_stance']})
        prompts_fa.append({'prompt':prompt_fa, 'factcheck_verdict':row['factcheck_verdict'], 'evidence_stance':row['evidence_stance']})
        prompts_fb.append({'prompt':prompt_fb, 'factcheck_verdict':row['factcheck_verdict'], 'evidence_stance':row['evidence_stance']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_wo_context_mf2(df, prompt_template, tokenizer):
    """
    Create prompt without context for MF2 dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
        prompts_tb.append({'prompt': prompt_tb, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
        prompts_fa.append({'prompt': prompt_fa, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
        prompts_fb.append({'prompt': prompt_fb, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_with_context_mf2(df, colname, prompt_template, tokenizer):
    """
    Create prompt with context for MF2 dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <{syn}>\n".format(syn=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <{syn}>\n".format(syn=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <{syn}>\n".format(syn=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[movie name]: <{mn} ({yr})>\n".format(mn=row['title'], yr=row['year']) + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[synopsis]: <{syn}>\n".format(syn=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
        prompts_tb.append({'prompt': prompt_tb, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
        prompts_fa.append({'prompt': prompt_fa, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
        prompts_fb.append({'prompt': prompt_fb, 'movie_id': row['movie_id'], 'claim_id': row['claim_id'], 'granularity': row['granularity'], 'category': row['category'], 'claim_type': row['claim_type']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_wo_context_cl_bill(df, prompt_template, tokenizer):
    """
    Create prompt without context for Corporate Lobbying dataset. Bill is the context.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'index': row['index'], 'answer': row['answer']})
        prompts_tb.append({'prompt': prompt_tb, 'index': row['index'], 'answer': row['answer']})
        prompts_fa.append({'prompt': prompt_fa, 'index': row['index'], 'answer': row['answer']})
        prompts_fb.append({'prompt': prompt_fb, 'index': row['index'], 'answer': row['answer']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_with_context_cl_bill(df, colname, prompt_template, tokenizer):
    """
    Create prompt with context for Corporate Lobbying dataset. Bill is the context.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <{bs}>\n".format(bs=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <{bs}>\n".format(bs=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <{bs}>\n".format(bs=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[company description]: <{desc}>\n".format(desc=row['company_description']) + "[bill title]: <{tt}>\n".format(tt=row['bill_title']) + "[bill summary]: <{bs}>\n".format(bs=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'index': row['index'], 'answer': row['answer']})
        prompts_tb.append({'prompt': prompt_tb, 'index': row['index'], 'answer': row['answer']})
        prompts_fa.append({'prompt': prompt_fa, 'index': row['index'], 'answer': row['answer']})
        prompts_fb.append({'prompt': prompt_fb, 'index': row['index'], 'answer': row['answer']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_wo_context_cl_company(df, prompt_template, tokenizer):
    """
    Create prompt without context for Corporate Lobbying dataset. Company Description is the context.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'index': row['index'], 'answer': row['answer']})
        prompts_tb.append({'prompt': prompt_tb, 'index': row['index'], 'answer': row['answer']})
        prompts_fa.append({'prompt': prompt_fa, 'index': row['index'], 'answer': row['answer']})
        prompts_fb.append({'prompt': prompt_fb, 'index': row['index'], 'answer': row['answer']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_with_context_cl_company(df, colname, prompt_template, tokenizer):
    """
    Create prompt with context for Corporate Lobbying dataset. Company Description is the context.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <{desc}>\n".format(desc=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <{desc}>\n".format(desc=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <{desc}>\n".format(desc=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[bill]: <{b}>\n".format(b=row['bill']) + "[company description]: <{desc}>\n".format(desc=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'index': row['index'], 'answer': row['answer']})
        prompts_tb.append({'prompt': prompt_tb, 'index': row['index'], 'answer': row['answer']})
        prompts_fa.append({'prompt': prompt_fa, 'index': row['index'], 'answer': row['answer']})
        prompts_fb.append({'prompt': prompt_fb, 'index': row['index'], 'answer': row['answer']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb

def create_prompt_wo_context_privacyqa(df, prompt_template, tokenizer):
    """
    Create prompt without context for PrivacyQA dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    unique_claims = list(df['claim'].unique())
    for claim in unique_claims:
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[policy_text]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[policy_text]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[policy_text]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=claim) + "[policy_text]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'original_index': 'None'})
        prompts_tb.append({'prompt': prompt_tb, 'original_index': 'None'})
        prompts_fa.append({'prompt': prompt_fa, 'original_index': 'None'})
        prompts_fb.append({'prompt': prompt_fb, 'original_index': 'None'})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_with_context_privacyqa(df, colname, prompt_template, tokenizer):
    """
    Create prompt with context for PrivacyQA dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[policy_text]: <{pt}>\n".format(pt=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[policy_text]: <{pt}>\n".format(pt=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[policy_text]: <{pt}>\n".format(pt=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[policy_text]: <{pt}>\n".format(pt=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'original_index': row['original_index']})
        prompts_tb.append({'prompt': prompt_tb, 'original_index': row['original_index']})
        prompts_fa.append({'prompt': prompt_fa, 'original_index': row['original_index']})
        prompts_fb.append({'prompt': prompt_fb, 'original_index': row['original_index']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb


def create_prompt_wo_context_conflictqa(df, prompt_template, tokenizer):
    """
    Create prompt without context for ConflictQA dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    df_copy = df.copy() # ConflictQA has unique claims, so dont need to create list of unique claims
    for idx, row in df_copy.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <None>\n" + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <None>\n" + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'index': row['index'], 'ground_truth': row['ground_truth']})
        prompts_tb.append({'prompt': prompt_tb, 'index': row['index'], 'ground_truth': row['ground_truth']})
        prompts_fa.append({'prompt': prompt_fa, 'index': row['index'], 'ground_truth': row['ground_truth']})
        prompts_fb.append({'prompt': prompt_fb, 'index': row['index'], 'ground_truth': row['ground_truth']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb

def create_prompt_with_context_conflictqa(df, colname, prompt_template, tokenizer):
    """
    Create prompt with context for ConflictQA dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    prompt_template = prompt_template.replace('<|eot_id|>', tokenizer.eos_token)
    prompts_ta = []
    prompts_tb = []
    prompts_fa = []
    prompts_fb = []
    for idx, row in df.iterrows():
        prompt_ta = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (A\n[completion]:"
        prompt_tb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (B\n[completion]:"
        prompt_fa = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) False (B) True> \n[selected choice]: (A\n[completion]:"
        prompt_fb = prompt_template + "[claim]: <{clm}>\n".format(clm=row['claim']) + "[evidence]: <{ev}>\n".format(ev=row[colname]) + "[choices]: <(A) True (B) False> \n[selected choice]: (B\n[completion]:"
        prompts_ta.append({'prompt': prompt_ta, 'index': row['index'], 'ground_truth': row['ground_truth']})
        prompts_tb.append({'prompt': prompt_tb, 'index': row['index'], 'ground_truth': row['ground_truth']})
        prompts_fa.append({'prompt': prompt_fa, 'index': row['index'], 'ground_truth': row['ground_truth']})
        prompts_fb.append({'prompt': prompt_fb, 'index': row['index'], 'ground_truth': row['ground_truth']})
    return prompts_ta, prompts_tb, prompts_fa, prompts_fb



def create_prompt_wo_context(df, prompt_template, tokenizer, dataset_name):
    """
    Create prompt without context. 
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
        dataset_name (str): Name of the dataset.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    if dataset_name == "druid":
        return create_prompt_wo_context_druid(df, prompt_template, tokenizer)
    elif dataset_name == "mf2":
        return create_prompt_wo_context_mf2(df, prompt_template, tokenizer)
    elif dataset_name == "cl_bill":
        return create_prompt_wo_context_cl_bill(df, prompt_template, tokenizer)
    elif dataset_name == "cl_company":
        return create_prompt_wo_context_cl_company(df, prompt_template, tokenizer)
    elif dataset_name == "privacyqa":
        return create_prompt_wo_context_privacyqa(df, prompt_template, tokenizer)
    elif dataset_name == "conflictqa":
        return create_prompt_wo_context_conflictqa(df, prompt_template, tokenizer)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def create_prompt_with_context(df, colname, prompt_template, tokenizer, dataset_name):
    """
    Create prompt with context. 
    Args:
        df (pd.DataFrame): DataFrame containing the claims.
        colname (str): Column name to use for evidence.
        prompt_template (str): Prompt template.
        tokenizer (AutoTokenizer): Tokenizer.
        dataset_name (str): Name of the dataset.
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Four lists of prompt dictionaries.
    """
    if dataset_name == "druid":
        return create_prompt_with_context_druid(df, colname, prompt_template, tokenizer)
    elif dataset_name == "mf2":
        return create_prompt_with_context_mf2(df, colname, prompt_template, tokenizer)
    elif dataset_name == "cl_bill":
        return create_prompt_with_context_cl_bill(df, colname, prompt_template, tokenizer)
    elif dataset_name == "cl_company":
        return create_prompt_with_context_cl_company(df, colname, prompt_template, tokenizer)
    elif dataset_name == "privacyqa":
        return create_prompt_with_context_privacyqa(df, colname, prompt_template, tokenizer)
    elif dataset_name == "conflictqa":
        return create_prompt_with_context_conflictqa(df, colname, prompt_template, tokenizer)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def load_model(model_name):
    """
    Load the model.
    Args:
        model_name (str): Name of the model.
    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: Tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Model '{model_name}' loaded successfully.")
    
    return tokenizer, model


def get_max_pooled_activations(activations):
    """
    Input: activations for each layer for each token of shape (num_layers, num_tokens, hidden_dim)
    Output: max pooled activations for each layer for each token of shape (num_layers, hidden_dim)
    """
    max_pooled_activations = torch.max(activations, dim=1)[0] 
    return max_pooled_activations

def get_mean_pooled_activations(activations):
    """
    Input: activations for each layer for each token of shape (num_layers, num_tokens, hidden_dim)
    Output: mean pooled activations for each layer for each token of shape (num_layers, hidden_dim)
    """
    mean_pooled_activations = torch.mean(activations, dim=1)
    return mean_pooled_activations

def get_concat_activations(mean_pooled_activations, max_pooled_activations):
    """
    Input: mean pooled activations and max pooled activations for each layer for each token of shape (num_layers, hidden_dim)
    Output: concat of mean pooled and max pooled activations for each layer for each token of shape (num_layers, 2*hidden_dim)
    """
    concat_activations = torch.cat([mean_pooled_activations, max_pooled_activations], dim=1)
    return concat_activations

def get_last_token_activations(activations):
    """
    Input: activations for each layer for each token of shape (num_layers, num_tokens, hidden_dim)
    Output: activations for the last token for each layer of shape (num_layers, hidden_dim)
    """
    last_token_activations = activations[:, -1, :]
    return last_token_activations.tolist()

def get_first_token_activations(activations):
    """
    Input: activations for each layer for each token of shape (num_layers, num_tokens, hidden_dim)
    Output: activations for the first token for each layer of shape (num_layers, hidden_dim)
    """
    first_token_activations = activations[:, 0, :]
    return first_token_activations.tolist()

def check_instruction_followed(prompt_selected_choice, completion):
    """
    Check if the instruction has been followed in the completion.
    Args:
        prompt_selected_choice (bool): The expected choice (True/False).
        completion (str): The model's completion.
    Returns:
        bool: True if instruction was followed, False otherwise.
    """
    completion = completion.strip()
    completion_choice_true = 'true' in completion.split('.')[0].lower()
    completion_choice_false = 'false' in completion.split('.')[0].lower()
    if not completion_choice_true and not completion_choice_false:
        return "None"
    return completion_choice_true == prompt_selected_choice

def _process_activations(generated_token_hidden_states):
    """
    Processes hidden states from a single example returned by model.generate
    (with output_hidden_states=True, return_dict_in_generate=True).
    Only the *newly generated* tokens are present in generated_token_hidden_states.

    Input: generated_token_hidden_states - hidden_states from model.generate()
    Returns: activations_by_layer - Tensor of shape (num_layers+1, num_generated_tokens, hidden_dim)
    """
    if not generated_token_hidden_states:
        return torch.empty(0, 0, 0)

    # Determine true number of layers (including embeddings at index 0)
    total_layers = len(generated_token_hidden_states[0])

    # Build per-layer tensors of shape (num_generated_tokens, hidden_dim)
    layer_major_activations = []
    for l in range(total_layers):
        # gather the (1, hidden_dim) tensor for each generated token
        layer_l_tokens = [step[l][:, -1, :] for step in generated_token_hidden_states]
        # concat on dim=0 => (num_generated_tokens, hidden_dim)
        layer_major_activations.append(torch.cat(layer_l_tokens, dim=0))

    # Stack into (layers, tokens, hidden)
    activations_by_layer = torch.stack(layer_major_activations, dim=0)

    return activations_by_layer


def _inference_with_activations_helper(prompt, tokenizer, model, max_length):
    """
    Inference with activations.
    Args:
        prompt (str): Prompt.
        tokenizer (AutoTokenizer): Tokenizer.
        model (AutoModelForCausalLM): Model.
        max_length (int): Maximum number of new tokens to generate.
    Returns:
        Tuple[str, torch.Tensor]: Completion text and activations tensor.
    """ 

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False,
                                 output_hidden_states=True, return_dict_in_generate=True)

    completion = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    activations_by_layer = _process_activations(outputs.hidden_states)
    
    return completion, activations_by_layer


def _process_prompt(prompt):
    """
    Process the prompt. Keep only the part after [Input], corresponding to the claim. 
    Args:
        prompt (str): Prompt.
    Returns:
        str: Processed prompt.
    """ 
    processed_prompt = prompt.split('[Input]')[-1]
    return processed_prompt


def _process_completion(completion):
    """
    Process the completion. Keep only the part after [completion], corresponding to the completion. 
    Args:
        completion (str): Completion.
    Returns:
        str: Processed completion.
    """ 
    processed_completion = completion.split('[completion]:')[-1]
    return processed_completion

def inference_with_activations(prompts, filename, tokenizer, model, max_length, selected_choice, dataset_name, chunk_size=10):
    """
    Inference with activations.
    Args:
        prompts (list): List of prompts.
        filename (str): Name of the file to save the data.
        tokenizer (AutoTokenizer): Tokenizer.
        model (AutoModelForCausalLM): Model.
        max_length (int): Maximum number of new tokens to generate.
        selected_choice (bool): Expected choice (True/False).
        chunk_size (int): Size of the chunk for processing.
    """
    num_prompts = len(prompts)
    with open(filename, 'w') as f:
        for i in range(0, num_prompts, chunk_size):
            prompt_chunk = prompts[i:i + chunk_size]
            print(f"Processing chunk {i // chunk_size + 1} / {(num_prompts - 1) // chunk_size + 1} (prompts {i} to {min(i + chunk_size, num_prompts) - 1})")
            outputs_for_chunk = []
            
            for prompt in prompt_chunk:
                completion, activations_by_layer = _inference_with_activations_helper(prompt['prompt'], tokenizer, model, max_length)
                processed_prompt = _process_prompt(prompt['prompt'])
                processed_completion = _process_completion(completion)
                instruction_followed = check_instruction_followed(selected_choice, processed_completion)
                # Check if activations are valid
                if activations_by_layer.numel() == 0:
                    print(f"Warning: Empty activations for prompt {len(outputs_for_chunk)}")
                    continue
                
                first_token_activations = get_first_token_activations(activations_by_layer)
                if dataset_name == "druid":
                    output = {
                        'prompt': processed_prompt,
                        'completion': processed_completion,
                        'first_token_activations': first_token_activations,
                        'factcheck_verdict': prompt['factcheck_verdict'],
                        'evidence_stance': prompt['evidence_stance'],
                        'instruction_followed': instruction_followed
                    }
                    outputs_for_chunk.append(output)
                elif dataset_name == "mf2":
                    output = {
                        'prompt': processed_prompt,
                        'completion': processed_completion,
                        'first_token_activations': first_token_activations,
                        'instruction_followed': instruction_followed,
                        'movie_id': prompt['movie_id'],
                        'claim_id': prompt['claim_id'],
                        'granularity': prompt['granularity'],
                        'category': prompt['category'],
                        'claim_type': prompt['claim_type']
                    }
                    outputs_for_chunk.append(output)
                elif dataset_name == "cl_bill":
                    output = {
                        'prompt': processed_prompt,
                        'completion': processed_completion,
                        'first_token_activations': first_token_activations,
                        'instruction_followed': instruction_followed,
                        'answer': prompt['answer'],
                        'index': prompt['index']
                    }
                    outputs_for_chunk.append(output)
                elif dataset_name == "cl_company":
                    output = {
                        'prompt': processed_prompt,
                        'completion': processed_completion,
                        'first_token_activations': first_token_activations,
                        'instruction_followed': instruction_followed,
                        'answer': prompt['answer'],
                        'index': prompt['index']
                    }
                    outputs_for_chunk.append(output)
                elif dataset_name == "privacyqa":
                    output = {
                        'prompt': processed_prompt,
                        'completion': processed_completion,
                        'first_token_activations': first_token_activations,
                        'instruction_followed': instruction_followed,
                        'index': prompt['original_index']
                    }
                    outputs_for_chunk.append(output)
                elif dataset_name == "conflictqa":
                    output = {
                        'prompt': processed_prompt,
                        'completion': processed_completion,
                        'first_token_activations': first_token_activations,
                        'instruction_followed': instruction_followed,
                        'index': prompt['index'],
                        'ground_truth': prompt['ground_truth']
                    }
                    outputs_for_chunk.append(output)
                else:
                    raise ValueError(f"Invalid dataset name: {dataset_name}")
            
            # Write chunk to file
            if outputs_for_chunk:
                lines_to_write = '\n'.join([json.dumps(item) for item in outputs_for_chunk])
                f.write(lines_to_write + '\n')
            
            # Clear GPU cache if using CUDA
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

def main(config, run_with_context=False):
    """
    Main function to generate counterfactual data. Save the data to a json file. 
    Args:
        config (dict): Configuration dictionary.
        run_with_context (bool): Whether to generate outputs with context.
    Returns: 
        None
    """
    start_time = time.time() 
    
    # Validate required config keys
    required_keys = ["prompt_template_path", "data_path", "model_name", "output_dir", "max_length", "colname"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Get the prompt template from the config
    prompt_template_path = config["prompt_template_path"]
    colname = config["colname"]
    prompt_template = get_prompt_template(prompt_template_path)
    
    # Get the data from the config
    data_path = config["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Get dataset name from the config
    dataset_name = config["dataset"].strip().lower()
    if dataset_name not in ["druid", "mf2", "cl_bill", "cl_company", "privacyqa", "conflictqa"]:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    df = pd.read_csv(data_path)
    if dataset_name not in ["cl_bill", "cl_company"]:
        df = remove_prompt_claims(df, prompt_template)
    if config["test_run"]:
        print("Running in test mode")
        df = df.sample(n=40, random_state=108)
    
    # Get the model from the config
    model_name = config["model_name"]
    tokenizer, model = load_model(model_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    if run_with_context:
        # Create the prompts with context
        prompts_ta_with_context, prompts_tb_with_context, prompts_fa_with_context, prompts_fb_with_context = create_prompt_with_context(df, colname, prompt_template, tokenizer, dataset_name)
        # Inference with activations with context
        print("Running inference with context")
        print("True A")
        inference_with_activations(prompts_ta_with_context, os.path.join(config["output_dir"], "ta_with_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=True, dataset_name=dataset_name)
        print("True B")
        inference_with_activations(prompts_tb_with_context, os.path.join(config["output_dir"], "tb_with_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=True, dataset_name=dataset_name)
        print("False A")
        inference_with_activations(prompts_fa_with_context, os.path.join(config["output_dir"], "fa_with_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=False, dataset_name=dataset_name)
        print("False B")
        inference_with_activations(prompts_fb_with_context, os.path.join(config["output_dir"], "fb_with_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=False, dataset_name=dataset_name)
    else:
        # Create the prompts without context
        prompts_ta, prompts_tb, prompts_fa, prompts_fb = create_prompt_wo_context(df, prompt_template, tokenizer, dataset_name)
        # Inference with activations without context 
        print("Running inference without context")
        print("True A")
        inference_with_activations(prompts_ta, os.path.join(config["output_dir"], "ta_wo_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=True, dataset_name=dataset_name)
        print("True B")
        inference_with_activations(prompts_tb, os.path.join(config["output_dir"], "tb_wo_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=True, dataset_name=dataset_name)
        print("False A")
        inference_with_activations(prompts_fa, os.path.join(config["output_dir"], "fa_wo_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=False, dataset_name=dataset_name)
        print("False B")
        inference_with_activations(prompts_fb, os.path.join(config["output_dir"], "fb_wo_context_activations.jsonl"), tokenizer, model, config["max_length"], selected_choice=False, dataset_name=dataset_name)

    end_time = time.time()
    print(f"Time taken: {np.round((end_time - start_time)/60, 2)} minutes")


if __name__ == "__main__":
    login_to_huggingface(HF_TOKEN)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--run_with_context", action="store_true", help="If set, generate outputs with context. If not set, generate without context.")
    args = parser.parse_args()
    config = read_config(args.config)
    main(config, run_with_context=args.run_with_context)