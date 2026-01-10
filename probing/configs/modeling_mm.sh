#!/bin/bash
# SLURM batch script for running modeling_mm.py
#SBATCH --job-name=modeling_mm
#SBATCH --ntasks=1
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --output=logs/modeling_mm_%A_%a.out
#SBATCH --error=logs/modeling_mm_%A_%a.err
#SBATCH --array=0-31%8

# Array of datasets to process
DATASETS=(
    "mf2"
    "druid/borderlines"
    "druid/politifact"
    "druid/sciencefeedback_cluster1"
    "corporate_lobbying/bill"
    "corporate_lobbying/company"
    "conflictqa/counter"
    "conflictqa/parametric"
)

MODELS=(
    "llama3-8b"
    "mistral-nemo-12b"
    "qwen3-4b"
    "smollm3-3b"
)

# Base paths
BASE_BASE_DIR="/home/theta-hypothesis/data-generation/outputs"
OUTPUT_BASE_DIR="/home/theta-hypothesis/plots"

# Calculate model and dataset indices from SLURM_ARRAY_TASK_ID
# This creates a grid: for each model, process all datasets
# 4 models * 8 datasets = 32 tasks (array IDs 0-31)
# Array task IDs 0-7: llama3-8b with all 8 datasets
# Array task IDs 8-15: mistral-nemo-12b with all 8 datasets
# Array task IDs 16-23: qwen3-4b with all 8 datasets
# Array task IDs 24-31: smollm3-3b with all 8 datasets
NUM_DATASETS=${#DATASETS[@]}
NUM_MODELS=${#MODELS[@]}

# Get the model and dataset for this array task
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# Construct paths
INPUT_BASE_DIR="${BASE_BASE_DIR}/${MODEL}/${DATASET}/modeling/implicit/all_data/shuffled/baseline"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/probes/mm/${MODEL}/${DATASET}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Input files
INPUT_WITH_CONTEXT="${INPUT_BASE_DIR}/df_shuffled_choices_with_context.jsonl"
INPUT_WITHOUT_CONTEXT="${INPUT_BASE_DIR}/df_shuffled_choices_without_context.jsonl"

# Output files
OUTPUT_WITH_CONTEXT="${OUTPUT_DIR}/with_context.png"
OUTPUT_WITHOUT_CONTEXT="${OUTPUT_DIR}/without_context.png"

echo "Processing model: $MODEL, dataset: $DATASET"

python ../modeling_mm.py \
    --input "$INPUT_WITH_CONTEXT" \
    --output "$OUTPUT_WITH_CONTEXT"

python ../modeling_mm.py \
    --input "$INPUT_WITHOUT_CONTEXT" \
    --output "$OUTPUT_WITHOUT_CONTEXT"

