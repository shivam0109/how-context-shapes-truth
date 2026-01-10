#!/bin/bash
# SLURM batch script for running plot_all_probes.py
#SBATCH --job-name=plot_all_probes
#SBATCH --ntasks=1
#SBATCH --mem=16000M
#SBATCH --time=02:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --output=logs/plot_all_probes_%j.out
#SBATCH --error=logs/plot_all_probes_%j.err

python ../plot_all_probes.py
