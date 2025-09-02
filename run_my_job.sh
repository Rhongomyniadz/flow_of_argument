#!/bin/bash
#SBATCH --job-name=assumption_extraction
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:A6000:2 




python extract_assumption.py 
