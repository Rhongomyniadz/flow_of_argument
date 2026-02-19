#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --output=_log/assumption_pipeline.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python assumption_extraction/pipeline.py
