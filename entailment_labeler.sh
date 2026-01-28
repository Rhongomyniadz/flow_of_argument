#!/bin/bash
#SBATCH --job-name=entailment_labeler
#SBATCH --output=_log/entailment_labeler.out
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

export VLLM_USE_V1=0
python data_processing/entailment_labeler.py --k 512
