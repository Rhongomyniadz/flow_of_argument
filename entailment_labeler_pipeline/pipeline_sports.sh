#!/bin/bash
#SBATCH --job-name=entailment_labeler_sports
#SBATCH --output=_log/entailment_labeler_sports.out
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/entailment_labeler.py \
    --categories sports \
    --tensor_parallel_size 2 \
    --k 512
