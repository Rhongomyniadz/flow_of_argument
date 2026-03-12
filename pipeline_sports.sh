#!/bin/bash
#SBATCH --job-name=pipeline_sports
#SBATCH --output=_log/assumption_pipeline_sports.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python assumption_extraction/pipeline.py \
    --categories sports \
    --tensor_parallel_size 2 \
    --batch_size 0
