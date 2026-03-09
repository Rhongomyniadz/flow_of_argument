#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --output=_log/assumption_pipeline.out
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=96G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python assumption_extraction/pipeline.py \
    --tensor_parallel_size 4 \
    --batch_size 0 \
    --max_tokens 512 \
    --max_model_len 8192
