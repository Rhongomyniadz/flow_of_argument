#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --output=_log/assumption_pipeline.out
#SBATCH --partition=gpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=128G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python assumption_extraction/pipeline.py \
    --tensor_parallel_size 8
