#!/bin/bash
#SBATCH --job-name=turn_type_labeler_sports
#SBATCH --output=_log/turn_type_labeler_sports.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/turn_type_labeler.py \
    --categories sports \
    --tensor_parallel_size 2 \
    --batch_size 0
