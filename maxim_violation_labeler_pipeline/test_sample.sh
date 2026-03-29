#!/bin/bash
#SBATCH --job-name=maxim_violation_labeler_test_sample
#SBATCH --output=_log/maxim_violation_labeler_test_sample.out
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/maxim_violation_labeler.py \
    --categories political \
    --tensor_parallel_size 1 \
    --batch_size 0 \
    --max_episodes 3
