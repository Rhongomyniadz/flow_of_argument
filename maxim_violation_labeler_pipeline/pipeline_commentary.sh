#!/bin/bash
#SBATCH --job-name=maxim_violation_labeler_commentary
#SBATCH --output=_log/maxim_violation_labeler_commentary.out
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/maxim_violation_labeler.py \
    --categories commentary \
    --tensor_parallel_size 2 \
    --batch_size 0
