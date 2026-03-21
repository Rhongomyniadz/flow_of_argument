#!/bin/bash
#SBATCH --job-name=stance_labeler_religion
#SBATCH --output=_log/stance_labeler_religion.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/stance_labeler.py \
    --categories religion \
    --tensor_parallel_size 2 \
    --batch_size 0 \
    --k 512
