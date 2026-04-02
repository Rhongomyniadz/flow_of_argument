#!/bin/bash
#SBATCH --job-name=stance_labeler_business
#SBATCH --output=_log/stance_labeler_business.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/stance_labeler.py \
    --categories business \
    --tensor_parallel_size 2 \
    --batch_size 0 \
    --k 1024
