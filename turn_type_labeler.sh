#!/bin/bash
#SBATCH --job-name=stance_labeler
#SBATCH --output=_log/turn_type_labeler.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/turn_type_labeler.py
