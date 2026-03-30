#!/bin/bash
#SBATCH --job-name=exp5_processing_load
#SBATCH --output=_log/exp5_processing_load.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python experiments/exp5_processing_load/exp5.py --no_tqdm
