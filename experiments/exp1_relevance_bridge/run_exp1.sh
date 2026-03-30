#!/bin/bash
#SBATCH --job-name=exp1_relevance_bridge
#SBATCH --output=_log/exp1_relevance_bridge.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python experiments/exp1_relevance_bridge/exp1_relevance_bridge.py --categories all --no_tqdm
