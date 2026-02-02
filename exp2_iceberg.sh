#!/bin/bash
#SBATCH --job-name=exp2_iceberg
#SBATCH --output=_log/exp2_iceberg.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument


python experiments/exp2_iceberg/test.py --data_dir "data/stance_labeled/512"
