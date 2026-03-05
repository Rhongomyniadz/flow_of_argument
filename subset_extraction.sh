#!/bin/bash
#SBATCH --job-name=subset_extraction
#SBATCH --output=_log/subset_extraction.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python data_processing/export_sporc_turns_by_category.py
