#!/bin/bash
#SBATCH --job-name=assumption_extraction
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument


python extract_assumption.py \
  --data_path  /home/edenzha/flow_of_argument/data/covid_episodes.jsonl.gz \
  --turns_path /home/edenzha/flow_of_argument/data/covid_episodes_turn.jsonl.gz
