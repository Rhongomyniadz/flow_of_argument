#!/bin/bash
#SBATCH --job-name=llm_as_a_judge
#SBATCH --output=llm_as_a_judge.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python label_turn_types.py \
  --input_dir results/political/parsed \
  --output data/turn_type_labeled.json \
  --batch_size 32
