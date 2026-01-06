#!/bin/bash
#SBATCH --job-name=llm_as_a_judge
#SBATCH --output=llm_as_a_judge.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

  python llm_as_a_judge.py \
    --input analysis_charts/clarification_prediction/questions.csv \
    --output analysis_charts/clarification_prediction/questions_labeled.csv \
    --batch-size 64