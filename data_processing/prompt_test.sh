#!/bin/bash
#SBATCH --job-name=prompt_test
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument


python prompt_test.py
