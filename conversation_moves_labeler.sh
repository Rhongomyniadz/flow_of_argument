#!/bin/bash
#SBATCH --job-name=conversation_moves_labeler
#SBATCH --output=conversation_moves_labeler.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python conversation_moves_labeler.py \
  --input_dir data/labeled \
  --output_dir data/conversation_moves_labeled \
  --batch_size 64
