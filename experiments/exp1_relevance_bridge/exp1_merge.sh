#!/bin/bash
#SBATCH --job-name=exp1_relevance_bridge_merge
#SBATCH --output=_log/exp1_relevance_bridge_merge.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

python experiments/exp1_relevance_bridge/merge_exp1_patches.py \
  --input_dir data/conversation_moves_labeled \
  --output_dir experiments/exp1_relevance_bridge/results \
  --embedding_model_name Qwen/Qwen3-Embedding-4B \
  --embedding_batch_size 8 \
  --embedding_device auto \
  --seed 42
