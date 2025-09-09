#!/bin/bash
#SBATCH --job-name=assumption_extraction
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --chdir=/home/edenzha/flow_of_argument

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_USE_FLASH_ATTN=0
export VLLM_USE_FUSED_RMSNORM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python extract_assumption.py \
  --data_path  /home/edenzha/flow_of_argument/data/covid_episodes.jsonl.gz \
  --turns_path /home/edenzha/flow_of_argument/data/covid_episodes_turn.jsonl.gz