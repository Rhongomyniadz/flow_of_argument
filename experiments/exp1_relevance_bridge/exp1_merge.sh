#!/bin/bash
#SBATCH --job-name=exp1_llm_merge
#SBATCH --output=_log/exp1_llm_merge.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results_llm}"
CATEGORIES_CSV="${CATEGORIES_CSV:-all}"
NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES to the patch count used for Exp 1.}"
EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-100}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/shared/4/models}"
SEED="${SEED:-42}"

IFS=',' read -r -a CATEGORY_VALUES <<< "${CATEGORIES_CSV}"

python experiments/exp1_relevance_bridge/merge_exp1_patches.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --categories "${CATEGORY_VALUES[@]}" \
  --num_patches "${NUM_PATCHES}" \
  --episodes_per_patch "${EPISODES_PER_PATCH}" \
  --model_name "${MODEL_NAME}" \
  --download_dir "${DOWNLOAD_DIR}" \
  --seed "${SEED}" \
  --no_tqdm
