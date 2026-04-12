#!/bin/bash
#SBATCH --job-name=exp1_relevance_bridge_patch
#SBATCH --output=_log/exp1_relevance_bridge_patch_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

set -euo pipefail

NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES in the sbatch environment.}"
PATCH_INDEX="${SLURM_ARRAY_TASK_ID:?This script is intended to run as a Slurm job array.}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results}"
NO_TQDM="${NO_TQDM:-1}"

CATEGORY_ARGS=(--categories all)
if [[ -n "${CATEGORIES_CSV:-}" ]]; then
  IFS=',' read -r -a CATEGORY_VALUES <<< "${CATEGORIES_CSV}"
  CATEGORY_ARGS=(--categories "${CATEGORY_VALUES[@]}")
fi

EXTRA_ARGS=()
if [[ -n "${MAX_EPISODES_PER_CATEGORY:-}" ]]; then
  EXTRA_ARGS+=(--max_episodes_per_category "${MAX_EPISODES_PER_CATEGORY}")
fi
if [[ "${NO_TQDM}" == "1" ]]; then
  EXTRA_ARGS+=(--no_tqdm)
fi

python -u experiments/exp1_relevance_bridge/exp1_relevance_bridge.py \
  --output_dir "${OUTPUT_DIR}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --num_patches "${NUM_PATCHES}" \
  --patch_index "${PATCH_INDEX}" \
  "${CATEGORY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
