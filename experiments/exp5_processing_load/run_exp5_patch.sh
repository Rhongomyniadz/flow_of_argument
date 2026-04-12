#!/bin/bash
#SBATCH --job-name=exp5_processing_load_patch
#SBATCH --output=_log/exp5_processing_load_patch_%A_%a.out
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
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp5_processing_load/results}"
NO_TQDM="${NO_TQDM:-1}"
INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"

EXTRA_ARGS=()
if [[ "${NO_TQDM}" == "1" ]]; then
  EXTRA_ARGS+=(--no_tqdm)
fi

python -u experiments/exp5_processing_load/exp5.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
  --num_patches "${NUM_PATCHES}" \
  --patch_index "${PATCH_INDEX}" \
  "${EXTRA_ARGS[@]}"
