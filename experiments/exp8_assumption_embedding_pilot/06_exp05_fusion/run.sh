#!/bin/bash
#SBATCH --job-name=exp8_exp05
#SBATCH --output=experiments/exp8_assumption_embedding_pilot/_log/exp05_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --array=0-8%4
#SBATCH --chdir=.

# Stage 06 GPU array: history/full/shuffled x seeds 42/43/44.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-experiments/exp8_assumption_embedding_pilot/shared_data}"
CACHE_DIR="${CACHE_DIR:-experiments/exp8_assumption_embedding_pilot/shared_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/exp05_results}"
FEATURE_DIM="${FEATURE_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
MAX_TRAIN_ANCHORS="${MAX_TRAIN_ANCHORS:-50000}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE="${PATIENCE:-2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
STAGE_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot/06_exp05_fusion"
cd "${REPO_ROOT}"

args=(
  --mode worker
  --data-dir "${DATA_DIR}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --num-patches 9
  --patch-index "${SLURM_ARRAY_TASK_ID}"
  --feature-dim "${FEATURE_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --max-train-anchors "${MAX_TRAIN_ANCHORS}"
  --max-epochs "${MAX_EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
)
[[ -n "${DEVICE:-}" ]] && args+=(--device "${DEVICE}")

"${PYTHON_BIN}" "${STAGE_DIR}/run.py" "${args[@]}"
