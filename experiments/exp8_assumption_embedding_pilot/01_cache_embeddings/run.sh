#!/bin/bash
#SBATCH --job-name=exp8_embed
#SBATCH --output=experiments/exp8_assumption_embedding_pilot/_log/embed_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:A6000:2
#SBATCH --array=0-403%4
#SBATCH --chdir=.

# Stage 01 GPU array: 20,189 episodes / 50 episodes per task = 404 tasks.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_PATCHES="${NUM_PATCHES:-404}"
DATA_DIR="${DATA_DIR:-experiments/exp8_assumption_embedding_pilot/shared_data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/shared_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
BATCH_SIZE="${BATCH_SIZE:-32}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
STAGE_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot/01_cache_embeddings"
cd "${REPO_ROOT}"

args=(
  --mode worker
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --model-revision "${MODEL_REVISION}"
  --batch-size "${BATCH_SIZE}"
  --devices cuda:0 cuda:1
  --episodes-per-task "${EPISODES_PER_TASK}"
  --num-patches "${NUM_PATCHES}"
  --patch-index "${SLURM_ARRAY_TASK_ID}"
)
[[ -n "${DEVICE:-}" ]] && args+=(--device "${DEVICE}")

"${PYTHON_BIN}" "${STAGE_DIR}/run.py" "${args[@]}"
