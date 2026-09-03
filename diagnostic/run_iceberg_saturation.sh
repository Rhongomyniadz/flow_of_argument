#!/bin/bash
#SBATCH --job-name=iceberg_saturation
#SBATCH --output=diagnostic/_log/iceberg_saturation_%j.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/shared/6/projects/flow_of_argument

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-diagnostic/results}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/shared/4/models}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MIN_P="${MIN_P:-0.1}"
TOP_K="${TOP_K:-20}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
SEED="${SEED:-42}"
NO_TQDM="${NO_TQDM:-0}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

ARGS=(
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --download_dir "${DOWNLOAD_DIR}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --batch_size "${BATCH_SIZE}"
  --max_tokens "${MAX_TOKENS}"
  --max_model_len "${MAX_MODEL_LEN}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --min_p "${MIN_P}"
  --top_k "${TOP_K}"
  --repetition_penalty "${REPETITION_PENALTY}"
  --seed "${SEED}"
)

if [[ "${NO_TQDM}" == "1" ]]; then
  ARGS+=(--no_tqdm)
fi

python -u diagnostic/iceberg_saturation.py "${ARGS[@]}"
