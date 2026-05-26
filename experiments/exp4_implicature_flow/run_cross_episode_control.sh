#!/bin/bash
#SBATCH --job-name=exp4_cross_episode_control
#SBATCH --output=_log/exp4_cross_episode_control.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/shared/6/projects/flow_of_argument

set -euo pipefail

PAIR_DIR="${PAIR_DIR:-data/implicature_flow/entailment_pairs_1to10}"
TURN_DIR="${TURN_DIR:-data/stance_labeled/512}"
CATEGORY_LOOKUP_DIR="${CATEGORY_LOOKUP_DIR:-data/conversation_moves_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp4_implicature_flow/results}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SEED="${SEED:-42}"
MIN_OVERLAP="${MIN_OVERLAP:-2}"
MAX_CLAIMS_PER_ASSUMPTION="${MAX_CLAIMS_PER_ASSUMPTION:-15}"
ENTAILMENT_THRESHOLD="${ENTAILMENT_THRESHOLD:-7}"
CONTEXT_W="${CONTEXT_W:-2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-500}"
RETRY_MAX_TOKENS="${RETRY_MAX_TOKENS:-4000}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/shared/4/models}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python experiments/exp4_implicature_flow/exp4_cross_episode_control.py \
    --pair_dir "${PAIR_DIR}" \
    --turn_dir "${TURN_DIR}" \
    --category_lookup_dir "${CATEGORY_LOOKUP_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample_size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --min_overlap "${MIN_OVERLAP}" \
    --max_claims_per_assumption "${MAX_CLAIMS_PER_ASSUMPTION}" \
    --entailment_threshold "${ENTAILMENT_THRESHOLD}" \
    --context_w "${CONTEXT_W}" \
    --model_name "${MODEL_NAME}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" \
    --max_tokens "${MAX_TOKENS}" \
    --retry_max_tokens "${RETRY_MAX_TOKENS}" \
    --download_dir "${DOWNLOAD_DIR}"
