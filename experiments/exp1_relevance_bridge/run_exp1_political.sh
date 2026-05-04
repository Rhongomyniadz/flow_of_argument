#!/bin/bash
#SBATCH --job-name=exp1_political_bridge
#SBATCH --output=_log/exp1_political_bridge_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results_political_test}"
EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-1000}"
MAX_EPISODES_PER_CATEGORY="${MAX_EPISODES_PER_CATEGORY:-}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
SEED="${SEED:-42}"
NO_TQDM="${NO_TQDM:-1}"

build_extra_args() {
  EXTRA_ARGS=()
  if [[ -n "${MAX_EPISODES_PER_CATEGORY}" ]]; then
    EXTRA_ARGS+=(--max_episodes_per_category "${MAX_EPISODES_PER_CATEGORY}")
  fi
  if [[ "${NO_TQDM}" == "1" ]]; then
    EXTRA_ARGS+=(--no_tqdm)
  fi
}

count_political_episodes() {
  export INPUT_DIR MAX_EPISODES_PER_CATEGORY
  python - <<'PY'
from pathlib import Path
import os

input_dir = Path(os.environ["INPUT_DIR"])
max_raw = os.environ.get("MAX_EPISODES_PER_CATEGORY", "").strip()
max_episodes = int(max_raw) if max_raw else None
political_files = sorted((input_dir / "political").glob("*.json"))
if max_episodes is not None:
    political_files = political_files[:max_episodes]
print(len(political_files))
PY
}

submit_exp1_political_stages() {
  TOTAL_EPISODES="$(count_political_episodes)"
  if (( TOTAL_EPISODES < 1 )); then
    echo "No political episodes matched the requested Exp 1 inputs." >&2
    exit 1
  fi

  NUM_PATCHES="${NUM_PATCHES:-$(( (TOTAL_EPISODES + EPISODES_PER_PATCH - 1) / EPISODES_PER_PATCH ))}"
  ARRAY_RANGE="0-$((NUM_PATCHES - 1))"
  EXPORT_VARS=(
    "ALL"
    "INPUT_DIR=${INPUT_DIR}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "EPISODES_PER_PATCH=${EPISODES_PER_PATCH}"
    "NUM_PATCHES=${NUM_PATCHES}"
    "EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}"
    "EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE}"
    "EMBEDDING_DEVICE=${EMBEDDING_DEVICE}"
    "SEED=${SEED}"
    "NO_TQDM=${NO_TQDM}"
  )

  if [[ -n "${MAX_EPISODES_PER_CATEGORY}" ]]; then
    EXPORT_VARS+=("MAX_EPISODES_PER_CATEGORY=${MAX_EPISODES_PER_CATEGORY}")
  fi

  EXPORT_STRING="$(IFS=,; echo "${EXPORT_VARS[*]}")"
  WHITENING_JOB_SUBMISSION="$(
    sbatch --parsable \
      --array="${ARRAY_RANGE}" \
      --export="${EXPORT_STRING},EXP1_STAGE=whitening" \
      "$0"
  )"
  WHITENING_JOB_ID="${WHITENING_JOB_SUBMISSION%%;*}"
  WHITENING_MERGE_JOB_SUBMISSION="$(
    sbatch --parsable \
      --dependency="afterok:${WHITENING_JOB_ID}" \
      --export="${EXPORT_STRING},EXP1_STAGE=whitening_merge" \
      "$0"
  )"
  WHITENING_MERGE_JOB_ID="${WHITENING_MERGE_JOB_SUBMISSION%%;*}"
  PATCH_JOB_SUBMISSION="$(
    sbatch --parsable \
      --dependency="afterok:${WHITENING_MERGE_JOB_ID}" \
      --array="${ARRAY_RANGE}" \
      --export="${EXPORT_STRING},EXP1_STAGE=patch" \
      "$0"
  )"
  PATCH_JOB_ID="${PATCH_JOB_SUBMISSION%%;*}"

  echo "Submitted Exp 1 political whitening array ${WHITENING_JOB_ID} with range ${ARRAY_RANGE}."
  echo "Submitted Exp 1 political whitening merge job ${WHITENING_MERGE_JOB_ID} after ${WHITENING_JOB_ID}."
  echo "Submitted Exp 1 political patch array ${PATCH_JOB_ID} after ${WHITENING_MERGE_JOB_ID}."
}

run_exp1_python() {
  python -u experiments/exp1_relevance_bridge/exp1_relevance_bridge.py "$@"
}

EXP1_STAGE="${EXP1_STAGE:-submit}"
if [[ "${EXP1_STAGE}" == "submit" ]]; then
  submit_exp1_political_stages
  exit 0
fi

NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES in the sbatch environment.}"
build_extra_args

if [[ "${EXP1_STAGE}" == "whitening_merge" ]]; then
  run_exp1_python \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --categories political \
    --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
    --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
    --embedding_device "${EMBEDDING_DEVICE}" \
    --seed "${SEED}" \
    --num_patches "${NUM_PATCHES}" \
    --episodes_per_patch "${EPISODES_PER_PATCH}" \
    --merge_whitening_patches_only \
    "${EXTRA_ARGS[@]}"
  exit 0
fi

PATCH_INDEX="${SLURM_ARRAY_TASK_ID:?EXP1_STAGE=${EXP1_STAGE} must run as a Slurm array task.}"

if [[ "${EXP1_STAGE}" == "whitening" ]]; then
  run_exp1_python \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --categories political \
    --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
    --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
    --embedding_device "${EMBEDDING_DEVICE}" \
    --seed "${SEED}" \
    --num_patches "${NUM_PATCHES}" \
    --patch_index "${PATCH_INDEX}" \
    --episodes_per_patch "${EPISODES_PER_PATCH}" \
    --prepare_whitening_patch_only \
    "${EXTRA_ARGS[@]}"
  exit 0
fi

if [[ "${EXP1_STAGE}" != "patch" ]]; then
  echo "Unknown EXP1_STAGE=${EXP1_STAGE}. Expected submit, whitening, whitening_merge, or patch." >&2
  exit 1
fi

run_exp1_python \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --categories political \
  --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --seed "${SEED}" \
  --num_patches "${NUM_PATCHES}" \
  --patch_index "${PATCH_INDEX}" \
  --episodes_per_patch "${EPISODES_PER_PATCH}" \
  "${EXTRA_ARGS[@]}"
