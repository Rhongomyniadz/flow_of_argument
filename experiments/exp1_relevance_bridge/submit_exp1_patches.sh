#!/bin/bash

set -euo pipefail

NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES before running this script.}"
if (( NUM_PATCHES < 1 )); then
  echo "NUM_PATCHES must be >= 1, got ${NUM_PATCHES}" >&2
  exit 1
fi

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results}"
NO_TQDM="${NO_TQDM:-1}"

EXPORT_VARS=(
  "ALL"
  "NUM_PATCHES=${NUM_PATCHES}"
  "EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}"
  "EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "NO_TQDM=${NO_TQDM}"
)

if [[ -n "${CATEGORIES_CSV:-}" ]]; then
  EXPORT_VARS+=("CATEGORIES_CSV=${CATEGORIES_CSV}")
fi

if [[ -n "${MAX_EPISODES_PER_CATEGORY:-}" ]]; then
  EXPORT_VARS+=("MAX_EPISODES_PER_CATEGORY=${MAX_EPISODES_PER_CATEGORY}")
fi

ARRAY_RANGE="0-$((NUM_PATCHES - 1))"
EXPORT_STRING="$(IFS=,; echo "${EXPORT_VARS[*]}")"

sbatch \
  --array="${ARRAY_RANGE}" \
  --export="${EXPORT_STRING}" \
  experiments/exp1_relevance_bridge/run_exp1_patch.sh
