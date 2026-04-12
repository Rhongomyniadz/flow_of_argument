#!/bin/bash

set -euo pipefail

NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES before running this script.}"
if (( NUM_PATCHES < 1 )); then
  echo "NUM_PATCHES must be >= 1, got ${NUM_PATCHES}" >&2
  exit 1
fi

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp5_processing_load/results}"
NO_TQDM="${NO_TQDM:-1}"
INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"

ARRAY_RANGE="0-$((NUM_PATCHES - 1))"
EXPORT_STRING="ALL,NUM_PATCHES=${NUM_PATCHES},EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME},OUTPUT_DIR=${OUTPUT_DIR},NO_TQDM=${NO_TQDM},INPUT_DIR=${INPUT_DIR}"

sbatch \
  --array="${ARRAY_RANGE}" \
  --export="${EXPORT_STRING}" \
  experiments/exp5_processing_load/run_exp5_patch.sh
