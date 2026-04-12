#!/bin/bash

set -euo pipefail

NUM_PATCHES="${NUM_PATCHES:-16}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results}"
NO_TQDM="${NO_TQDM:-1}"

export NUM_PATCHES
export EMBEDDING_MODEL_NAME
export EMBEDDING_BATCH_SIZE
export OUTPUT_DIR
export NO_TQDM

if [[ -n "${CATEGORIES_CSV:-}" ]]; then
  export CATEGORIES_CSV
fi

if [[ -n "${MAX_EPISODES_PER_CATEGORY:-}" ]]; then
  export MAX_EPISODES_PER_CATEGORY
fi

bash experiments/exp1_relevance_bridge/submit_exp1_patches.sh
