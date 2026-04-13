#!/bin/bash

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"
EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-100}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results}"
NO_TQDM="${NO_TQDM:-1}"

export INPUT_DIR
export EPISODES_PER_PATCH
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
