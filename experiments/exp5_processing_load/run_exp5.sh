#!/bin/bash

set -euo pipefail

EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-100}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp5_processing_load/results}"
SILENCE_GAP_QUANTILE="${SILENCE_GAP_QUANTILE:-0.95}"
MIN_SILENCE_GAP="${MIN_SILENCE_GAP:-5.0}"
NO_TQDM="${NO_TQDM:-1}"
INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"

export EPISODES_PER_PATCH
export EMBEDDING_MODEL_NAME
export EMBEDDING_BATCH_SIZE
export OUTPUT_DIR
export SILENCE_GAP_QUANTILE
export MIN_SILENCE_GAP
export NO_TQDM
export INPUT_DIR

bash experiments/exp5_processing_load/submit_exp5_patches.sh
