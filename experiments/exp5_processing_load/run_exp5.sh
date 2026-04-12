#!/bin/bash

set -euo pipefail

NUM_PATCHES="${NUM_PATCHES:-16}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp5_processing_load/results}"
NO_TQDM="${NO_TQDM:-1}"
INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"

export NUM_PATCHES
export EMBEDDING_MODEL_NAME
export OUTPUT_DIR
export NO_TQDM
export INPUT_DIR

bash experiments/exp5_processing_load/submit_exp5_patches.sh
