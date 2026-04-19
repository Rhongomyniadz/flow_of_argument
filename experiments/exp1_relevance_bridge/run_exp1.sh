#!/bin/bash
#SBATCH --job-name=exp1_relevance_bridge_patch
#SBATCH --output=_log/exp1_relevance_bridge_patch_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

set -euo pipefail

INPUT_DIR="data/conversation_moves_labeled"
EPISODES_PER_PATCH="1000"
EMBEDDING_MODEL_NAME="Qwen/Qwen3-Embedding-4B"
EMBEDDING_BATCH_SIZE="8"
EMBEDDING_DEVICE="auto"
OUTPUT_DIR="experiments/exp1_relevance_bridge/results"
NO_TQDM="1"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TOTAL_EPISODES="$(
  python - <<'PY'
from pathlib import Path
import os

input_dir = Path("data/conversation_moves_labeled")
categories_csv = os.environ.get("CATEGORIES_CSV", "").strip()
max_per_category_raw = os.environ.get("MAX_EPISODES_PER_CATEGORY", "").strip()
max_per_category = int(max_per_category_raw) if max_per_category_raw else None

available = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
requested = [item.strip() for item in categories_csv.split(",") if item.strip()]
if not requested or any(item.lower() == "all" for item in requested):
    selected = available
else:
    lookup = {name.lower(): name for name in available}
    selected = []
    for raw_name in requested:
        match = lookup.get(raw_name.lower())
        if match is None:
            raise SystemExit(f"Unknown category: {raw_name}. Available: {', '.join(available)}")
        if match not in selected:
            selected.append(match)

count = 0
for category in selected:
    category_files = sorted((input_dir / category).glob("*.json"))
    if max_per_category is not None:
        category_files = category_files[:max_per_category]
    count += len(category_files)
print(count)
PY
  )"

  if (( TOTAL_EPISODES < 1 )); then
    echo "No episodes matched the requested Exp 1 inputs." >&2
    exit 1
  fi

  NUM_PATCHES=$(( (TOTAL_EPISODES + EPISODES_PER_PATCH - 1) / EPISODES_PER_PATCH ))

  EXPORT_VARS=(
    "ALL"
    "INPUT_DIR=${INPUT_DIR}"
    "NUM_PATCHES=${NUM_PATCHES}"
    "EPISODES_PER_PATCH=${EPISODES_PER_PATCH}"
    "EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}"
    "EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE}"
    "EMBEDDING_DEVICE=${EMBEDDING_DEVICE}"
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
    "$0"
  exit 0
fi

NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES in the sbatch environment.}"
PATCH_INDEX="${SLURM_ARRAY_TASK_ID:?This script is intended to run as a Slurm job array.}"

CATEGORY_ARGS=(--categories all)
if [[ -n "${CATEGORIES_CSV:-}" ]]; then
  IFS=',' read -r -a CATEGORY_VALUES <<< "${CATEGORIES_CSV}"
  CATEGORY_ARGS=(--categories "${CATEGORY_VALUES[@]}")
fi

EXTRA_ARGS=()
if [[ -n "${MAX_EPISODES_PER_CATEGORY:-}" ]]; then
  EXTRA_ARGS+=(--max_episodes_per_category "${MAX_EPISODES_PER_CATEGORY}")
fi
if [[ "${NO_TQDM}" == "1" ]]; then
  EXTRA_ARGS+=(--no_tqdm)
fi

python -u experiments/exp1_relevance_bridge/exp1_relevance_bridge.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --num_patches "${NUM_PATCHES}" \
  --patch_index "${PATCH_INDEX}" \
  --episodes_per_patch "${EPISODES_PER_PATCH}" \
  "${CATEGORY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"