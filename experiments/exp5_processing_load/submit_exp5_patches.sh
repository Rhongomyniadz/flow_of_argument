#!/bin/bash

set -euo pipefail

EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-100}"
if (( EPISODES_PER_PATCH < 1 )); then
  echo "EPISODES_PER_PATCH must be >= 1, got ${EPISODES_PER_PATCH}" >&2
  exit 1
fi

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp5_processing_load/results}"
SILENCE_GAP_QUANTILE="${SILENCE_GAP_QUANTILE:-0.95}"
MIN_SILENCE_GAP="${MIN_SILENCE_GAP:-5.0}"
NO_TQDM="${NO_TQDM:-1}"
INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"

TOTAL_EPISODES="$(
python - <<'PY'
from pathlib import Path
import os

input_dir = Path(os.environ.get("INPUT_DIR", "data/conversation_moves_labeled"))
paths = sorted(input_dir.glob("*.json")) or sorted(input_dir.glob("*/*.json"))
print(len(paths))
PY
)"

if (( TOTAL_EPISODES < 1 )); then
  echo "No episodes matched the requested Exp 5 input directory." >&2
  exit 1
fi

NUM_PATCHES=$(( (TOTAL_EPISODES + EPISODES_PER_PATCH - 1) / EPISODES_PER_PATCH ))

SILENCE_GAP_THRESHOLD_SEC="$(
python - <<'PY'
import json
import math
import os
from pathlib import Path

import numpy as np


def to_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def gap_seconds(turn_a, turn_b):
    end_time = to_float(turn_a.get("endTime", turn_a.get("end_time")))
    start_time = to_float(turn_b.get("startTime", turn_b.get("start_time")))
    if math.isfinite(start_time) and math.isfinite(end_time):
        return start_time - end_time
    return float("nan")


input_dir = Path(os.environ.get("INPUT_DIR", "data/conversation_moves_labeled"))
silence_gap_quantile = float(os.environ.get("SILENCE_GAP_QUANTILE", "0.95"))
min_silence_gap = float(os.environ.get("MIN_SILENCE_GAP", "5.0"))

paths = sorted(input_dir.glob("*.json")) or sorted(input_dir.glob("*/*.json"))
gaps = []
for path in paths:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(payload, list) or not payload:
        continue
    turns = list(payload)
    turns.sort(
        key=lambda turn: (
            int(turn.get("turn_idx", 0))
            if str(turn.get("turn_idx", 0)).lstrip("-").isdigit()
            else 0
        )
    )
    for first_turn, second_turn in zip(turns, turns[1:]):
        gap = gap_seconds(first_turn, second_turn)
        if math.isfinite(gap) and gap >= 0.0:
            gaps.append(gap)

if gaps:
    threshold = max(min_silence_gap, float(np.quantile(np.asarray(gaps, dtype=float), silence_gap_quantile)))
else:
    threshold = min_silence_gap

print(threshold)
PY
)"

ARRAY_RANGE="0-$((NUM_PATCHES - 1))"
EXPORT_STRING="ALL,NUM_PATCHES=${NUM_PATCHES},EPISODES_PER_PATCH=${EPISODES_PER_PATCH},EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME},EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE},OUTPUT_DIR=${OUTPUT_DIR},SILENCE_GAP_QUANTILE=${SILENCE_GAP_QUANTILE},MIN_SILENCE_GAP=${MIN_SILENCE_GAP},SILENCE_GAP_THRESHOLD_SEC=${SILENCE_GAP_THRESHOLD_SEC},NO_TQDM=${NO_TQDM},INPUT_DIR=${INPUT_DIR}"

sbatch \
  --array="${ARRAY_RANGE}" \
  --export="${EXPORT_STRING}" \
  experiments/exp5_processing_load/run_exp5_patch.sh
