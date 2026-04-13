#!/bin/bash

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"
EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-100}"
if (( EPISODES_PER_PATCH < 1 )); then
  echo "EPISODES_PER_PATCH must be >= 1, got ${EPISODES_PER_PATCH}" >&2
  exit 1
fi

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results}"
NO_TQDM="${NO_TQDM:-1}"

EPISODE_AND_POOL_INFO="$(
python - <<'PY'
from pathlib import Path
import json
import os
import tempfile

input_dir = Path(os.environ.get("INPUT_DIR", "data/conversation_moves_labeled"))
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
selected_files = []
for category in selected:
    category_files = sorted((input_dir / category).glob("*.json"))
    if max_per_category is not None:
        category_files = category_files[:max_per_category]
    count += len(category_files)
    selected_files.extend((category, path) for path in category_files)

fd, pool_path = tempfile.mkstemp(prefix="exp1_global_assumption_pool_", suffix=".jsonl", dir="/tmp")
os.close(fd)

with open(pool_path, "w", encoding="utf-8") as handle:
    for _, path in selected_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        turns = payload if isinstance(payload, list) else payload.get("turns", [])
        def item_texts(items):
            values = []
            for item in items or []:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                else:
                    text = str(item or "").strip()
                if text:
                    values.append(text)
            return values

        def item_text(items):
            return " ".join(item_texts(items)).strip()

        substantive_turns = [
            turn for turn in turns if str(turn.get("turn_type_label") or "").strip() == "Substantive"
        ]
        substantive_turns.sort(
            key=lambda turn: float(
                turn.get(
                    "start_time",
                    turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))),
                )
                or 0.0
            )
        )
        for previous_turn, current_turn in zip(substantive_turns, substantive_turns[1:]):
            a_text = item_text(previous_turn.get("explicit_propositions")) or str(previous_turn.get("turn_text") or "").strip()
            b_claim = item_text(current_turn.get("explicit_propositions"))
            if not a_text or not b_claim:
                continue
            episode_id = str(current_turn.get("episode_id") or path.stem)
            turn_b_idx = int(current_turn.get("turn_idx", -1))
            assumption_texts = item_texts(current_turn.get("assumptions"))
            for assumption_idx, assumption_text in enumerate(assumption_texts):
                handle.write(
                    json.dumps(
                        {
                            "assumption_id": f"{episode_id}:{turn_b_idx}:{assumption_idx}",
                            "assumption_text": assumption_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

print(f"{count}\t{pool_path}")
PY
)"

TOTAL_EPISODES="${EPISODE_AND_POOL_INFO%%$'\t'*}"
GLOBAL_ASSUMPTION_POOL_PATH="${EPISODE_AND_POOL_INFO#*$'\t'}"

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
  "GLOBAL_ASSUMPTION_POOL_PATH=${GLOBAL_ASSUMPTION_POOL_PATH}"
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
