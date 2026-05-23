#!/bin/bash
#SBATCH --job-name=exp1_llm
#SBATCH --output=_log/exp1_llm_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/shared/6/projects/flow_of_argument

set -euo pipefail

CUDA_HOME="${EXP1_CUDA_HOME:-/usr/local/cuda-12.9}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA nvcc not found at ${CUDA_HOME}/bin/nvcc. Set EXP1_CUDA_HOME to a CUDA toolkit path with nvcc." >&2
  exit 1
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${EXP1_TORCH_CUDA_ARCH_LIST:-8.6}"
export FLASHINFER_CUDA_ARCH_LIST="${EXP1_FLASHINFER_CUDA_ARCH_LIST:-8.6}"
export FLASHINFER_COMPUTE_CAPS="${EXP1_FLASHINFER_COMPUTE_CAPS:-86}"
export FLASHINFER_NVCC="${CUDA_HOME}/bin/nvcc"
FLASHINFER_JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-na}"
export FLASHINFER_WORKSPACE_BASE="${EXP1_FLASHINFER_WORKSPACE_BASE:-/tmp/${USER}/flashinfer_exp1_${FLASHINFER_JOB_KEY}}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp1_relevance_bridge/results/}"
EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-100}"
MAX_EPISODES_PER_CATEGORY="${MAX_EPISODES_PER_CATEGORY-250}"
CATEGORIES_CSV="${CATEGORIES_CSV:-all}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/shared/4/models}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-192}"
SEED="${SEED:-42}"
NO_TQDM="${NO_TQDM:-1}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

build_category_args() {
  CATEGORY_ARGS=(--categories all)
  if [[ -n "${CATEGORIES_CSV}" ]]; then
    IFS=',' read -r -a CATEGORY_VALUES <<< "${CATEGORIES_CSV}"
    CATEGORY_ARGS=(--categories "${CATEGORY_VALUES[@]}")
  fi
}

build_extra_args() {
  EXTRA_ARGS=()
  if [[ -n "${MAX_EPISODES_PER_CATEGORY}" ]]; then
    EXTRA_ARGS+=(--max_episodes_per_category "${MAX_EPISODES_PER_CATEGORY}")
  fi
  if [[ "${NO_TQDM}" == "1" ]]; then
    EXTRA_ARGS+=(--no_tqdm)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--dry_run)
  fi
}

count_total_episodes() {
  export INPUT_DIR CATEGORIES_CSV MAX_EPISODES_PER_CATEGORY
  python - <<'PY'
from pathlib import Path
import os

input_dir = Path(os.environ["INPUT_DIR"])
categories_csv = os.environ.get("CATEGORIES_CSV", "").strip()
max_raw = os.environ.get("MAX_EPISODES_PER_CATEGORY", "").strip()
max_episodes = int(max_raw) if max_raw else None
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
    if max_episodes is not None:
        category_files = category_files[:max_episodes]
    count += len(category_files)
print(count)
PY
}

resolve_selected_categories() {
  export INPUT_DIR CATEGORIES_CSV
  python - <<'PY'
from pathlib import Path
import os

input_dir = Path(os.environ["INPUT_DIR"])
categories_csv = os.environ.get("CATEGORIES_CSV", "").strip()
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
print(",".join(selected))
PY
}

submit_exp1_stages() {
  TOTAL_EPISODES="$(count_total_episodes)"
  if (( TOTAL_EPISODES < 1 )); then
    echo "No episodes matched the requested Exp 1 inputs." >&2
    exit 1
  fi

  SELECTED_CATEGORIES="$(resolve_selected_categories)"
  MAX_EPISODES_LABEL="${MAX_EPISODES_PER_CATEGORY:-all}"
  NUM_PATCHES="${NUM_PATCHES:-$(( (TOTAL_EPISODES + EPISODES_PER_PATCH - 1) / EPISODES_PER_PATCH ))}"
  ARRAY_RANGE="0-$((NUM_PATCHES - 1))"
  EXPORT_VARS=(
    "ALL"
    "INPUT_DIR=${INPUT_DIR}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "EPISODES_PER_PATCH=${EPISODES_PER_PATCH}"
    "NUM_PATCHES=${NUM_PATCHES}"
    "CATEGORIES_CSV=${CATEGORIES_CSV}"
    "MODEL_NAME=${MODEL_NAME}"
    "DOWNLOAD_DIR=${DOWNLOAD_DIR}"
    "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    "PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE}"
    "MAX_TOKENS=${MAX_TOKENS}"
    "SEED=${SEED}"
    "NO_TQDM=${NO_TQDM}"
    "DRY_RUN=${DRY_RUN}"
  )

  if [[ -n "${MAX_EPISODES_PER_CATEGORY}" ]]; then
    EXPORT_VARS+=("MAX_EPISODES_PER_CATEGORY=${MAX_EPISODES_PER_CATEGORY}")
  fi

  EXPORT_STRING="$(IFS=,; echo "${EXPORT_VARS[*]}")"
  PATCH_JOB_SUBMISSION="$(
    sbatch --parsable \
      --array="${ARRAY_RANGE}" \
      --export="${EXPORT_STRING},EXP1_STAGE=patch" \
      "$0"
  )"
  PATCH_JOB_ID="${PATCH_JOB_SUBMISSION%%;*}"
  MERGE_JOB_SUBMISSION="$(
    sbatch --parsable \
      --dependency="afterok:${PATCH_JOB_ID}" \
      --export="${EXPORT_STRING},EXP1_STAGE=merge" \
      "$0"
  )"
  MERGE_JOB_ID="${MERGE_JOB_SUBMISSION%%;*}"

  echo "Exp 1 selected categories: ${SELECTED_CATEGORIES}."
  echo "Exp 1 max episodes per category: ${MAX_EPISODES_LABEL}."
  echo "Exp 1 selected episode files: ${TOTAL_EPISODES}."
  echo "Exp 1 patch count: ${NUM_PATCHES}; array range: ${ARRAY_RANGE}."
  echo "Submitted Exp 1 LLM patch array ${PATCH_JOB_ID} with range ${ARRAY_RANGE}."
  echo "Submitted Exp 1 LLM merge job ${MERGE_JOB_ID} after ${PATCH_JOB_ID}."
}

run_exp1_python() {
  python -u experiments/exp1_relevance_bridge/exp1_relevance_bridge.py "$@"
}

EXP1_STAGE="${EXP1_STAGE:-submit}"
if [[ "${EXP1_STAGE}" == "submit" ]]; then
  submit_exp1_stages
  exit 0
fi

NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES in the sbatch environment.}"
build_category_args
build_extra_args

COMMON_ARGS=(
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  "${CATEGORY_ARGS[@]}"
  --model_name "${MODEL_NAME}"
  --download_dir "${DOWNLOAD_DIR}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --prompt_batch_size "${PROMPT_BATCH_SIZE}"
  --max_tokens "${MAX_TOKENS}"
  --seed "${SEED}"
  --num_patches "${NUM_PATCHES}"
  --episodes_per_patch "${EPISODES_PER_PATCH}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${EXP1_STAGE}" == "merge" ]]; then
  run_exp1_python "${COMMON_ARGS[@]}" --merge_patches_only
  exit 0
fi

if [[ "${EXP1_STAGE}" != "patch" ]]; then
  echo "Unknown EXP1_STAGE=${EXP1_STAGE}. Expected submit, patch, or merge." >&2
  exit 1
fi

PATCH_INDEX="${SLURM_ARRAY_TASK_ID:?EXP1_STAGE=patch must run as a Slurm array task.}"
run_exp1_python "${COMMON_ARGS[@]}" --patch_index "${PATCH_INDEX}"
