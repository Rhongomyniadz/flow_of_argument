#!/bin/bash
#SBATCH --job-name=exp1_repr_diagnostic
#SBATCH --output=iclr/exp1_representation_baselines/_log/exp1_repr_diagnostic_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/shared/6/projects/flow_of_argument

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data_cleaned/conversation_moves_labeled}"
OUTPUT_ROOT="${OUTPUT_ROOT:-iclr/exp1_representation_baselines/results}"
EPISODES_PER_PATCH="${EPISODES_PER_PATCH:-5}"
MAX_EPISODES_PER_CATEGORY="${MAX_EPISODES_PER_CATEGORY-}"
CATEGORIES_CSV="${CATEGORIES_CSV:-all}"
CONDITIONS_CSV="${CONDITIONS_CSV:-raw_turn,raw_turn_with_history,raw_turn_plus_assumptions,explicit_only,explicit_plus_top3_assumptions,explicit_plus_shuffled_assumptions,explicit_plus_wrong_episode_assumptions}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_OUTPUT_NAME="${MODEL_NAME//\//__}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_OUTPUT_NAME}}"
PREPARED_PAIRS_JSONL="${PREPARED_PAIRS_JSONL:-${OUTPUT_DIR}/exp1_representation_prepared_pairs.jsonl}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/shared/4/models}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_SCORE_RETRIES="${MAX_SCORE_RETRIES:-2}"
MAX_RETRY_TOKENS="${MAX_RETRY_TOKENS:-128}"
SEED="${SEED:-42}"
HISTORY_TURNS="${HISTORY_TURNS:-3}"
SOURCE_TAIL_WORDS="${SOURCE_TAIL_WORDS:-100}"
CANDIDATE_HEAD_WORDS="${CANDIDATE_HEAD_WORDS:-100}"
ASSUMPTION_BUDGET="${ASSUMPTION_BUDGET:-3}"
BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-1000}"
NO_TQDM="${NO_TQDM:-1}"
DRY_RUN="${DRY_RUN:-0}"
STRICT_ALL_CONDITIONS="${STRICT_ALL_CONDITIONS:-0}"
OVERWRITE_SCORES="${OVERWRITE_SCORES:-0}"
AUDIT_SAMPLE_SIZE_PER_OUTCOME="${AUDIT_SAMPLE_SIZE_PER_OUTCOME:-25}"
ALLOW_FULL_RUN="${ALLOW_FULL_RUN:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

run_python() {
  python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py "$@"
}

build_common_args() {
  IFS=',' read -r -a CATEGORY_VALUES <<< "${CATEGORIES_CSV}"
  IFS=',' read -r -a CONDITION_VALUES <<< "${CONDITIONS_CSV}"
  COMMON_ARGS=(
    --input_dir "${INPUT_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --prepared_pairs_jsonl "${PREPARED_PAIRS_JSONL}"
    --categories "${CATEGORY_VALUES[@]}"
    --conditions "${CONDITION_VALUES[@]}"
    --model_name "${MODEL_NAME}"
    --download_dir "${DOWNLOAD_DIR}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
    --prompt_batch_size "${PROMPT_BATCH_SIZE}"
    --max_tokens "${MAX_TOKENS}"
    --max_score_retries "${MAX_SCORE_RETRIES}"
    --max_retry_tokens "${MAX_RETRY_TOKENS}"
    --seed "${SEED}"
    --history_turns "${HISTORY_TURNS}"
    --source_tail_words "${SOURCE_TAIL_WORDS}"
    --candidate_head_words "${CANDIDATE_HEAD_WORDS}"
    --assumption_budget "${ASSUMPTION_BUDGET}"
    --bootstrap_draws "${BOOTSTRAP_DRAWS}"
    --audit_sample_size_per_outcome "${AUDIT_SAMPLE_SIZE_PER_OUTCOME}"
  )
  if [[ -n "${MAX_EPISODES_PER_CATEGORY}" ]]; then
    COMMON_ARGS+=(--max_episodes_per_category "${MAX_EPISODES_PER_CATEGORY}")
  fi
  if [[ "${NO_TQDM}" == "1" ]]; then
    COMMON_ARGS+=(--no_tqdm)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    COMMON_ARGS+=(--dry_run)
  fi
  if [[ "${STRICT_ALL_CONDITIONS}" == "1" ]]; then
    COMMON_ARGS+=(--strict_all_conditions)
  fi
  if [[ "${OVERWRITE_SCORES}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite_scores)
  fi
}

configure_cuda() {
  if [[ "${TENSOR_PARALLEL_SIZE}" != "2" ]]; then
    echo "This runner requests two A6000 GPUs and requires TENSOR_PARALLEL_SIZE=2 for vLLM tensor parallelism." >&2
    exit 1
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a CUDA_DEVICE_VALUES <<< "${CUDA_VISIBLE_DEVICES}"
    if (( ${#CUDA_DEVICE_VALUES[@]} != 2 )); then
      echo "Expected exactly two allocated GPUs in CUDA_VISIBLE_DEVICES, found ${CUDA_VISIBLE_DEVICES}." >&2
      exit 1
    fi
  fi
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
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  FLASHINFER_JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-na}"
  export FLASHINFER_WORKSPACE_BASE="${EXP1_FLASHINFER_WORKSPACE_BASE:-/tmp/${USER}/flashinfer_exp1_repr_${FLASHINFER_JOB_KEY}}"
  mkdir -p "${FLASHINFER_WORKSPACE_BASE}"
}

prepared_episode_count() {
  python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_episode_count"])' \
    "${OUTPUT_DIR}/exp1_representation_prepare_manifest.json"
}

submit_stages() {
  if [[ -z "${MAX_EPISODES_PER_CATEGORY}" && "${ALLOW_FULL_RUN}" != "1" ]]; then
    echo "Refusing an ungated full-corpus submission. Run the five-episode diagnostic smoke test first, then review the audit and diagnostic gate before setting ALLOW_FULL_RUN=1." >&2
    exit 1
  fi
  build_common_args
  run_python "${COMMON_ARGS[@]}" --prepare_only
  TOTAL_EPISODES="$(prepared_episode_count)"
  if (( TOTAL_EPISODES < 1 )); then
    echo "Preparation selected no source episodes." >&2
    exit 1
  fi
  NUM_PATCHES="${NUM_PATCHES:-$(( (TOTAL_EPISODES + EPISODES_PER_PATCH - 1) / EPISODES_PER_PATCH ))}"
  ARRAY_RANGE="0-$((NUM_PATCHES - 1))"
  export INPUT_DIR OUTPUT_ROOT OUTPUT_DIR PREPARED_PAIRS_JSONL EPISODES_PER_PATCH NUM_PATCHES
  export MAX_EPISODES_PER_CATEGORY CATEGORIES_CSV CONDITIONS_CSV MODEL_NAME DOWNLOAD_DIR
  export TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION PROMPT_BATCH_SIZE MAX_TOKENS MAX_SCORE_RETRIES MAX_RETRY_TOKENS SEED
  export HISTORY_TURNS SOURCE_TAIL_WORDS CANDIDATE_HEAD_WORDS ASSUMPTION_BUDGET
  export BOOTSTRAP_DRAWS NO_TQDM DRY_RUN STRICT_ALL_CONDITIONS OVERWRITE_SCORES
  export AUDIT_SAMPLE_SIZE_PER_OUTCOME
  export ALLOW_FULL_RUN
  PATCH_SUBMISSION="$(sbatch --parsable --array="${ARRAY_RANGE}" --export="ALL,EXP1_BASELINE_STAGE=patch" "$0")"
  PATCH_JOB_ID="${PATCH_SUBMISSION%%;*}"
  MERGE_SUBMISSION="$(sbatch --parsable --dependency="afterok:${PATCH_JOB_ID}" --export="ALL,EXP1_BASELINE_STAGE=merge" "$0")"
  MERGE_JOB_ID="${MERGE_SUBMISSION%%;*}"
  ANALYSIS_SUBMISSION="$(sbatch --parsable --dependency="afterok:${MERGE_JOB_ID}" --export="ALL,EXP1_BASELINE_STAGE=analysis" "$0")"
  ANALYSIS_JOB_ID="${ANALYSIS_SUBMISSION%%;*}"
  echo "Prepared ${TOTAL_EPISODES} episodes and submitted ${NUM_PATCHES} scoring patches (${PATCH_JOB_ID}), merge (${MERGE_JOB_ID}), and analysis (${ANALYSIS_JOB_ID})."
}

EXP1_BASELINE_STAGE="${EXP1_BASELINE_STAGE:-submit}"
if [[ "${EXP1_BASELINE_STAGE}" == "submit" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this runner with sbatch, not bash: sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh" >&2
    exit 1
  fi
  submit_stages
  exit 0
fi

build_common_args
case "${EXP1_BASELINE_STAGE}" in
  prepare)
    run_python "${COMMON_ARGS[@]}" --prepare_only
    ;;
  patch)
    configure_cuda
    NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES for patch scoring.}"
    PATCH_INDEX="${SLURM_ARRAY_TASK_ID:?Patch scoring requires a Slurm array task.}"
    run_python "${COMMON_ARGS[@]}" \
      --score_only \
      --num_patches "${NUM_PATCHES}" \
      --episodes_per_patch "${EPISODES_PER_PATCH}" \
      --patch_index "${PATCH_INDEX}"
    ;;
  merge)
    NUM_PATCHES="${NUM_PATCHES:?Set NUM_PATCHES for patch merge.}"
    python -u iclr/exp1_representation_baselines/merge_exp1_representation_patches.py \
      "${COMMON_ARGS[@]}" \
      --num_patches "${NUM_PATCHES}" \
      --episodes_per_patch "${EPISODES_PER_PATCH}"
    ;;
  analysis)
    run_python "${COMMON_ARGS[@]}" --analysis_only
    ;;
  *)
    echo "Unknown EXP1_BASELINE_STAGE=${EXP1_BASELINE_STAGE}; expected submit, prepare, patch, merge, or analysis." >&2
    exit 1
    ;;
esac
