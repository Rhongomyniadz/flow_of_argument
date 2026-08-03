#!/bin/bash
#SBATCH --job-name=exp8_exp05_submit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=.

# Slurm stage 06: three fusion conditions x seeds 42/43/44 (nine GPU tasks).
set -euo pipefail

MODE="${MODE:-submit}"; DRY_RUN="${DRY_RUN:-0}"; LOCAL="${LOCAL:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; GPU_PARTITION="${GPU_PARTITION:-gpu}"; CPU_PARTITION="${CPU_PARTITION:-cpu}"; MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-4}"
DATA_DIR="${DATA_DIR:-experiments/exp8_assumption_embedding_pilot/shared_data}"; CACHE_DIR="${CACHE_DIR:-experiments/exp8_assumption_embedding_pilot/shared_cache}"; OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/exp05_results}"; FEATURE_DIM="${FEATURE_DIM:-256}"; HIDDEN_DIM="${HIDDEN_DIM:-256}"; MAX_TRAIN_ANCHORS="${MAX_TRAIN_ANCHORS:-50000}"; MAX_EPOCHS="${MAX_EPOCHS:-10}"; PATIENCE="${PATIENCE:-2}"; BATCH_SIZE="${BATCH_SIZE:-512}"; LEARNING_RATE="${LEARNING_RATE:-0.0002}"; SMOKE="${SMOKE:-0}"; NUM_PATCHES=9
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then REPO_ROOT="${PWD}"; PACKAGE_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"; elif [[ -f "${PWD}/run.py" && -f "${PWD}/run.sh" ]]; then STAGE_DIR="${PWD}"; PACKAGE_DIR="$(cd "${STAGE_DIR}/.." && pwd)"; REPO_ROOT="$(cd "${PACKAGE_DIR}/../.." && pwd)"; else STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PACKAGE_DIR="$(cd "${STAGE_DIR}/.." && pwd)"; REPO_ROOT="$(cd "${PACKAGE_DIR}/../.." && pwd)"; fi
STAGE_DIR="${PACKAGE_DIR}/06_exp05_fusion"; SELF="${STAGE_DIR}/run.sh"; cd "${REPO_ROOT}"; mkdir -p "${PACKAGE_DIR}/_log"
args=(--data-dir "${DATA_DIR}" --cache-dir "${CACHE_DIR}" --output-dir "${OUTPUT_DIR}" --feature-dim "${FEATURE_DIM}" --hidden-dim "${HIDDEN_DIM}" --max-train-anchors "${MAX_TRAIN_ANCHORS}" --max-epochs "${MAX_EPOCHS}" --patience "${PATIENCE}" --batch-size "${BATCH_SIZE}" --learning-rate "${LEARNING_RATE}" --num-patches 9)
[[ "${SMOKE}" == "1" ]] && args+=(--smoke); [[ -n "${DEVICE:-}" ]] && args+=(--device "${DEVICE}")
if [[ "${MODE}" == "worker" ]]; then "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.06_exp05_fusion.run --mode worker "${args[@]}" --patch-index "${SLURM_ARRAY_TASK_ID:-${PATCH_INDEX:-0}}"; exit; fi
if [[ "${MODE}" == "merge" ]]; then "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.06_exp05_fusion.run --mode merge "${args[@]}"; exit; fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit, worker, or merge" >&2; exit 2; }; echo "NUM_PATCHES=9"; echo "OUTPUT_DIR=${OUTPUT_DIR}"; echo "TASK_MAPPING=history:{42,43,44},full:{42,43,44},shuffled:{42,43,44}"
if [[ "${LOCAL}" == "1" ]]; then PATCH_INDEX="${PATCH_INDEX:-0}" MODE=worker bash "${SELF}"; echo "LOCAL_PATCH_COMPLETE=${PATCH_INDEX:-0}"; exit; fi
dependency=(); [[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
gres=(--gres=gpu:A6000:1); [[ "${SMOKE}" == "1" ]] && gres=()
worker_command=(sbatch --parsable --job-name=exp8_exp05 --partition="${GPU_PARTITION}" --time=05:45:00 --cpus-per-task=4 --mem=64G "${gres[@]}" --array="0-8%${MAX_CONCURRENT_TASKS}" --output="${PACKAGE_DIR}/_log/exp05_%A_%a.out" "${dependency[@]}" --export=ALL,MODE=worker "${SELF}")
if [[ "${DRY_RUN}" == "1" ]]; then printf 'DRY_RUN_COMMAND='; printf '%q ' "${worker_command[@]}"; echo; echo "DRY_RUN_COMMAND=sbatch --dependency=afterok:<array_job_id> --time=01:00:00 --export=ALL,MODE=merge ${SELF}"; echo "FINAL_JOB_ID=DRY_RUN_merge"; exit; fi
array_job_id="$("${worker_command[@]}")"; merge_job_id="$(sbatch --parsable --job-name=exp8_exp05_merge --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=4 --mem=16G --output="${PACKAGE_DIR}/_log/exp05_merge_%j.out" --dependency="afterok:${array_job_id}" --export=ALL,MODE=merge "${SELF}")"
echo "ARRAY_JOB_ID=${array_job_id}"; echo "MERGE_JOB_ID=${merge_job_id}"; echo "FINAL_JOB_ID=${merge_job_id}"
