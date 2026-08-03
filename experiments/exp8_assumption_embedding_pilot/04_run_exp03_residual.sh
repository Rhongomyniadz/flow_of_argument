#!/bin/bash
#SBATCH --job-name=exp8_exp03_submit
#SBATCH --output=_log/exp8_exp03_submit_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=.

# Slurm stage 04: one CPU array task per residual-model condition.
set -euo pipefail

MODE="${MODE:-submit}"; DRY_RUN="${DRY_RUN:-0}"; LOCAL="${LOCAL:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; CPU_PARTITION="${CPU_PARTITION:-cpu}"; MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-32}"
DATA_DIR="${DATA_DIR:-experiments/exp8_assumption_embedding_pilot/shared_data}"; CACHE_DIR="${CACHE_DIR:-experiments/exp8_assumption_embedding_pilot/shared_cache}"; OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/exp03_results}"; FEATURE_DIM="${FEATURE_DIM:-256}"; MAX_TRAIN_ANCHORS="${MAX_TRAIN_ANCHORS:-50000}"; RIDGE_ALPHA="${RIDGE_ALPHA:-10}"; SEED="${SEED:-42}"; NUM_PATCHES=3
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then REPO_ROOT="${PWD}"; SCRIPT_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"; elif [[ -d "${PWD}/common" && -f "${PWD}/04_run_exp03_residual.sh" ]]; then SCRIPT_DIR="${PWD}"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; fi
SELF="${SCRIPT_DIR}/04_run_exp03_residual.sh"; cd "${REPO_ROOT}"; mkdir -p "${SCRIPT_DIR}/_log"
args=(--data-dir "${DATA_DIR}" --cache-dir "${CACHE_DIR}" --output-dir "${OUTPUT_DIR}" --feature-dim "${FEATURE_DIM}" --max-train-anchors "${MAX_TRAIN_ANCHORS}" --ridge-alpha "${RIDGE_ALPHA}" --seed "${SEED}" --num-patches 3)
if [[ "${MODE}" == "worker" ]]; then "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.exp03_linear_residual.run --mode worker "${args[@]}" --patch-index "${SLURM_ARRAY_TASK_ID:-${PATCH_INDEX:-0}}"; exit; fi
if [[ "${MODE}" == "merge" ]]; then "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.exp03_linear_residual.run --mode merge "${args[@]}"; exit; fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit, worker, or merge" >&2; exit 2; }; echo "NUM_PATCHES=3"; echo "OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${LOCAL}" == "1" ]]; then PATCH_INDEX="${PATCH_INDEX:-0}" MODE=worker bash "${SELF}"; echo "LOCAL_PATCH_COMPLETE=${PATCH_INDEX:-0}"; exit; fi
dependency=(); [[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
worker_command=(sbatch --parsable --job-name=exp8_exp03 --partition="${CPU_PARTITION}" --time=05:45:00 --cpus-per-task=4 --mem=32G --array="0-2%${MAX_CONCURRENT_TASKS}" --output="${SCRIPT_DIR}/_log/exp03_%A_%a.out" "${dependency[@]}" --export=ALL,MODE=worker "${SELF}")
if [[ "${DRY_RUN}" == "1" ]]; then printf 'DRY_RUN_COMMAND='; printf '%q ' "${worker_command[@]}"; echo; echo "DRY_RUN_COMMAND=sbatch --dependency=afterok:<array_job_id> --time=01:00:00 --export=ALL,MODE=merge ${SELF}"; echo "FINAL_JOB_ID=DRY_RUN_merge"; exit; fi
array_job_id="$("${worker_command[@]}")"; merge_job_id="$(sbatch --parsable --job-name=exp8_exp03_merge --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=4 --mem=16G --output="${SCRIPT_DIR}/_log/exp03_merge_%j.out" --dependency="afterok:${array_job_id}" --export=ALL,MODE=merge "${SELF}")"
echo "ARRAY_JOB_ID=${array_job_id}"; echo "MERGE_JOB_ID=${merge_job_id}"; echo "FINAL_JOB_ID=${merge_job_id}"
