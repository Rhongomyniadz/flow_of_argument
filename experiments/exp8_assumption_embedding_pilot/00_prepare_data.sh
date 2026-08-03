#!/bin/bash
#SBATCH --job-name=exp8_prepare_submit
#SBATCH --output=_log/exp8_prepare_submit_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=.

# Slurm stage 00: normalize episode shards, then create the show-disjoint pilot split.
set -euo pipefail

MODE="${MODE:-submit}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL="${LOCAL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPU_PARTITION="${CPU_PARTITION:-cpu}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-32}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-250}"
INPUT_DIR="${INPUT_DIR:-data/conversation_moves_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/shared_data}"
DEVELOPMENT_LIMIT="${DEVELOPMENT_LIMIT:-10000}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-25}"
SEED="${SEED:-42}"
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then
  REPO_ROOT="${PWD}"
  SCRIPT_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"
elif [[ -d "${PWD}/common" && -f "${PWD}/00_prepare_data.sh" ]]; then
  SCRIPT_DIR="${PWD}"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SELF="${SCRIPT_DIR}/00_prepare_data.sh"
cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/_log"

extra_args=()
[[ -n "${SHOW_MAP:-}" ]] && extra_args+=(--show-map "${SHOW_MAP}")
[[ "${ALLOW_EPISODE_FALLBACK:-0}" == "1" ]] && extra_args+=(--allow-episode-fallback)
base_args=(--input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" --episodes-per-task "${EPISODES_PER_TASK}" --development-limit "${DEVELOPMENT_LIMIT}" --candidate-count "${CANDIDATE_COUNT}" --seed "${SEED}" "${extra_args[@]}")

if [[ "${MODE}" == "worker" ]]; then
  "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.prepare.build_pilot_dataset --mode worker "${base_args[@]}" --patch-index "${SLURM_ARRAY_TASK_ID:-${PATCH_INDEX:-0}}" --num-patches "${NUM_PATCHES:?NUM_PATCHES is required}"
  exit
fi
if [[ "${MODE}" == "merge" ]]; then
  "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.prepare.build_pilot_dataset --mode merge "${base_args[@]}" --num-patches "${NUM_PATCHES:?NUM_PATCHES is required}"
  exit
fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit, worker, or merge" >&2; exit 2; }

count_output="$("${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.prepare.build_pilot_dataset --mode count "${base_args[@]}")"
num_patches="$(awk -F= '$1=="NUM_PATCHES" {print $2}' <<<"${count_output}")"
[[ "${num_patches}" =~ ^[1-9][0-9]*$ ]] || { echo "No preparation patches found" >&2; exit 1; }
echo "${count_output}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${LOCAL}" == "1" ]]; then
  NUM_PATCHES="${num_patches}" PATCH_INDEX="${PATCH_INDEX:-0}" MODE=worker bash "${SELF}"
  echo "LOCAL_PATCH_COMPLETE=${PATCH_INDEX:-0}"
  exit
fi
dependency=()
[[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
worker_command=(sbatch --parsable --job-name=exp8_prepare --partition="${CPU_PARTITION}" --time=05:45:00 --cpus-per-task=4 --mem=16G --array="0-$((num_patches-1))%${MAX_CONCURRENT_TASKS}" --output="${SCRIPT_DIR}/_log/prepare_%A_%a.out" "${dependency[@]}" --export="ALL,MODE=worker,NUM_PATCHES=${num_patches}" "${SELF}")
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN_COMMAND='; printf '%q ' "${worker_command[@]}"; echo
  echo "DRY_RUN_COMMAND=sbatch --dependency=afterok:<array_job_id> --time=01:00:00 --export=ALL,MODE=merge,NUM_PATCHES=${num_patches} ${SELF}"
  echo "FINAL_JOB_ID=DRY_RUN_merge"
  exit
fi
array_job_id="$("${worker_command[@]}")"
merge_job_id="$(sbatch --parsable --job-name=exp8_prepare_merge --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=4 --mem=16G --output="${SCRIPT_DIR}/_log/prepare_merge_%j.out" --dependency="afterok:${array_job_id}" --export="ALL,MODE=merge,NUM_PATCHES=${num_patches}" "${SELF}")"
echo "ARRAY_JOB_ID=${array_job_id}"
echo "MERGE_JOB_ID=${merge_job_id}"
echo "FINAL_JOB_ID=${merge_job_id}"
