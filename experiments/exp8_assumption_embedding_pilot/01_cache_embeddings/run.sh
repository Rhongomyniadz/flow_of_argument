#!/bin/bash
#SBATCH --job-name=exp8_embed_submit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=.

# Slurm stage 01: embed complete episodes independently, then validate/cache the index.
set -euo pipefail

MODE="${MODE:-submit}"; DRY_RUN="${DRY_RUN:-0}"; LOCAL="${LOCAL:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_PARTITION="${GPU_PARTITION:-gpu}"; CPU_PARTITION="${CPU_PARTITION:-cpu}"; MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-8}"; EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
DATA_DIR="${DATA_DIR:-experiments/exp8_assumption_embedding_pilot/shared_data}"; OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/shared_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"; MODEL_REVISION="${MODEL_REVISION:-main}"; EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-sentence_transformer}"
BATCH_SIZE="${BATCH_SIZE:-32}"; HASH_DIM="${HASH_DIM:-64}"
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then REPO_ROOT="${PWD}"; PACKAGE_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"; elif [[ -f "${PWD}/run.py" && -f "${PWD}/run.sh" ]]; then STAGE_DIR="${PWD}"; PACKAGE_DIR="$(cd "${STAGE_DIR}/.." && pwd)"; REPO_ROOT="$(cd "${PACKAGE_DIR}/../.." && pwd)"; else STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PACKAGE_DIR="$(cd "${STAGE_DIR}/.." && pwd)"; REPO_ROOT="$(cd "${PACKAGE_DIR}/../.." && pwd)"; fi
STAGE_DIR="${PACKAGE_DIR}/01_cache_embeddings"; SELF="${STAGE_DIR}/run.sh"
cd "${REPO_ROOT}"; mkdir -p "${PACKAGE_DIR}/_log"
base_args=(--data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --episodes-per-task "${EPISODES_PER_TASK}" --model-name "${MODEL_NAME}" --model-revision "${MODEL_REVISION}" --backend "${EMBEDDING_BACKEND}" --batch-size "${BATCH_SIZE}" --hash-dim "${HASH_DIM}")
[[ -n "${DEVICE:-}" ]] && base_args+=(--device "${DEVICE}")

if [[ "${MODE}" == "worker" ]]; then
  "${PYTHON_BIN}" "${STAGE_DIR}/run.py" --mode worker "${base_args[@]}" --patch-index "${SLURM_ARRAY_TASK_ID:-${PATCH_INDEX:-0}}" --num-patches "${NUM_PATCHES:?NUM_PATCHES is required}"
  exit
fi
if [[ "${MODE}" == "merge" ]]; then
  "${PYTHON_BIN}" "${STAGE_DIR}/run.py" --mode merge "${base_args[@]}" --num-patches "${NUM_PATCHES:?NUM_PATCHES is required}"
  exit
fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit, worker, or merge" >&2; exit 2; }
count_output="$("${PYTHON_BIN}" "${STAGE_DIR}/run.py" --mode count "${base_args[@]}")"
num_patches="$(awk -F= '$1=="NUM_PATCHES" {print $2}' <<<"${count_output}")"; [[ "${num_patches}" =~ ^[1-9][0-9]*$ ]] || { echo "No embedding patches found" >&2; exit 1; }
echo "${count_output}"; echo "OUTPUT_DIR=${OUTPUT_DIR}"; echo "MODEL_REVISION=${MODEL_REVISION}"
if [[ "${LOCAL}" == "1" ]]; then NUM_PATCHES="${num_patches}" PATCH_INDEX="${PATCH_INDEX:-0}" MODE=worker bash "${SELF}"; echo "LOCAL_PATCH_COMPLETE=${PATCH_INDEX:-0}"; exit; fi
dependency=(); [[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
gres=(--gres=gpu:A6000:1); [[ "${EMBEDDING_BACKEND}" == "hash" ]] && gres=()
worker_command=(sbatch --parsable --job-name=exp8_embed --partition="${GPU_PARTITION}" --time=05:45:00 --cpus-per-task=4 --mem=64G "${gres[@]}" --array="0-$((num_patches-1))%${MAX_CONCURRENT_TASKS}" --output="${PACKAGE_DIR}/_log/embed_%A_%a.out" "${dependency[@]}" --export="ALL,MODE=worker,NUM_PATCHES=${num_patches}" "${SELF}")
if [[ "${DRY_RUN}" == "1" ]]; then printf 'DRY_RUN_COMMAND='; printf '%q ' "${worker_command[@]}"; echo; echo "DRY_RUN_COMMAND=sbatch --dependency=afterok:<array_job_id> --time=01:00:00 --export=ALL,MODE=merge,NUM_PATCHES=${num_patches} ${SELF}"; echo "FINAL_JOB_ID=DRY_RUN_merge"; exit; fi
array_job_id="$("${worker_command[@]}")"
merge_job_id="$(sbatch --parsable --job-name=exp8_embed_merge --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=4 --mem=16G --output="${PACKAGE_DIR}/_log/embed_merge_%j.out" --dependency="afterok:${array_job_id}" --export="ALL,MODE=merge,NUM_PATCHES=${num_patches}" "${SELF}")"
echo "ARRAY_JOB_ID=${array_job_id}"; echo "MERGE_JOB_ID=${merge_job_id}"; echo "FINAL_JOB_ID=${merge_job_id}"
