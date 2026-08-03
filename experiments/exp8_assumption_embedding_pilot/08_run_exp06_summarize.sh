#!/bin/bash
#SBATCH --job-name=exp8_exp06_summary
#SBATCH --output=_log/exp8_exp06_summary_%j.out
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=.

# Slurm stage 08: validate and summarize the manually completed audit CSV.
set -euo pipefail

if [[ -z "${MODE+x}" && -n "${SLURM_JOB_ID:-}" ]]; then MODE=worker; else MODE="${MODE:-submit}"; fi
DRY_RUN="${DRY_RUN:-0}"; LOCAL="${LOCAL:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; CPU_PARTITION="${CPU_PARTITION:-cpu}"; OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/exp06_results}"; AUDIT_CSV="${AUDIT_CSV:-${OUTPUT_DIR}/audit_sample.csv}"
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then REPO_ROOT="${PWD}"; SCRIPT_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"; elif [[ -d "${PWD}/common" && -f "${PWD}/08_run_exp06_summarize.sh" ]]; then SCRIPT_DIR="${PWD}"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; fi
SELF="${SCRIPT_DIR}/08_run_exp06_summarize.sh"; cd "${REPO_ROOT}"; mkdir -p "${SCRIPT_DIR}/_log"
run_stage() { "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.exp06_human_audit.summarize --audit-csv "${AUDIT_CSV}" --output-dir "${OUTPUT_DIR}"; }
if [[ "${MODE}" == "worker" || "${LOCAL}" == "1" ]]; then run_stage; exit; fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit or worker" >&2; exit 2; }
dependency=(); [[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
command=(sbatch --parsable --job-name=exp8_exp06_summary --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=4 --mem=16G --output="${SCRIPT_DIR}/_log/exp06_summary_%j.out" "${dependency[@]}" --export=ALL,MODE=worker "${SELF}")
echo "OUTPUT_DIR=${OUTPUT_DIR}"; echo "AUDIT_CSV=${AUDIT_CSV}"
if [[ "${DRY_RUN}" == "1" ]]; then printf 'DRY_RUN_COMMAND='; printf '%q ' "${command[@]}"; echo; echo "FINAL_JOB_ID=DRY_RUN_exp06_summary"; exit; fi
job_id="$("${command[@]}")"; echo "FINAL_JOB_ID=${job_id}"
