#!/bin/bash
#SBATCH --job-name=exp8_exp01
#SBATCH --output=_log/exp8_exp01_%j.out
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=.

# Slurm stage 02: audit the supplied Exp1 pair-level CSV in a single CPU job.
set -euo pipefail

if [[ -z "${MODE+x}" && -n "${SLURM_JOB_ID:-}" ]]; then MODE=worker; else MODE="${MODE:-submit}"; fi
DRY_RUN="${DRY_RUN:-0}"; LOCAL="${LOCAL:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; CPU_PARTITION="${CPU_PARTITION:-cpu}"
PAIRS_CSV="${PAIRS_CSV:-}"; OUTPUT_DIR="${OUTPUT_DIR:-experiments/exp8_assumption_embedding_pilot/exp01_results}"; SEED="${SEED:-42}"; BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-1000}"
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then REPO_ROOT="${PWD}"; SCRIPT_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"; elif [[ -d "${PWD}/common" && -f "${PWD}/02_run_exp01_audit.sh" ]]; then SCRIPT_DIR="${PWD}"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; fi
SELF="${SCRIPT_DIR}/02_run_exp01_audit.sh"
cd "${REPO_ROOT}"; mkdir -p "${SCRIPT_DIR}/_log"
run_stage() { [[ -n "${PAIRS_CSV}" ]] || { echo "Set PAIRS_CSV to the existing Exp1 pair table" >&2; exit 2; }; "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.exp01_existing_result_audit.run --pairs-csv "${PAIRS_CSV}" --output-dir "${OUTPUT_DIR}" --seed "${SEED}" --bootstrap-draws "${BOOTSTRAP_DRAWS}"; }
if [[ "${MODE}" == "worker" || "${LOCAL}" == "1" ]]; then run_stage; exit; fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit or worker" >&2; exit 2; }
dependency=(); [[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
command=(sbatch --parsable --job-name=exp8_exp01 --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=4 --mem=16G --output="${SCRIPT_DIR}/_log/exp01_%j.out" "${dependency[@]}" --export=ALL,MODE=worker "${SELF}")
echo "OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then printf 'DRY_RUN_COMMAND='; printf '%q ' "${command[@]}"; echo; echo "FINAL_JOB_ID=DRY_RUN_exp01"; exit; fi
job_id="$("${command[@]}")"; echo "FINAL_JOB_ID=${job_id}"
