#!/bin/bash
#SBATCH --job-name=exp8_pilot_summary
#SBATCH --output=_log/exp8_pilot_summary_%j.out
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=.

# Slurm stage 09: collect all available experiment summaries into JSON and Markdown.
set -euo pipefail

if [[ -z "${MODE+x}" && -n "${SLURM_JOB_ID:-}" ]]; then MODE=worker; else MODE="${MODE:-submit}"; fi
DRY_RUN="${DRY_RUN:-0}"; LOCAL="${LOCAL:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; CPU_PARTITION="${CPU_PARTITION:-cpu}"
if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then REPO_ROOT="${PWD}"; SCRIPT_DIR="${REPO_ROOT}/experiments/exp8_assumption_embedding_pilot"; elif [[ -d "${PWD}/common" && -f "${PWD}/09_summarize_pilot.sh" ]]; then SCRIPT_DIR="${PWD}"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; fi
SELF="${SCRIPT_DIR}/09_summarize_pilot.sh"; cd "${REPO_ROOT}"; mkdir -p "${SCRIPT_DIR}/_log"; ROOT_DIR="${ROOT_DIR:-experiments/exp8_assumption_embedding_pilot}"
run_stage() { "${PYTHON_BIN}" -m experiments.exp8_assumption_embedding_pilot.summarize_pilot --root "${ROOT_DIR}"; }
if [[ "${MODE}" == "worker" || "${LOCAL}" == "1" ]]; then run_stage; exit; fi
[[ "${MODE}" == "submit" ]] || { echo "MODE must be submit or worker" >&2; exit 2; }
dependency=(); [[ -n "${UPSTREAM_JOB_ID:-}" ]] && dependency+=(--dependency="afterok:${UPSTREAM_JOB_ID}")
command=(sbatch --parsable --job-name=exp8_pilot_summary --partition="${CPU_PARTITION}" --time=01:00:00 --cpus-per-task=2 --mem=8G --output="${SCRIPT_DIR}/_log/pilot_summary_%j.out" "${dependency[@]}" --export=ALL,MODE=worker "${SELF}")
echo "OUTPUT_JSON=${ROOT_DIR}/pilot_summary.json"; echo "OUTPUT_MARKDOWN=${ROOT_DIR}/pilot_summary.md"
if [[ "${DRY_RUN}" == "1" ]]; then printf 'DRY_RUN_COMMAND='; printf '%q ' "${command[@]}"; echo; echo "FINAL_JOB_ID=DRY_RUN_pilot_summary"; exit; fi
job_id="$("${command[@]}")"; echo "FINAL_JOB_ID=${job_id}"
