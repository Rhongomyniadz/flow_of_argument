#!/bin/bash
#SBATCH --job-name=exp8_submit_all
#SBATCH --output=_log/exp8_submit_all_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=.

# Submit stages 00-07 with afterok dependencies. Stages 08-09 remain manual.
set -euo pipefail

if [[ -d "${PWD}/experiments/exp8_assumption_embedding_pilot" ]]; then SCRIPT_DIR="${PWD}/experiments/exp8_assumption_embedding_pilot"; elif [[ -f "${PWD}/submit_all.sh" && -d "${PWD}/common" ]]; then SCRIPT_DIR="${PWD}"; else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
DRY_RUN="${DRY_RUN:-0}"

submit_stage() {
  local script="$1"
  local upstream="$2"
  local output
  output="$(MODE=submit DRY_RUN="${DRY_RUN}" UPSTREAM_JOB_ID="${upstream}" bash "${SCRIPT_DIR}/${script}")"
  echo "${output}" >&2
  awk -F= '$1=="FINAL_JOB_ID" {value=$2} END {print value}' <<<"${output}"
}

job=""
job="$(submit_stage 00_prepare_data.sh "${job}")"
job="$(submit_stage 01_cache_embeddings.sh "${job}")"

# Exp01 is independent of the shared cache but is kept in the declared stage order.
if [[ -n "${PAIRS_CSV:-}" ]]; then
  job="$(submit_stage 02_run_exp01_audit.sh "${job}")"
elif [[ "${SKIP_EXP01:-0}" == "1" ]]; then
  echo "Skipping stage 02 because SKIP_EXP01=1." >&2
else
  echo "Set PAIRS_CSV for stage 02, or explicitly set SKIP_EXP01=1." >&2
  exit 2
fi

job="$(submit_stage 03_run_exp02_retrieval.sh "${job}")"
job="$(submit_stage 04_run_exp03_residual.sh "${job}")"
job="$(submit_stage 05_run_exp04_controls.sh "${job}")"
job="$(submit_stage 06_run_exp05_fusion.sh "${job}")"
job="$(submit_stage 07_run_exp06_sample.sh "${job}")"

echo "FINAL_JOB_ID=${job}"
echo "MANUAL_NEXT=Annotate exp06_results/audit_sample.csv, then run 08_run_exp06_summarize.sh and 09_summarize_pilot.sh"
