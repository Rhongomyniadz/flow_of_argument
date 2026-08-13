# Flow of Argument: ACL Submission Repository

## Overview
This repository contains the data processing pipelines, experiment scripts, and paper-facing outputs for analyzing conversational assumptions, implicature flow, and conversational enforcement dynamics. It is organized as a lean ACL submission repository: code, minimal manifests/examples, and experiment summary plots/results for review, while bulk data and large generated intermediates stay local.

## Repository Map
- `data_processing/`: data export and labeling utilities (SPORC export, labelers).
- `*_pipeline/`: Slurm pipeline entrypoints by category (news, sports, etc.).
- `experiments/exp1_relevance_bridge/`: v2 relevance-bridge analysis with global whitening, hard-negative bridge lift, falsification baseline, and patch-safe merge outputs (GPU/Slurm patch array).
- `experiments/exp2_iceberg/`: stance/assumption "iceberg" analysis (direct script).
- `experiments/exp4_implicature_flow/`: implicature flow analysis and global report (direct script).
- `experiments/exp5_processing_load/`: processing load and response delay analysis (GPU/Slurm patch array).
- `experiments/exp6_quantity_repair_cascades/`: quantity violations and repair cascades (direct script).
- `experiments/exp7_social_power/`: social power and enforcement dynamics (direct script).
- `iclr/rq1_timing_analysis/`: self-contained timing-aware robustness analysis for RQ1.
- `visualization_code/`: analysis visualizations and summary plots.
- `raw/`: exported raw episode JSONs (generated locally, not tracked).
- `results/`: non-experiment bulk outputs kept local in the slim submission version.
- `subset_extraction.sh`, `test.py`: legacy/internal utilities.

## Environment Setup
- Python: `>=3.11`
- Install base analysis dependencies:
  - `pip install -e .`
- Install optional model/data dependencies when running LLM/embedding workflows:
  - `pip install -e .[llm]`

## Data Sources and Prerequisites
- The data export uses the SPORC dataset via `sporc` and expects access to a local SPORC directory or auth token.
- Labelers rely on GPU-backed inference and `vllm` for batch labeling.
- Many experiments operate on labeled data under `data/*` and `data/*_labeled`.
- The ACL submission repo keeps only minimal manifests/examples in git; bulk labeled data and generated intermediates are omitted from version control.

## End-to-End Pipeline Order
1. Export SPORC turns into `raw/`:
   - `python data_processing/export_sporc_turns_by_category.py`
2. Run labelers (examples):
   - `bash stance_labeler_pipeline/pipeline_news.sh`
   - `bash turn_type_labeler_pipeline/pipeline_news.sh`
   - `bash conversation_moves_labeler_pipeline/pipeline_news.sh`
   - `bash maxim_violation_labeler_pipeline/pipeline_news.sh`
   - `bash entailment_labeler_pipeline/pipeline_news.sh`
3. Run experiments:
   - `exp1` and `exp5` via Slurm patch arrays
   - `exp2`, `exp4`, `exp6`, `exp7` via direct scripts
4. Inspect committed outputs under `experiments/*/results` and `iclr/*/results`.

## Experiment Index
| Experiment | Input directory | Main script | Primary outputs | GPU/Slurm? |
| --- | --- | --- | --- | --- |
| Exp 1: Relevance Bridge | `data/conversation_moves_labeled` | `experiments/exp1_relevance_bridge/exp1_relevance_bridge.py` | `exp1_summary.json`, `exp1_bridge_by_category.csv`, `exp1_bridge_by_move.csv`, `exp1_bridge_lift_by_category.png`, `exp1_ablation_by_category.png` | Yes (Slurm array) |
| Exp 2: Iceberg | `data/stance_labeled/1024` | `experiments/exp2_iceberg/exp2_iceberg.py` | `exp2_summary.json`, `exp2_regression_coefficients.csv`, `exp2_local_relationship.png` | No |
| Exp 4: Implicature Flow | `data/implicature_flow/entailment_pairs_1to10` | `experiments/exp4_implicature_flow/exp4_implicature.py` | `implicature_flow_global_report.json`, `global_implicature_flow_summary.png` | No |
| Exp 5: Processing Load | `data/conversation_moves_labeled` | `experiments/exp5_processing_load/exp5.py` | `exp5_summary.json`, `exp5_logit_coefficients.csv`, `exp5_probability_curves.png` | Yes (Slurm array) |
| Exp 6: Quantity Repair Cascades | `data/conversation_moves_labeled` | `experiments/exp6_quantity_repair_cascades/exp6.py` | `exp6_summary.json`, `exp6_cascade_events.csv`, `exp6_event_cascades.png` | No |
| Exp 7: Social Power | `data/maxim_violations_labeled` | `experiments/exp7_social_power/exp7.py` | `exp7_summary.json`, `exp7_model_coefficients.csv`, `exp7_status_shield_plot.png` | No |
| ICLR RQ1 Timing | `data/stance_labeled/1024` | `iclr/rq1_timing_analysis/rq1_timing_analysis.py` | `rq1_timing_summary.json`, `rq1_timing_stance_comparison.csv`, `rq1_timing_comparison.png` | No |

## Running Locally
- Direct scripts (exp2/exp4/exp6/exp7):
  - `python experiments/exp2_iceberg/exp2_iceberg.py`
  - `python experiments/exp4_implicature_flow/exp4_implicature.py`
  - `python experiments/exp6_quantity_repair_cascades/exp6.py`
  - `python experiments/exp7_social_power/exp7.py`
  - `python iclr/rq1_timing_analysis/rq1_timing_analysis.py`

## Running on Slurm
- Exp 1 patch array submission:
  - `bash experiments/exp1_relevance_bridge/run_exp1.sh`
- Exp 5 patch array submission:
  - `bash experiments/exp5_processing_load/run_exp5.sh`

These scripts compute patch counts, create required environment variables, prepare the shared Exp 1 whitening artifact when needed, and submit Slurm array jobs.

## Results and Artifact Policy
- Tracked/canonical: source code, `README.md`, `pyproject.toml`, minimal `data/` manifests/examples, and experiment-facing summaries/plots under `experiments/*/results`.
- Generated/local: `raw/`, `_log/`, top-level `results/`, patch shards under `experiments/*/results/*/patches/`, row-level intermediates such as `exp1_bridge_pairs.csv` and `exp5_turn_level_features.csv`, model-scoped caches, and per-episode `exp2` plot trees.
- What is already in git: the slim submission keeps experiment summary plots and main summary tables, but omits bulk data and large generated intermediates needed only for full reruns.

Exp 1 v2 operationalizes the bridge hypothesis as a ranking problem instead of a raw cosine delta. It fits one global whitening artifact over the canonical text pool, scores adjacent `Substantive -> Substantive` pairs against fixed hard negatives, and reports `bridge_lift = win_rate_context - win_rate_claim` on retained headline-constructive pairs. The canonical falsification baseline is `win_rate_random_context`, built from random assumptions with a fixed backoff hierarchy, while raw cosine remains diagnostic-only in `legacy_diagnostics`. Patch runs must read the same whitening artifact, and merge only combines shards that share the same whitening manifest hash.

## Citation
TBD

## Limitations
- Full regeneration of all results requires GPU resources and access to SPORC data.
- Some pipelines assume Slurm and may need adaptation for local-only environments.
- Full reruns require local access to omitted bulk data artifacts in addition to the tracked code and summary outputs.
