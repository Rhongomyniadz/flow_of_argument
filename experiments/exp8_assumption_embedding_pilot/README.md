# Exp8 assumption-embedding pilot

The pilot is split into eight independent stages. Every stage directory contains one `run.py`. Only the two GPU stages, 01 and 06, additionally contain `run.sh` for Slurm submission.

| Stage | Directory | Execution | Output |
|---:|---|---|---|
| 00 | `00_prepare_data/` | local CPU | `shared_data/` |
| 01 | `01_cache_embeddings/` | Slurm GPU array | `shared_cache/` |
| 02 | `02_exp01_audit/` | local CPU | `exp01_results/` |
| 03 | `03_exp02_retrieval/` | local CPU | `exp02_results/` |
| 04 | `04_exp03_residual/` | local CPU | `exp03_results/` |
| 05 | `05_exp04_controls/` | local CPU | `exp04_results/` |
| 06 | `06_exp05_fusion/` | Slurm GPU array | `exp05_results/` |
| 07 | `07_exp06_audit/` | local CPU/manual annotation | `exp06_results/` and pilot summary |

Run commands from the repository root. Install dependencies first:

```bash
pip install -e .
pip install -e '.[llm]'
```

Successful CPU stages emit only `tqdm` progress bars. Matching completed patches are reused automatically.

## Stage 00: prepare data locally

```bash
python experiments/exp8_assumption_embedding_pilot/00_prepare_data/run.py \
  --input-dir data/conversation_moves_labeled \
  --episodes-per-task 250 \
  --jobs 8
```

If show identity is stored separately, add `--show-map /path/to/episode_show_map.csv`.

## Stage 01: cache embeddings on Slurm

```bash
sbatch experiments/exp8_assumption_embedding_pilot/01_cache_embeddings/run.sh
```

The wrapper submits Qwen embedding workers as a `05:45:00` A6000 array and then submits the merge job with an `afterok` dependency.

## Stage 02: audit the existing Exp1 table locally

```bash
python experiments/exp8_assumption_embedding_pilot/02_exp01_audit/run.py \
  --pairs-csv /path/to/exp1_pairs.csv
```

Required columns are `episode_id`, `reciprocal_rank_without_assumptions`, `reciprocal_rank_with_assumptions`, `top1_without_assumptions`, and `top1_with_assumptions`.

## Stage 03: frozen retrieval locally

Run after the Stage 01 merge completes:

```bash
python experiments/exp8_assumption_embedding_pilot/03_exp02_retrieval/run.py \
  --anchors-per-task 1000 \
  --jobs 8
```

## Stage 04: residual models locally

```bash
python experiments/exp8_assumption_embedding_pilot/04_exp03_residual/run.py \
  --jobs 3
```

The three local tasks are the baseline, full, and shuffled-assumption conditions.

## Stage 05: counterfactual controls locally

```bash
python experiments/exp8_assumption_embedding_pilot/05_exp04_controls/run.py \
  --anchors-per-task 1000 \
  --jobs 8
```

The local tasks cross same-episode, same-category, and explicit-matched controls with anchor shards.

## Stage 06: fusion training on Slurm

```bash
sbatch experiments/exp8_assumption_embedding_pilot/06_exp05_fusion/run.sh
```

This submits history/full/shuffled crossed with seeds 42, 43, and 44 as nine A6000 tasks, followed by a CPU merge.

## Stage 07: human audit locally

Create the immutable blinded sample:

```bash
python experiments/exp8_assumption_embedding_pilot/07_exp06_audit/run.py \
  --mode sample \
  --sample-size 100
```

Annotate `exp06_results/audit_sample.csv`, then calculate agreement:

```bash
python experiments/exp8_assumption_embedding_pilot/07_exp06_audit/run.py \
  --mode summarize
```

Finally collect all experiment summaries:

```bash
python experiments/exp8_assumption_embedding_pilot/07_exp06_audit/run.py \
  --mode pilot-summary
```

## Common local options

- `--jobs N` controls local shard concurrency for Stages 00, 03, 04, and 05.
- `--force` recomputes matching completed outputs.
- `--seed` defaults to 42.
- `--data-dir`, `--cache-dir`, and `--output-dir` override default artifact paths.

## Validation

```bash
python -m experiments.exp8_assumption_embedding_pilot.tests.smoke_pipeline
python -m unittest discover -s experiments/exp8_assumption_embedding_pilot/tests -t . -v
```
