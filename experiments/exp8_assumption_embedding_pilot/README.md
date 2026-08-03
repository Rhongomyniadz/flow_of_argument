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

Run commands from the repository root. Each stage is self-contained and does not require installing this repository as a Python package. The cluster Python environment must provide the dependencies listed in `pyproject.toml`; GPU stages additionally require the `llm` dependencies.

CPU stages use ordinary sequential Python loops and emit only `tqdm` progress bars. Slurm sharding is limited to the two GPU stages.

## Stage 00: prepare data locally

```bash
python experiments/exp8_assumption_embedding_pilot/00_prepare_data/run.py \
  --input-dir data/conversation_moves_labeled
```

The labeled corpus has no reliable show identifier, so Stage 00 uses `episode_id` as the split group and records `split_grouping: episode_id` in `shared_data/summary.json`.

## Stage 01: cache embeddings on Slurm

```bash
sbatch experiments/exp8_assumption_embedding_pilot/01_cache_embeddings/run.sh
```

This directly submits a `05:45:00` A6000 array with 404 tasks for the current 20,189 episodes. After the array finishes, run the CPU merge locally:

```bash
python experiments/exp8_assumption_embedding_pilot/01_cache_embeddings/run.py \
  --mode merge \
  --num-patches 404
```

If the episode count changes, calculate the required array size and override both the array and patch count:

```bash
EPISODES=$(python -c 'import json; print(json.load(open("experiments/exp8_assumption_embedding_pilot/shared_data/summary.json"))["episode_count"])')
PATCHES=$(( (EPISODES + 49) / 50 ))
sbatch --array=0-$((PATCHES - 1))%8 --export=ALL,NUM_PATCHES=$PATCHES \
  experiments/exp8_assumption_embedding_pilot/01_cache_embeddings/run.sh
```

## Stage 02: audit the existing Exp1 table locally

```bash
python experiments/exp8_assumption_embedding_pilot/02_exp01_audit/run.py \
  --pairs-csv /path/to/exp1_pairs.csv
```

Required columns are `episode_id`, `reciprocal_rank_without_assumptions`, `reciprocal_rank_with_assumptions`, `top1_without_assumptions`, and `top1_with_assumptions`.

## Stage 03: frozen retrieval locally

Run after the Stage 01 merge completes:

```bash
python experiments/exp8_assumption_embedding_pilot/03_exp02_retrieval/run.py
```

## Stage 04: residual models locally

```bash
python experiments/exp8_assumption_embedding_pilot/04_exp03_residual/run.py
```

The script runs the baseline, full, and shuffled-assumption conditions sequentially.

## Stage 05: counterfactual controls locally

```bash
python experiments/exp8_assumption_embedding_pilot/05_exp04_controls/run.py
```

The script evaluates same-episode, same-category, and explicit-matched controls sequentially.

## Stage 06: fusion training on Slurm

```bash
sbatch experiments/exp8_assumption_embedding_pilot/06_exp05_fusion/run.sh
```

This directly submits history/full/shuffled crossed with seeds 42, 43, and 44 as nine A6000 tasks. After the array finishes, run the CPU merge locally:

```bash
python experiments/exp8_assumption_embedding_pilot/06_exp05_fusion/run.py \
  --mode merge \
  --num-patches 9
```

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

- `--seed` defaults to 42.
- `--data-dir`, `--cache-dir`, and `--output-dir` override default artifact paths.

## Validation

```bash
python experiments/exp8_assumption_embedding_pilot/tests/smoke_pipeline.py
python -m unittest discover -s experiments/exp8_assumption_embedding_pilot/tests -t . -v
```
