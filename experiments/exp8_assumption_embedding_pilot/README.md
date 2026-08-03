# Exp8: assumption-embedding pilot

This directory is a self-contained ICLR-oriented pilot. It does not import or write Exp1-Exp7 experiment code or result directories. Preparation and embeddings are shared; every exploratory result is isolated in `exp01_results/` through `exp06_results/`.

## Execution model

Only GPU work enters the Slurm queue:

| Shell file | Slurm work | Output |
|---|---|---|
| `01_cache_embeddings.sh` | Qwen embedding array, one A6000 per task | `shared_cache/` |
| `06_run_exp05_fusion.sh` | history/full/shuffled crossed with seeds 42/43/44 | `exp05_results/` |

All CPU-only stages run from the command line through `run_cpu_stage.py`. Its `--jobs` option runs independent shards concurrently on the local machine; after all workers succeed, it runs the merge automatically. Existing matching patches are reused.

| CLI stage | Work | Output |
|---|---|---|
| `prepare` | normalize data, create show-disjoint split and candidates | `shared_data/` |
| `exp01` | audit the supplied Exp1 pair table | `exp01_results/` |
| `exp02` | frozen retrieval shards and clustered bootstrap | `exp02_results/` |
| `exp03` | three linear-residual conditions | `exp03_results/` |
| `exp04` | control type crossed with anchor shards | `exp04_results/` |
| `exp06-sample` | create the immutable blinded audit sample | `exp06_results/` |
| `exp06-summarize` | summarize completed annotations | `exp06_results/` |
| `pilot-summary` | collect all available experiment summaries | `pilot_summary.json` and `.md` |

## Installation and inputs

Run commands from the repository root. Install standard dependencies locally; GPU nodes additionally need the optional model dependencies:

```bash
pip install -e .
pip install -e '.[llm]'
```

Production embeddings use the official `SentenceTransformer` interface for [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B). Input episode JSON must contain a show identifier such as `show_id`, `rssKey`, `podcast_id`, or `feed_id`. Otherwise provide `--show-map /path/to/episode_show_map.csv`.

Exp01 requires a CSV containing `episode_id`, `reciprocal_rank_without_assumptions`, `reciprocal_rank_with_assumptions`, `top1_without_assumptions`, and `top1_with_assumptions`.

## Recommended command sequence

### 1. Prepare data locally

```bash
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage prepare \
  --jobs 8 \
  --input-dir data/conversation_moves_labeled \
  --episodes-per-task 250
```

If show metadata is external, add `--show-map /path/to/episode_show_map.csv`. `--allow-episode-fallback` is available only for diagnostic data lacking show identity and weakens the split guarantee.

### 2. Submit GPU embedding work

```bash
sbatch experiments/exp8_assumption_embedding_pilot/01_cache_embeddings.sh
```

The submitted orchestration job counts prepared episodes, creates the `05:45:00` A6000 worker array, and submits the cache-index merge with an `afterok` dependency. Inspect its `_log/exp8_embed_submit_<job>.out` file for `FINAL_JOB_ID`.

### 3. Run Exp01-Exp04 locally

Exp01 is independent of the embedding cache:

```bash
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp01 \
  --pairs-csv /path/to/exp1_pairs.csv
```

After the embedding merge has completed:

```bash
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp02 --jobs 8
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp03 --jobs 3
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp04 --jobs 8
```

The defaults are 1,000 anchors per Exp02/Exp04 shard, 1,000 clustered-bootstrap draws, and seed 42. Override these with `--anchors-per-task`, `--bootstrap-draws`, and `--seed`.

### 4. Submit GPU fusion training

```bash
sbatch experiments/exp8_assumption_embedding_pilot/06_run_exp05_fusion.sh
```

This submits nine A6000 jobs with a default `%4` concurrency limit and then a CPU merge job.

### 5. Create and summarize the human audit locally

Create the blinded sample:

```bash
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp06-sample \
  --sample-size 100
```

Annotate `exp06_results/audit_sample.csv` following `annotation_guidelines.md`. Then run:

```bash
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp06-summarize
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage pilot-summary
```

The immutable source columns are hashed before annotation; summarization rejects changed sample text or metadata.

## Local runner options

Preview commands and shard counts without running them:

```bash
python -m experiments.exp8_assumption_embedding_pilot.run_cpu_stage exp04 \
  --jobs 8 \
  --dry-run
```

Useful common options:

- `--jobs N`: maximum concurrent local worker processes.
- `--root PATH`: alternate Exp8 artifact root.
- `--data-dir PATH` and `--cache-dir PATH`: override shared inputs.
- `--output-dir PATH`: override the selected experiment output.
- `--force`: ignore matching resume manifests and recompute.
- `--seed`, `--bootstrap-draws`, `--episodes-per-task`, and `--anchors-per-task`: override defaults.

## GPU Slurm configuration

The two remaining shell files have repository-style `#SBATCH` headers and can be submitted from either the repository root or this directory. Override cluster defaults through environment variables when needed:

```bash
GPU_PARTITION=gpu MAX_CONCURRENT_TASKS=8 \
  sbatch --export=ALL experiments/exp8_assumption_embedding_pilot/01_cache_embeddings.sh
```

Embedding defaults are 50 episodes per task and `%8` concurrency. Fusion defaults to `%4`. GPU workers request one A6000 and 64 GB RAM. `DRY_RUN=1 bash <gpu-stage>.sh` prints the planned array and merge commands without submitting them.

## Validation

Run the complete CPU-only synthetic pipeline and unit tests directly with Python:

```bash
python -m experiments.exp8_assumption_embedding_pilot.tests.smoke_pipeline
python -m unittest discover -s experiments/exp8_assumption_embedding_pilot/tests -t . -v
```

The synthetic pipeline uses deterministic hash embeddings and is only an orchestration/output-contract test, not a scientific result.
