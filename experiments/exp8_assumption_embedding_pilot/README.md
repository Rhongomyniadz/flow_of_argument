# Exp8: assumption-embedding pilot

This directory is a self-contained ICLR-oriented pilot. It does not import or write Exp1–Exp7 experiment code or result directories. Preparation and embeddings are shared; every exploratory result is isolated in `exp01_results/` through `exp06_results/`.

## What each experiment tests

| Result | Question | Main comparison |
|---|---|---|
| `exp01_results/` | Did assumptions help in the existing Exp1 pair table, and where? | with vs. without assumptions, episode-clustered audit |
| `exp02_results/` | Do frozen assumption embeddings improve next-turn retrieval? | current/history/explicit/assumption/full/shuffled |
| `exp03_results/` | Is there linearly decodable next-turn signal beyond surface/history features? | baseline vs. full vs. shuffled residual model |
| `exp04_results/` | Is any gain specific to the correct assumptions? | correct vs. same-episode, same-category, and explicit-length controls |
| `exp05_results/` | Does a small trainable fusion module consistently use the signal? | history/full/shuffled × seeds 42, 43, 44 |
| `exp06_results/` | Are sampled assumptions human-supported rather than fluent noise? | blinded two-annotator audit |

The retrieval candidate pool is frozen during preparation. Each anchor has exactly one true next turn, no duplicates, and no current-turn self candidate. Show identity—not episode identity—defines the train/validation/test split.

## Stage entrypoints

| File | Default resources | Output |
|---|---|---|
| `00_prepare_data.sh` | CPU array, 250 episodes/task, `%32` | `shared_data/` |
| `01_cache_embeddings.sh` | A6000 array, 50 episodes/task, `%8` | `shared_cache/` |
| `02_run_exp01_audit.sh` | one CPU job | `exp01_results/` |
| `03_run_exp02_retrieval.sh` | CPU array, 1,000 anchors/task, `%32` | `exp02_results/` |
| `04_run_exp03_residual.sh` | three CPU tasks | `exp03_results/` |
| `05_run_exp04_controls.sh` | control × anchor-shard CPU array | `exp04_results/` |
| `06_run_exp05_fusion.sh` | nine A6000 tasks, `%4` | `exp05_results/` |
| `07_run_exp06_sample.sh` | one CPU job | `exp06_results/audit_sample.csv` |
| `08_run_exp06_summarize.sh` | one CPU job after annotation | `exp06_results/` |
| `09_summarize_pilot.sh` | one CPU job | `pilot_summary.json` and `.md` |

Array workers request `05:45:00`; merge jobs request one hour. Every array worker writes a private `patch_NNNN_of_NNNN/` directory. A patch is reused only when the input, split, model/instruction, and configuration hashes match. Merge jobs reject missing/misindexed patches, mixed hashes, duplicate anchors, changed candidate pools, and a fusion condition/seed mapping different from the declared nine tasks.

## Before submitting

Production embeddings use the official `SentenceTransformer` interface for [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B). Query text receives the retrieval instruction; candidate turns are encoded as documents; `normalize_embeddings=True` uses the model's configured normalized last-token pooling. Install the optional model dependencies and make the model available on compute nodes:

```bash
pip install -e '.[llm]'
```

Input episode JSON must contain a resolvable show identifier (`show_id`, `rssKey`, `podcast_id`, or `feed_id`). If it does not, pass `SHOW_MAP=/path/to/episode_show_map.csv`. `ALLOW_EPISODE_FALLBACK=1` is available only as an explicit diagnostic fallback and weakens the split guarantee.

Exp01 additionally requires `PAIRS_CSV`. Required columns are `episode_id`, `reciprocal_rank_without_assumptions`, `reciprocal_rank_with_assumptions`, `top1_without_assumptions`, and `top1_with_assumptions`.

## Dry run and submission

All numbered stage files include repository-style `#SBATCH` headers. They retain the submission directory as the Slurm working directory and resolve whether submission happened from the repository root or this Exp8 directory. An individual stage can therefore be submitted directly from either location:

```bash
sbatch experiments/exp8_assumption_embedding_pilot/00_prepare_data.sh
# or, from experiments/exp8_assumption_embedding_pilot:
sbatch 00_prepare_data.sh
```

Stage 00 is a short Slurm orchestration job: it counts episodes, submits the `05:45:00` worker array, and submits the merge job with an `afterok` dependency. Its log prints the worker, merge, and final job IDs. The other array stages behave the same way.

Inspect every array size, command, dependency, and output without calling `sbatch`:

```bash
DRY_RUN=1 PAIRS_CSV=/path/to/exp1_pairs.csv bash experiments/exp8_assumption_embedding_pilot/submit_all.sh
```

Submit stages 00–07:

```bash
PAIRS_CSV=/path/to/exp1_pairs.csv bash experiments/exp8_assumption_embedding_pilot/submit_all.sh
```

If Exp01 is deliberately excluded, set `SKIP_EXP01=1`. Partition, sharding, and concurrency settings can be overridden with `CPU_PARTITION`, `GPU_PARTITION`, `EPISODES_PER_TASK`, `ANCHORS_PER_TASK`, and `MAX_CONCURRENT_TASKS`. Each stage prints `FINAL_JOB_ID=<id>` for dependency automation.

To smoke-test one shard without Slurm, set `LOCAL=1` and optionally `PATCH_INDEX`:

```bash
LOCAL=1 PATCH_INDEX=0 EMBEDDING_BACKEND=hash bash experiments/exp8_assumption_embedding_pilot/01_cache_embeddings.sh
```

To merge locally after all patches exist, invoke the same array stage with `MODE=merge NUM_PATCHES=N`.

## Manual audit pause

Stage 07 creates the sample and guidelines. The sample is treated as immutable: rerunning with the same hashes reuses it, while a changed configuration is rejected. After two annotators fill the two label columns, run:

```bash
bash experiments/exp8_assumption_embedding_pilot/08_run_exp06_summarize.sh
bash experiments/exp8_assumption_embedding_pilot/09_summarize_pilot.sh
```

## Validation

The CPU-only synthetic pipeline runs all six experiments, every array merge, annotation summarization, and the pilot summary using deterministic hash embeddings:

```bash
bash experiments/exp8_assumption_embedding_pilot/run_smoke.sh
python -m unittest discover -s experiments/exp8_assumption_embedding_pilot/tests -t . -v
```

The smoke backend is intentionally not a scientific result. It exists only to validate orchestration and output contracts without a GPU or model download.
