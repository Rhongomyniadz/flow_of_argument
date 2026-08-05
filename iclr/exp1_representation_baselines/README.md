# Explicit–Implicit Representation Baselines

This isolated experiment asks whether explicit propositions plus inferred assumptions make the true immediate next turn easier to rank than raw, explicit-only, history, assumption-only, or corrupted-assumption representations. It does not train an encoder or change any annotation pipeline.

All commands are run from the repository root. Results are model-scoped by default: a model ID such as `Qwen/Qwen3-30B-A3B-Instruct-2507` writes to `results/Qwen__Qwen3-30B-A3B-Instruct-2507/`. The `__` safely represents the `/` in a Hugging Face repository ID. The repository-level `.gitignore` excludes only bulk prepared/scored/donor JSONL data, patch shards, and Slurm logs; generated CSV tables, PDF/PNG plots, summaries, and manifests remain visible to Git.

## Conditions

The default run scores these exact condition IDs:

1. `raw_turn`
2. `raw_turn_with_history`
3. `explicit_only`
4. `assumptions_only`
5. `explicit_plus_assumptions`
6. `explicit_plus_shuffled_assumptions`
7. `explicit_plus_wrong_episode_assumptions`

`raw_turn_plus_assumptions` is available only when named explicitly with `--conditions`.

## CPU validation

Run the tests without `vllm` or a GPU:

```bash
python -m unittest discover \
  -s iclr/exp1_representation_baselines/tests \
  -p 'test_*.py' -v
```

Build a prepared cache without loading the judge model:

```bash
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py \
  --prepare_only
```

Run preparation, deterministic fake scoring, analysis, and plot generation locally:

```bash
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py \
  --dry_run \
  --max_episodes_per_category 1 \
  --bootstrap_draws 20
```

The fake scorer always gives the true continuation score 10. It validates plumbing only and must never be reported as an experimental result.

## Resumable stages

The main entrypoint supports four mutually exclusive stage flags:

```bash
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py --prepare_only
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py --score_only
python -u iclr/exp1_representation_baselines/merge_exp1_representation_patches.py --num_patches 15
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py --analysis_only
```

Successful score rows are resumed by `(pair_id, candidate_id, condition, model_name, prompt_version)`. Failed parses are removed once at restart and retried; `--overwrite_scores` intentionally replaces completed rows for the active model and prompt version.

## Slurm

Submit the runner with `sbatch`, not `bash`. The coordinator prepares once, submits a score array whose tasks each request two A6000s, and then submits dependent merge and analysis jobs. Each scoring task runs one vLLM process with `tensor_parallel_size=2` and the multiprocessing executor so the model is sharded across both allocated GPUs. An uncapped full-corpus submission is refused unless it is explicitly unlocked.

```bash
ALLOW_FULL_RUN=1 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

Five-episode-per-category GPU smoke test:

```bash
MAX_EPISODES_PER_CATEGORY=5 \
EPISODES_PER_PATCH=5 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

Useful overrides include `INPUT_DIR`, `OUTPUT_DIR`, `CATEGORIES_CSV`, `CONDITIONS_CSV`, `MODEL_NAME`, `PROMPT_BATCH_SIZE`, `MAX_TOKENS`, `SEED`, `HISTORY_TURNS`, `DRY_RUN`, and the CUDA/FlashInfer variables retained from the original Experiment 1 runner.

`OUTPUT_ROOT` changes the parent of all model folders. `OUTPUT_DIR` is an intentional exact-directory override; normally leave it unset so changing `MODEL_NAME` automatically selects a different result folder.

Before setting `ALLOW_FULL_RUN=1`, inspect at least 50 rows in `exp1_representation_scores.jsonl` across all conditions and review `exp1_representation_coverage.csv` plus the control-unavailability and parsing sections of `exp1_representation_summary.json`.

## Analysis contract

The analysis produces condition-level and pair-level metrics for the full retained, assumption-eligible, and strict-control subsets. Pairwise improvements are oriented so positive always favors the target condition. The paper plots use only strict-control pairs, ensuring all seven conditions are compared on the same examples.

The default full corpus represented 29,163 pairs in the previous experiment, or roughly 5.1 million pointwise prompts across seven conditions. Inspect the smoke-test preparation coverage, control availability, parsing failures, and saved source representations before submitting that full run.
