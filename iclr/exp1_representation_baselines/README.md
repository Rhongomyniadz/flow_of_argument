# Explicit–Implicit Diagnostic Decomposition

This experiment diagnoses where next-turn information is lost instead of treating
`explicit_plus_assumptions` as a single paper claim. Candidate pools, pair construction,
and deterministic task ordering remain unchanged. The new prompt identity is
`representation-diagnostic-v3-full-json-20`, so scores from the previous judge prompt
cannot be resumed accidentally.

Results remain model-scoped. For example, `Qwen/Qwen3-30B-A3B-Instruct-2507` writes to:

```text
iclr/exp1_representation_baselines/results/Qwen__Qwen3-30B-A3B-Instruct-2507/
```

Changing `MODEL_NAME` selects another model-named folder automatically.

## Input data

Preparation reads the cleaned conversation-move dataset by default:

```text
data_cleaned/conversation_moves_labeled/
```

Generate it from the repository root before preparing the experiment:

```bash
python deduplicate_data.py
```

The cleaner removes turns shorter than 50 words, merges adjacent remaining turns from
the same speaker, caps explicit propositions and assumptions at the ten
highest-confidence items, removes duplicate episodes within each logical dataset, and
validates two-speaker ABAB alternation. Set `INPUT_DIR` in the Slurm runner or pass
`--input_dir` to the Python script only when intentionally using another prepared
dataset.

## Default diagnostic conditions

The default pipeline scores seven conditions:

1. `raw_turn`
2. `raw_turn_with_history`
3. `raw_turn_plus_assumptions`
4. `explicit_only`
5. `explicit_plus_top1_assumption`
6. `explicit_plus_top3_assumptions`
7. `explicit_plus_assumptions`

The first-one and first-three conditions use the deterministic extraction order because
the annotations do not contain an importance ranking. The original conditions
`assumptions_only`, `explicit_plus_shuffled_assumptions`, and
`explicit_plus_wrong_episode_assumptions` remain available through `--conditions`, but
they are no longer part of the default diagnostic run.

The principal contrasts are:

| Contrast | Diagnostic question |
|---|---|
| `explicit_plus_assumptions - explicit_only` | Do assumptions add signal after abstraction? |
| `raw_turn_plus_assumptions - raw_turn` | Do assumptions add signal beyond lexical context? |
| `raw_turn - explicit_only` | How much information is lost during proposition extraction? |
| `raw_turn_with_history - raw_turn` | How much does discourse history contribute? |
| first 1 / first 3 / all assumptions | Does assumption volume create overload? |

Positive pairwise values always favor the condition on the left.

## CPU validation

Run tests without importing `vllm`:

```bash
python -m unittest discover \
  -s iclr/exp1_representation_baselines/tests \
  -p 'test_*.py' -v
```

Run a one-episode fake-score pipeline:

```bash
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py \
  --dry_run \
  --max_episodes_per_category 1 \
  --bootstrap_draws 20
```

Fake scoring validates plumbing only and must not be reported as an experimental result.

## GPU smoke pipeline

The `_log` directory must exist before Slurm opens the job log. Submit the shell file with
`sbatch`, not `bash`:

```bash
mkdir -p iclr/exp1_representation_baselines/_log

MAX_EPISODES_PER_CATEGORY=5 \
EPISODES_PER_PATCH=5 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

The coordinator performs preparation, submits a scoring array, then submits dependent
merge and analysis jobs. Every scoring array task requests two A6000 GPUs and starts one
vLLM process with `tensor_parallel_size=2` and the multiprocessing executor.

Judge scores use an integer **1--20** scale. A judgment is valid only when the complete
JSON object parses with exactly three fields: integer `score`, non-empty string
`rationale`, and numeric `confidence` in `[0, 1]`.

The default initial output budget is 256 tokens per judgment. If a full response fails
to parse, only the failed judgment is retried with a larger budget: 512 tokens on the
first retry and 1024 on the second retry by default. The fixed 25/25 candidate-pool
requirement remains unchanged; retries repair judge-format failures rather than weakening
the ranking comparison.

Useful overrides include `MODEL_NAME`, `CONDITIONS_CSV`, `PROMPT_BATCH_SIZE`,
`MAX_TOKENS`, `MAX_SCORE_RETRIES`, `MAX_RETRY_TOKENS`, `SEED`, `HISTORY_TURNS`,
`AUDIT_SAMPLE_SIZE_PER_OUTCOME`, and the CUDA/FlashInfer variables.

## Diagnostic outputs

Analysis emits the usual long/wide metrics, category/move summaries, coverage, and
pairwise deltas plus:

- `exp1_representation_decomposition.csv`: the planned contrasts in a compact table.
- `exp1_representation_audit_sample.csv`: up to 25 strongest wins, losses, and ties,
  including the raw turn, extracted representations, history, and true continuation.
- `exp1_representation_diagnostic_gate.json`: a machine-readable interpretation and
  smoke-test gate.
- `exp1_representation_diagnostic_comparison.{pdf,png}`: complete-case MRR and Top-1.
- `exp1_representation_decomposition_lifts.{pdf,png}`: assumption-eligible MRR lifts.

The analysis subsets are `full`, `assumption_eligible`, and `complete_case`.
`complete_case` requires all selected conditions to parse successfully on the same pair.

## Decision gate

The script never automatically authorizes a full-corpus run from one judge. It first checks whether an
incremental MRR interval excludes zero, whether positive effects span at least two
categories, and whether every condition retains at least 98% of eligible pairs. A
passing single-model diagnostic advances to a second-model smoke test and manual audit.

Run the same smoke test with another judge by changing `MODEL_NAME`; its artifacts go to
a separate model folder. Only after both smoke runs and the audit are reviewed should the
full corpus be unlocked:

```bash
ALLOW_FULL_RUN=1 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

The full run remains expensive. Do not unlock it merely because preparation and parsing
succeeded.
