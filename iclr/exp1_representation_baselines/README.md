# Experiment 1: Matched-History Representation Baselines

This experiment tests whether an accumulated explicit–implicit discourse state predicts a
future turn better than raw dialogue history when both representations are built from the
same source turns and have the same maximum length. It does not train an encoder and it does
not select a condition after seeing the result.

The prompt identity is `representation-matched-history-v2-json-evidence`, so scores from the
older representation experiment cannot be resumed or merged. Results remain model-scoped;
the default Qwen run writes under:

```text
iclr/exp1_representation_baselines/results/Qwen__Qwen3-30B-A3B-Instruct-2507/
```

## Prespecified design

For each source turn `t`, the default history is the previous three substantive turns plus
the current source turn: `t-3, ..., t`. All conditions use this same candidate-blind window.
The default source-representation caps are 128, 256, and 512 whitespace tokens. The cap
includes turn and state labels, and the actual realized length is saved with every condition.
Content is allocated from the current turn backward, then displayed chronologically.

The five default base conditions are expanded at every budget:

1. `raw_history`
2. `structured_explicit_history`
3. `structured_explicit_assumption_history`
4. `structured_explicit_shuffled_assumption_history`
5. `structured_explicit_wrong_episode_assumption_history`

The structured state keeps temporal turn boundaries. Repeated explicit propositions and
assumptions are deterministically deduplicated, retaining their most recent occurrence.
Assumptions are selected within each source turn without looking at candidates. The corrupted
conditions preserve the explicit history and replace only the implicit content with a
different-episode donor or a nonadjacent earlier donor from the same episode.

The primary comparison is fixed in advance:

```text
structured_explicit_assumption_history - raw_history
```

The 256-token, one-turn-horizon contrast is the primary diagnostic. The 128- and 512-token
results show the budget curve. Secondary contrasts test incremental assumption value against
structured explicit-only history and specificity against both corrupted-assumption controls.
Positive pairwise values always favor the condition on the left. The analysis reports all
budgets even when the expected ordering does not occur.

The existing legacy single-turn conditions remain available when explicitly requested, but
they are no longer the default experiment.

## Data and scoring

Preparation reads `data_cleaned/conversation_moves_labeled/`. Regenerate it from the
repository root before a confirmatory run:

```bash
python deduplicate_data.py --overwrite
```

Preparation rejects pairs without contiguous original-turn provenance. The positive at
horizon `n` is the turn exactly `n` positions after the source. Default horizons are 1, 3,
and 5; only positive odd horizons are accepted for the cleaned ABAB dialogues.

Each positive is paired with 24 hard negatives. Every pair is shown in both A/B orders, so a
complete pair-condition contains 48 valid binary judgments. Accuracy is the proportion that
selects the true future turn. The judge must emit strict JSON with exactly `answer` and
`evidence`. Fake scoring validates plumbing only and is not an experimental result.

## CPU validation

```bash
python -m unittest discover \
  -s iclr/exp1_representation_baselines/tests \
  -p 'test_*.py' -v
```

For a small end-to-end plumbing run:

```bash
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py \
  --dry_run \
  --max_episodes_per_category 1 \
  --bootstrap_draws 20
```

## Slurm run

Create the log directory, then submit the coordinator:

```bash
mkdir -p iclr/exp1_representation_baselines/_log

MAX_EPISODES_PER_CATEGORY=5 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

The coordinator prepares once, submits a GPU scoring array, then dependent merge and analysis
jobs. Important overrides include `MODEL_NAME`, `HISTORY_TURNS`,
`REPRESENTATION_BUDGETS_CSV`, `FUTURE_HORIZONS_CSV`, `CONDITIONS_CSV`,
`MAX_EPISODES_PER_CATEGORY`, and `EPISODES_PER_PATCH`. The default runner uses two A6000 GPUs
and `tensor_parallel_size=2`.

An ungated full-corpus submission is blocked unless `ALLOW_FULL_RUN=1`. Even a passing primary
run advances only to a different-family judge smoke test and manual audit.

## Outputs

In addition to long/wide metrics, coverage, category/move summaries, and score manifests, the
analysis emits:

- `exp1_representation_decomposition.csv`: all planned matched-budget contrasts.
- `exp1_representation_audit_sample.csv`: strongest wins, losses, and ties with both primary
  source representations.
- `exp1_representation_diagnostic_gate.json`: the prespecified 256-token primary decision and
  all-budget curve.
- `exp1_representation_diagnostic_comparison.{pdf,png}`: accuracy by representation budget.
- `exp1_representation_decomposition_lifts.{pdf,png}`: paired lifts by budget.

The preparation manifest records the exact budgeting policy, token-count convention,
deduplication rule, source window, seed, and hashes needed to audit or reproduce the run.
