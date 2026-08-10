# Pairwise Explicit–Implicit Confirmatory Ranking

This experiment tests whether a small, locally grounded implicit representation improves
next-turn discrimination when explicit propositions underspecify the current turn. It uses
hard negative pools and order-swapped forced-choice judgments. The prompt identity is
`representation-pairwise-v4-order-swapped`, so pointwise 1–20 scores from older runs cannot
be resumed or merged.

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
python deduplicate_data.py --overwrite
```

The cleaner removes turns shorter than 50 words, merges adjacent remaining turns from
the same speaker, caps explicit propositions and assumptions at the ten
highest-confidence items, removes duplicate episodes within each logical dataset, and
validates two-speaker ABAB alternation. Set `INPUT_DIR` in the Slurm runner or pass
`--input_dir` to the Python script only when intentionally using another prepared
dataset. The cleaner now records original indices on every retained turn. Preparation rejects
pairs that bridge deleted turns or contain a merge across nonconsecutive original indices, and
fails with an actionable error if it sees old cleaned data without this provenance.

## Default diagnostic conditions

The default pipeline scores seven conditions:

1. `raw_turn`
2. `raw_turn_with_history`
3. `raw_turn_plus_assumptions`
4. `explicit_only`
5. `explicit_plus_top3_assumptions`
6. `explicit_plus_shuffled_assumptions`
7. `explicit_plus_wrong_episode_assumptions`

Preparation uses the final 100 words of the current turn and the first 100 words of every
candidate by default. It selects three assumptions without looking at candidates, ranking
them by lexical grounding in the final source window. The same three-item budget applies to
true, shuffled, and wrong-episode blocks. `assumptions_only`, `explicit_plus_top1_assumption`,
and `explicit_plus_assumptions` remain optional diagnostics.

The principal contrasts are:

| Contrast | Diagnostic question |
|---|---|
| `explicit_plus_top3_assumptions - explicit_only` | Do assumptions help sparse explicit representations? |
| true top 3 minus shuffled top 3 | Is the gain specific to the extracted assumptions? |
| true top 3 minus wrong-episode top 3 | Is the gain specific to the local dialogue state? |
| `raw_turn_plus_assumptions - raw_turn` | Do assumptions add signal beyond lexical context? |
| `raw_turn - explicit_only` | How much information is lost during proposition extraction? |
| `raw_turn_with_history - raw_turn` | How much does discourse history contribute? |

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
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

The coordinator performs preparation, submits a scoring array, then submits dependent
merge and analysis jobs. Every scoring array task requests two A6000 GPUs and starts one
vLLM process with `tensor_parallel_size=2` and the multiprocessing executor.
The runner defaults to five episodes per array patch and an eight-hour wall-time ceiling;
jobs stop as soon as their patch is complete.

Each true continuation is compared directly with each of its 24 negatives. Every comparison
is presented twice, once in each A/B order. The judge must output exactly `A` or `B`.
The positive preference is averaged across the two orders; disagreements become ties, and
the fixed candidate order resolves rank ties. A complete pair-condition therefore requires
48 parsed choice rows. The default output budget is four tokens, with malformed outputs
retried using budgets up to 16 tokens.

Negative pools prioritize up to 12 hard same-episode nonadjacent turns, then six
same-category/same-move turns, followed by topic- and length-matched category and global
backfills. A same-episode assumption donor is reserved before candidate construction so the
wrong-episode control remains independent of the candidate pool.

Useful overrides include `MODEL_NAME`, `CONDITIONS_CSV`, `PROMPT_BATCH_SIZE`,
`MAX_TOKENS`, `MAX_SCORE_RETRIES`, `MAX_RETRY_TOKENS`, `SEED`, `HISTORY_TURNS`,
`SOURCE_TAIL_WORDS`, `CANDIDATE_HEAD_WORDS`, `ASSUMPTION_BUDGET`,
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

The analysis subsets are `full`, `assumption_eligible`, `sparse_explicit` (at most four
explicit propositions), `dense_explicit` (at least five), and `complete_case`.
`complete_case` requires all selected conditions to parse successfully on the same pair.

## Decision gate

The script never automatically authorizes a full-corpus run from one judge. The primary
gate requires a positive sparse-explicit MRR interval, superiority to both corrupted
controls, positive effects in at least two categories, and at least 98% retained coverage.
A passing run advances only to a different-family judge smoke test and manual audit.

Run the same smoke test with another judge by changing `MODEL_NAME`; its artifacts go to
a separate model folder. Only after both smoke runs and the audit are reviewed should the
full corpus be unlocked:

```bash
ALLOW_FULL_RUN=1 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

The full run remains expensive. Do not unlock it merely because preparation and parsing
succeeded.
