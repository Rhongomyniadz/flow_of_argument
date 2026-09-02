# Experiment 1: Assumption-Augmented Raw-History Baselines

This version keeps the original LLM-judge future-turn prediction design, but changes the
scientific comparison. Instead of asking whether a structured explicit–implicit state can
replace raw dialogue, it asks whether **externalizing the inferred implicit state helps a
fixed downstream judge distinguish the true future turn from hard negatives when the
observed dialogue itself is held constant**.

The prompt identity is `raw-augmentation-v1-json-evidence`, so scores from earlier versions
cannot be resumed or merged with this run.

## Five default conditions

At the fixed 256-word raw-history budget, the code prepares:

1. `raw_history`
2. `raw_history_true_assumptions`
3. `raw_history_different_episode_assumptions`
4. `raw_history_same_episode_random_turn_assumptions`
5. `raw_history_explicit`

The raw-history block is constructed once for a pair/budget and is reused byte-for-byte in
all five conditions. The budget therefore applies to the **raw-history block**. Augmented
conditions append an auxiliary block after the unchanged raw history.

The auxiliary content budget is the number of content words in the candidate-blind,
confidence-ranked assumptions selected for the current source turn (up to
`ASSUMPTION_BUDGET`, default 3). Assumptions are ranked by descending extraction confidence;
ties retain extraction order, and legacy entries without confidence follow scored entries.
True, Different-Episode, Same-Episode Random Turn, and explicit auxiliary blocks are
matched to that content-word budget. If an exact length-matched control cannot be built, that
condition is explicitly marked unavailable for the pair.

### True implicit

`raw_history_true_assumptions` appends the selected assumptions from the current source turn.

### Different-Episode

`raw_history_different_episode_assumptions` appends assumptions from a **different episode**. The
control prefers the same category and chooses the most lexically similar eligible donor,
with length similarity and a deterministic seed-based tie break. This is deliberately harder
than a random cross-topic donor.

### Same-Episode Random Turn

`raw_history_same_episode_random_turn_assumptions` appends assumptions from the **same episode**, at least three
substantive turns earlier and outside the current history window. The reserved donor is the
most lexically similar eligible earlier same-episode turn. It is reserved before negative-candidate sampling
so it cannot also become a candidate.

The name is a clearer presentation label; the sampling logic is unchanged from the previous version of this control.
It is not a uniform random draw: among eligible earlier turns, the implementation still uses the matched hard donor.

### Explicit augmentation

`raw_history_explicit` appends extracted explicit propositions from the same observed context,
starting from the current turn and moving backward, truncated to the same auxiliary content
budget as the true assumptions. This controls for the possibility that any extracted semantic
restatement helps the judge.

## Prespecified contrasts

At horizon 1 and the 256-word raw-history budget, the primary contrast is:

```text
Raw + True Implicit - Raw + Different-Episode
```

Secondary contrasts are:

```text
Raw + True Implicit - Raw + Same-Episode Random Turn
Raw + True Implicit - Raw
Raw + True Implicit - Raw + Explicit
```

The intended interpretation hierarchy is:

- **True > Different-Episode:** the identity/content of the inferred assumptions matters.
- **True > Same-Episode Random Turn:** the assumptions reflect the current conversational state rather than only
  episode/topic semantics.
- **True > Explicit:** the effect is specific to implicit state rather than generic extracted
  semantic text.
- **True > Raw:** making the inferred state explicit improves a bounded downstream model's
  access to predictive structure already latent in the transcript.

Because assumptions are inferred from the transcript, `True > Raw` should not be described as
adding information in a strict information-theoretic sense. The defensible claim is that the
assumption representation makes useful conversational structure more accessible to the judge.

## Plotting reference

The paper comparison plot uses **Raw history** as one fixed reference condition. Every plotted
point is a paired accuracy change relative to that same raw-history representation, so the
condition labels show only the augmentation names:

```text
True implicit
Different-Episode
Same-Episode Random Turn
Explicit augmentation
```

Raw history is not drawn as a separate performance series. In the comparison plot, zero denotes the
raw-history reference and is shown only as an unlabeled vertical guide. This plotting choice does
not remove the prespecified True-vs-control contrasts from the CSV analysis; it only keeps the visual
performance reference constant so vertical differences across curves are directly interpretable.

## Token-budget sanity check

The analysis also reports how much conversational context each nominal raw-history budget
actually contains. For every assumption-eligible, candidate-complete source context, the code
counts a dialogue turn only when at least one raw-text token from that turn is included after
budgeting; structural turn headers alone do not count. Source turns are deduplicated across
future horizons.

For each configured budget (for example 128, 256, and 512), the sanity-check CSV reports the
mean included-turn count and an episode-clustered bootstrap 95% confidence interval. The
corresponding plot shows the mean with those confidence-interval error bars. This makes the
nominal token budgets interpretable in units of conversational turns without changing any
scoring condition.

## Candidate task and judge

Candidate construction remains unchanged: one true future turn plus 24 hard negatives, with
each true/negative pair shown in both A/B orders. The exact candidate pool is shared across
all conditions.

The judge prompt is deliberately neutral. It asks which candidate is the more plausible
continuation and considers:

- semantic relevance to the observed conversation;
- conversational obligations created by preceding turns;
- speaker intent, stance, and continuity;
- supported presuppositions/assumptions;
- local discourse coherence at the requested horizon.

It no longer explicitly privileges the final dialogue act, because that instruction created
an avoidable representational asymmetry in the previous structured-vs-raw design.

The judge must return strict JSON with exactly `answer` and `evidence`. For cross-model
normalization, the parser also accepts one complete `json`-fenced object with no text before or
after the fence; all key, answer, and evidence constraints remain identical.

## Statistics

Accuracy is computed from the order-swapped binary judgments. All condition contrasts are
paired on the same source/candidate decisions. Confidence intervals use the existing
conversation/episode-clustered bootstrap.

The diagnostic gate now prioritizes:

1. `True - Different-Episode`;
2. `True - Same-Episode Random Turn`;
3. `True - Explicit`;
4. `True - Raw`.

A positive `True - Different-Episode` result alone is not treated as sufficient evidence of a dynamic
latent state; the Same-Episode Random Turn control is required for the stronger state-specificity
interpretation.

## Preserving the previous run

The previous structured-vs-raw results should be archived once under:

```text
iclr/exp1_representation_baselines/results/previous_matched_history_run/
```

For the default Qwen judge, the old model output directory becomes:

```text
iclr/exp1_representation_baselines/results/previous_matched_history_run/Qwen__Qwen3-30B-A3B-Instruct-2507/
```

The experiment code itself keeps the original output logic unchanged. New runs still write to:

```text
iclr/exp1_representation_baselines/results/Qwen__Qwen3-30B-A3B-Instruct-2507/
```

On an existing checkout, archive the old result directory once before launching this new design:

```bash
mkdir -p iclr/exp1_representation_baselines/results/previous_matched_history_run

mv iclr/exp1_representation_baselines/results/Qwen__Qwen3-30B-A3B-Instruct-2507 \
   iclr/exp1_representation_baselines/results/previous_matched_history_run/
```

Do not change `OUTPUT_ROOT` or `OUTPUT_DIR` for the new run unless you intentionally want a
different destination. This keeps downstream result and Overleaf paths unchanged while preserving
the previous experiment separately.

## CPU validation

```bash
python -m unittest discover \
  -s iclr/exp1_representation_baselines/tests \
  -p 'test_*.py' -v
```

For a small plumbing run:

```bash
python -u iclr/exp1_representation_baselines/exp1_representation_baselines.py \
  --dry_run \
  --max_episodes_per_category 1 \
  --bootstrap_draws 20
```

## Slurm smoke run

Run CPU-only preparation directly, without Slurm:

```bash
python -u iclr/exp1_representation_baselines/prepare_exp1_representation.py \
  --max_episodes_per_category 5
```

After that job finishes successfully, submit GPU scoring:

```bash
MAX_EPISODES_PER_CATEGORY=5 \
sbatch iclr/exp1_representation_baselines/run_exp1_representation_baselines.sh
```

`prepare_exp1_representation.py` writes the prepared pairs and preparation manifest without
loading `vllm`. The scoring runner does not repeat preparation; it fails clearly if those artifacts
are absent. Use matching settings for preparation and scoring, especially the episode limit.

The default runner uses:

```text
raw_history
raw_history_true_assumptions
raw_history_different_episode_assumptions
raw_history_same_episode_random_turn_assumptions
raw_history_explicit
```

Set `ALLOW_FULL_RUN=1` only after reviewing the smoke-run donor audit, control coverage, and
diagnostic gate. A different-family judge should be used as a robustness run before making a
full-corpus claim.

## Main outputs

- `exp1_representation_metrics_long.csv`
- `exp1_representation_pairwise_deltas.csv`
- `exp1_representation_flip_analysis.csv` (strict helpful/harmful candidate flips plus order-averaged deltas)
- `exp1_representation_decomposition.csv`
- `exp1_representation_donors.jsonl`
- `exp1_representation_audit_sample.csv`
- `exp1_representation_diagnostic_gate.json`
- `exp1_representation_diagnostic_comparison.{pdf,png}`: strict complete-case paired accuracy change relative to raw history across future horizons.
- `exp1_representation_decomposition_lifts.{pdf,png}`: paired true-assumption accuracy differences against each comparator across horizons.
- `exp1_representation_turns_per_budget.csv`
- `exp1_representation_turns_per_budget.{pdf,png}`

The donor audit records donor source, fallback level, target auxiliary length, and the exact
length-matched assumptions used for Different-Episode and Same-Episode Random Turn controls.
