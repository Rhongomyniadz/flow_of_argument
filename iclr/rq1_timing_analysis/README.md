# Exp2 Timing Sensitivity Analysis

This experiment restores the directional model used in the original Exp2 analysis and then
fits a step-wise specification panel around it. The outcome, stance variables, lagged stance
controls, previous outcome, timeline terms, category fixed effects, and episode-clustered
inference match `experiments/exp2_iceberg/`. The sample is narrower because every retained
transition must belong to a timing-complete, three-turn, speaker-alternating window.

## Observation contract

The input defaults to `data/stance_labeled/1024`. An observation uses three consecutive raw
records, `t-2`, `t-1`, and `t`. All three must be substantive, nonempty, stance-labeled turns,
and the speaker must change at both boundaries. A backchannel, procedural turn, missing
record, or same-speaker boundary breaks the window; the script never skips it to create a
transition.

Timing for the current and previous turns must be recoverable from `start_time` and
`end_time`, or from one endpoint plus a positive `duration`. Invalid or nonpositive timing is
reported and excluded. A negative transition gap is represented by `overlap=1` and a zero
non-overlap gap.

The restored Exp2 outcome is iceberg density per second:

```text
D(t) = [explicit_count / (assumption_count + 1)] / duration_seconds
Y(t) = log1p(D(t)) - log1p(D(t-1))
```

Directional stance movement is calculated from `(stance_t - stance_t-1) / 5` and split into
agreement and disagreement components. The preceding boundary is split into the same two
lagged directional controls.

## Models

Four headline OLS specifications use the identical timing-complete sample and
episode-clustered standard errors:

1. `per_second`: the original Exp2 directional specification.
2. `per_second_duration_adjusted`: Exp2 plus current-turn duration.
3. `per_second_timing_adjusted`: Exp2 plus current duration, pre-turn gap, and overlap.
4. `per_second_timing_previous_duration`: the preceding model plus previous-turn duration.

The eight-stage specification panel is:

1. stance movement only;
2. lagged stance movement;
3. previous per-second iceberg density;
4. linear and quadratic timeline position;
5. category fixed effects, completing the original Exp2 directional model;
6. current-turn duration;
7. pre-turn gap and overlap; and
8. previous-turn duration.

Each stage retains all earlier groups. The panel reports the agreement and disagreement
coefficients, their episode-clustered 95% intervals, and the incremental change introduced by
each group.

Current-turn duration already appears in the outcome denominator. Stages 6–8 are therefore
sensitivity specifications with a different estimand, not replacements for the original Exp2
model and not causal models.

## Run

Install the dependencies declared in `pyproject.toml`, then run from the repository root:

```bash
python -u iclr/rq1_timing_analysis/rq1_timing_analysis.py
```

For a small plumbing run:

```bash
python -u iclr/rq1_timing_analysis/rq1_timing_analysis.py \
  --max_episodes 100 \
  --output_dir iclr/rq1_timing_analysis/results_smoke
```

The default output directory is `iclr/rq1_timing_analysis/results/`. Summary CSVs, JSON files,
and PDF/PNG figures remain reviewable in git. The observation-level
`rq1_timing_observations.csv` is ignored by the repository-level `.gitignore`.

## Outputs

- `rq1_timing_coefficients.csv`: coefficients for the four headline specifications.
- `rq1_timing_stance_comparison.csv`: agreement/disagreement paths through timing controls.
- `rq1_timing_model_fit.csv`: shared-sample fit statistics.
- `rq1_stepwise_coefficients.csv`: coefficients from all eight nested specifications.
- `rq1_stepwise_stance_comparison.csv`: stance estimates and incremental changes by stage.
- `rq1_stepwise_model_fit.csv`: fit statistics for all eight specifications.
- `rq1_timing_observations.csv`: observation-level audit data.
- `rq1_timing_data_audit.json`: exclusions, hashes, formulas, timing coverage, and versions.
- `rq1_timing_summary.json`: headline estimates and interpretation fields.
- `rq1_timing_comparison.pdf` and `.png`: original Exp2 baseline through timing adjustment.
- `rq1_stepwise_comparison.pdf` and `.png`: two-panel step-wise coefficient paths.

Attenuation is `100 * (1 - abs(beta_adjusted) / abs(beta_Exp2))`. Positive values indicate
attenuation and negative values indicate amplification. Because the retained sample is large,
coefficient movement and confidence intervals should be emphasized over significance alone.

## Tests

```bash
python -m unittest discover \
  -s iclr/rq1_timing_analysis/tests \
  -p 'test_*.py' -v
```
