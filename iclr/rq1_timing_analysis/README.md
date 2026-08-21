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

### Grouped add/remove specification panel

Because several controls are related, the analysis also fits 17 grouped specifications rather
than interpreting every term as independent:

1. a stance-only core;
2. seven models that add one control group to that core;
3. the original Exp2 baseline;
4. the full timing specification; and
5. seven models that remove one control group from the full specification.

The groups are current stance, lagged stance, previous per-second density, linear and
quadratic timeline position, category fixed effects, current-turn duration, response timing
(gap plus overlap), and previous-turn duration. Agreement/disagreement terms, lagged
agreement/disagreement terms, timeline terms, and gap/overlap are added or removed together.

The main fit metric is adjusted R², not raw R², because the specifications contain different
numbers of predictors. For add-one models, `delta_adjusted_r_squared` is relative to the
stance-only core. For remove-one models, it is relative to the full timing model. The Exp2
baseline is compared with the stance-only core, and the full model is compared with Exp2.
This design shows incremental and conditional fit when controls overlap, but it does not make
correlated predictors independent or support causal attribution.

The specification figure is formatted as two stacked regression tables: agreement movement
and disagreement movement. Rows are direction-specific stance terms plus shared control
variables, columns are the 17 model configurations, and populated cells report the coefficient
for change in `delta_log_density_per_second` with the episode-clustered standard error in
parentheses. Both stance directions are still estimated jointly in every regression; the two
panels only separate their presentation. Sample size, episode clusters, and adjusted R² appear
at the bottom of each panel. The final row converts the panel's current-stance coefficient into
the estimated percentage change in the per-second iceberg ratio as
`100 * (exp(beta_stance) - 1)`.

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
- `rq1_specification_panel.csv`: grouped add/remove membership, stance estimates, R², adjusted
  R², and reference-relative changes for all 17 specifications.
- `rq1_specification_coefficients.csv`: every coefficient and episode-clustered inferential
  statistic used in the two regression-table panels.
- `rq1_timing_observations.csv`: observation-level audit data.
- `rq1_timing_data_audit.json`: exclusions, hashes, formulas, timing coverage, and versions.
- `rq1_timing_summary.json`: headline estimates and interpretation fields.
- `rq1_timing_comparison.pdf` and `.png`: original Exp2 baseline through timing adjustment.
- `rq1_stepwise_comparison.pdf` and `.png`: two-panel step-wise coefficient paths.
- `rq1_specification_panel.pdf` and `.png`: agreement and disagreement regression-table panels
  across all grouped specifications.

Attenuation is `100 * (1 - abs(beta_adjusted) / abs(beta_Exp2))`. Positive values indicate
attenuation and negative values indicate amplification. Because the retained sample is large,
coefficient movement and confidence intervals should be emphasized over significance alone.

## Tests

```bash
python -m unittest discover \
  -s iclr/rq1_timing_analysis/tests \
  -p 'test_*.py' -v
```
