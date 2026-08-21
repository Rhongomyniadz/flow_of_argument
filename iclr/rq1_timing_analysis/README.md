# Duration-Free Iceberg-Ratio Timing Sensitivity Analysis

This experiment keeps the directional stance variables and control structure from the original
Exp2 analysis but removes duration from the outcome denominator. It then fits a step-wise
specification panel around that duration-free outcome. The sample is narrower because every
retained transition must belong to a timing-complete, three-turn, speaker-alternating window.

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

The outcome is the change in the raw iceberg ratio:

```text
R(t) = explicit_count / (assumption_count + 1)
Y(t) = log1p(R(t)) - log1p(R(t-1))
```

Duration does not appear in `R(t)` or `Y(t)`. Current- and previous-turn duration enter only
as explicit predictors in specifications that include those control groups.

Two decomposition outcomes use the same transition:

```text
Y_implicit(t) = log1p(assumption_count_t) - log1p(assumption_count_t-1)
Y_explicit(t) = log1p(explicit_count_t) - log1p(explicit_count_t-1)
```

Directional stance movement is calculated from `(stance_t - stance_t-1) / 5` and split into
agreement and disagreement components. The preceding boundary is split into the same two
lagged directional controls.

## Models

Four headline OLS specifications use the identical timing-complete sample and
episode-clustered standard errors:

1. `iceberg_ratio`: the duration-free outcome with the original Exp2 control structure.
2. `iceberg_ratio_duration_adjusted`: the baseline plus current-turn duration.
3. `iceberg_ratio_timing_adjusted`: the baseline plus current duration, pre-turn gap, and
   overlap.
4. `iceberg_ratio_timing_previous_duration`: the preceding model plus previous-turn duration.

The eight-stage specification panel is:

1. stance movement only;
2. lagged stance movement;
3. previous iceberg ratio;
4. linear and quadratic timeline position;
5. category fixed effects, completing the original Exp2 control structure;
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
3. the duration-free ratio model with the original Exp2 controls;
4. the full timing specification; and
5. seven models that remove one control group from the full specification.

The groups are current stance, lagged stance, previous outcome, linear and quadratic timeline
position, category fixed effects, current-turn duration, response timing (gap plus overlap), and
previous-turn duration. The previous-outcome term matches each dependent variable: previous
iceberg ratio, previous implicit-assumption count, or previous explicit-claim count.
Agreement/disagreement terms, lagged agreement/disagreement terms, timeline terms, and
gap/overlap are added or removed together.

The main fit metric is adjusted R², not raw R², because the specifications contain different
numbers of predictors. For add-one models, `delta_adjusted_r_squared` is relative to the
stance-only core. For remove-one models, it is relative to the full timing model. The Exp2
control-structure baseline is compared with the stance-only core, and the full model is compared
with that baseline.
This design shows incremental and conditional fit when controls overlap, but it does not make
correlated predictors independent or support causal attribution.

Three specification figures are produced for `delta_log_iceberg_ratio`,
`delta_log_assumption_count`, and `delta_log_explicit_count`. Each is formatted as two stacked
regression tables: agreement movement and disagreement movement. Rows are direction-specific
stance terms plus shared control variables, columns are the same 17 model configurations, and
populated cells report coefficients with episode-clustered standard errors in parentheses.
Both stance directions are still estimated jointly in every regression; the two panels only
separate their presentation. Sample size, episode clusters, and adjusted R² appear at the
bottom of each panel. The final row reports `100 * (exp(beta_stance) - 1)` for the displayed
log-change outcome.

The duration controls no longer duplicate a denominator component of the outcome. They still
change the estimand from the unadjusted duration-free association to an association conditional
on turn timing; none of the specifications is a causal model.

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
Each successful run removes the obsolete `rq1_timing_comparison` and
`rq1_stepwise_comparison` PDF/PNG files from the selected output directory.

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
  statistic used in the iceberg-ratio regression-table panels.
- `rq1_specification_panel_implicit_assumptions.csv` and
  `rq1_specification_coefficients_implicit_assumptions.csv`: model summaries and coefficients
  for `delta_log_assumption_count`.
- `rq1_specification_panel_explicit_claims.csv` and
  `rq1_specification_coefficients_explicit_claims.csv`: model summaries and coefficients for
  `delta_log_explicit_count`.
- `rq1_timing_observations.csv`: observation-level audit data.
- `rq1_timing_data_audit.json`: exclusions, hashes, formulas, timing coverage, and versions.
- `rq1_timing_summary.json`: headline estimates and interpretation fields.
- `rq1_specification_panel.pdf` and `.png`: agreement and disagreement regression-table panels
  for the iceberg-ratio outcome.
- `rq1_specification_panel_implicit_assumptions.pdf` and `.png`: matching panels for change in
  implicit assumptions.
- `rq1_specification_panel_explicit_claims.pdf` and `.png`: matching panels for change in
  explicit claims.

Attenuation is `100 * (1 - abs(beta_adjusted) / abs(beta_ratio_baseline))`. Positive values indicate
attenuation and negative values indicate amplification. Because the retained sample is large,
coefficient movement and confidence intervals should be emphasized over significance alone.

## Tests

```bash
python -m unittest discover \
  -s iclr/rq1_timing_analysis/tests \
  -p 'test_*.py' -v
```
