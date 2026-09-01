# Absolute Stance-Change Timing Sensitivity Analysis

This experiment uses current raw stance as the focal predictor while retaining the control
structure from the preceding panel design. The dependent variable is absolute turn-to-turn stance
change in raw stance-scale points. The sample is narrower because every retained transition must
belong to a timing-complete, three-turn, speaker-alternating window.

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

The outcome is absolute change in raw stance:

```text
Y(t) = abs(stance_t - stance_t-1)
```

No normalization is applied to this outcome. Current- and previous-turn duration enter only as
explicit predictors in specifications that include those control groups.

The focal IV is `stance_t` in its original units. Specifications that include the lagged-stance
group add `stance_t-1`, also in its original units. The only constructed stance-change measure
is the absolute outcome `abs(stance_t - stance_t-1)`.

## Models

Four headline OLS specifications use the identical timing-complete sample and
episode-clustered standard errors:

1. `absolute_stance_change`: current and previous raw stance, previous log iceberg ratio,
   timeline terms, and category fixed effects.
2. `absolute_stance_change_duration_adjusted`: the baseline plus current-turn duration.
3. `absolute_stance_change_timing_adjusted`: the baseline plus current duration, pre-turn gap,
   and overlap.
4. `absolute_stance_change_timing_previous_duration`: the preceding model plus previous-turn
   duration.

The eight-stage specification panel is:

1. current raw stance only;
2. previous raw stance;
3. previous iceberg ratio;
4. linear and quadratic timeline position;
5. category fixed effects, completing the original Exp2 control structure;
6. current-turn duration;
7. pre-turn gap and overlap; and
8. previous-turn duration.

Each stage retains all earlier groups. The panel reports the current raw-stance coefficient, its
episode-clustered 95% interval, and the incremental change introduced by each group.

### Grouped add/remove specification panel

Because several controls are related, the analysis also fits 17 grouped specifications rather
than interpreting every term as independent:

1. a stance-only core;
2. seven models that add one control group to that core;
3. the baseline with the original non-timing controls;
4. the full timing specification; and
5. seven models that remove one control group from the full specification.

The groups are current raw stance, previous raw stance, previous log iceberg ratio, linear and
quadratic timeline position, category fixed effects, current-turn duration, response timing (gap
plus overlap), and previous-turn duration. Timeline terms and gap/overlap are added or removed
together.

The main fit metric is adjusted R², not raw R², because the specifications contain different
numbers of predictors. For add-one models, `delta_adjusted_r_squared` is relative to the
stance-only core. For remove-one models, it is relative to the full timing model. The Exp2
control-structure baseline is compared with the stance-only core, and the full model is compared
with that baseline.
This design shows incremental and conditional fit when controls overlap, but it does not make
correlated predictors independent or support causal attribution.

One specification figure is produced for `absolute_stance_change`. Rows are current raw stance,
previous raw stance, and the shared controls; columns are the same 17 model configurations.
Populated cells report coefficients with episode-clustered standard errors in parentheses. Sample
size, episode clusters, and adjusted R² appear at the bottom. The final row reports the estimated
change in absolute stance-change points for a one-point increase in current raw stance.

A separate duration figure rounds durations to milliseconds, divides observations into up to 20
equal-frequency duration intervals, and plots the median raw iceberg ratio with its interquartile
range. Its x-axis is current-turn duration in seconds on a log scale. This plot is descriptive and
does not adjust for stance or other controls.

Because current and previous raw stance also define the absolute-change outcome, their
coefficients are partly mechanical and must be interpreted descriptively. Timing controls change
the estimand to an association conditional on turn timing; none of the specifications is causal.

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
Each successful run removes the obsolete comparison plots and the former implicit-assumption and
explicit-claim specification outputs from the selected output directory.

## Outputs

- `rq1_timing_coefficients.csv`: coefficients for the four headline specifications.
- `rq1_timing_stance_comparison.csv`: the current raw-stance coefficient path through timing controls.
- `rq1_timing_model_fit.csv`: shared-sample fit statistics.
- `rq1_stepwise_coefficients.csv`: coefficients from all eight nested specifications.
- `rq1_stepwise_stance_comparison.csv`: raw-stance estimates and incremental changes by stage.
- `rq1_stepwise_model_fit.csv`: fit statistics for all eight specifications.
- `rq1_specification_panel.csv`: grouped add/remove membership, stance estimates, R², adjusted
  R², and reference-relative changes for all 17 specifications.
- `rq1_specification_coefficients.csv`: every coefficient and episode-clustered inferential
  statistic used in the absolute stance-change regression table.
- `rq1_timing_observations.csv`: observation-level audit data.
- `rq1_timing_data_audit.json`: exclusions, hashes, formulas, timing coverage, and versions.
- `rq1_timing_summary.json`: headline estimates and interpretation fields.
- `rq1_specification_panel.pdf` and `.png`: raw-stance regression table for absolute stance
  change.
- `rq1_iceberg_ratio_by_duration.csv`: the binned descriptive statistics used by the duration
  figure.
- `rq1_iceberg_ratio_by_duration.pdf` and `.png`: median iceberg ratio and interquartile range
  across current-turn duration bins.

Attenuation is `100 * (1 - abs(beta_adjusted) / abs(beta_baseline))`. Positive values indicate
attenuation and negative values indicate amplification. Because the retained sample is large,
coefficient movement and confidence intervals should be emphasized over significance alone.

## Tests

```bash
python -m unittest discover \
  -s iclr/rq1_timing_analysis/tests \
  -p 'test_*.py' -v
```
