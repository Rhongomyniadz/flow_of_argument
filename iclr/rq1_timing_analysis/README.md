# RQ1 Timing Analysis

This self-contained ICLR experiment restores the paper's directional RQ1 regression and
tests how its stance coefficients change after adding timing predictors. It does not import
or modify the legacy analysis under `experiments/exp2_iceberg/`.

## Observation contract

The input defaults to `data/stance_labeled/1024`. An observation uses three consecutive raw
records, `t-2`, `t-1`, and `t`. All three must be substantive, nonempty, stance-labeled turns,
and the speaker must change at both boundaries. A backchannel, procedural turn, missing
record, or same-speaker boundary breaks the window; the script never skips it to create a
transition.

For the current and previous turns, timing must be recoverable from `start_time` and
`end_time`, or from one endpoint plus a positive `duration`. Invalid or nonpositive timing is
reported and excluded rather than replaced with an artificial minimum. A negative transition
gap is represented by `overlap=1` and a zero non-overlap gap.

The paper outcome is per-second iceberg density:

```text
D_second(t) = [explicit_count / (assumption_count + 1)] / duration_seconds
```

The response is the current minus previous `log1p` per-second density. Directional stance
movement is calculated from `(stance_t - stance_t-1) / 5`. The preceding boundary is split
into separate lagged agreement and disagreement controls, matching the paper's directional
specification.

## Models

Four OLS specifications use the identical retained sample and episode-clustered standard
errors:

1. `per_second`: the paper's directional per-second density model.
2. `per_second_duration_adjusted`: the paper model plus current-turn duration.
3. `per_second_timing_adjusted`: the paper model plus current duration, pre-turn gap, and
   overlap.
4. `per_second_timing_previous_duration`: a robustness model that also controls for
   previous-turn duration.

The first model restores the paper formula on the stricter timing-complete sample. The second
isolates coefficient movement associated with current-turn duration. The third then shows the
incremental movement associated with pre-turn gap and overlap. None supports a causal claim;
current duration is already part of the outcome denominator and may also be a mediator, so
its adjusted model is a sensitivity analysis rather than a preferred causal specification.

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
and PDF/PNG figures are intended to remain reviewable in git. The observation-level
`rq1_timing_observations.csv` is the only new artifact ignored by the repository-level
`.gitignore`.

## Outputs

- `rq1_timing_coefficients.csv`: all coefficients and episode-clustered inference.
- `rq1_timing_stance_comparison.csv`: paper-facing agreement/disagreement comparison.
- `rq1_timing_model_fit.csv`: sample counts and comparable fit statistics for the shared
  per-second outcome.
- `rq1_timing_observations.csv`: observation-level audit data.
- `rq1_timing_data_audit.json`: exclusions, hashes, formulas, timing coverage, and versions.
- `rq1_timing_summary.json`: headline estimates and timing-adjustment interpretation.
- `rq1_timing_comparison.pdf` and `.png`: coefficient paths from the paper baseline through
  duration and gap/overlap adjustment.

Timing attenuation is
`100 * (1 - abs(beta_adjusted) / abs(beta_paper_baseline))`. Positive values mean attenuation
and negative values mean amplification. The comparison CSV reports both attenuation from the
paper baseline and incremental attenuation from duration-only to the full timing model.
Because the dataset is large, coefficient movement and confidence intervals should be
emphasized over statistical significance alone.

## Tests

```bash
python -m unittest discover \
  -s iclr/rq1_timing_analysis/tests \
  -p 'test_*.py' -v
```
