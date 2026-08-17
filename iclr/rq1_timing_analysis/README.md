# RQ1 Timing Analysis

This self-contained ICLR experiment tests whether the local relationship between stance
movement and explicit-to-implicit density remains after removing duration from the outcome
and adding timing as predictors. It does not import or modify the legacy analysis under
`experiments/exp2_iceberg/`.

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

The two densities are:

```text
D_second(t) = [explicit_count / (assumption_count + 1)] / duration_seconds
D_token(t)  = [explicit_count / (assumption_count + 1)] / word_count
```

Each response is the current minus previous `log1p` density. Directional stance movement is
calculated from `(stance_t - stance_t-1) / 5`, with the corresponding signed change for the
preceding boundary retained as a lag control.

## Models

Five OLS specifications use the identical retained sample and episode-clustered standard
errors:

1. `per_second`: the original duration-dependent outcome.
2. `token_normalized`: the duration-free outcome.
3. `token_duration_adjusted`: token normalization plus current-turn duration only.
4. `token_timing_adjusted`: token normalization plus current duration, pre-turn gap, and
   overlap.
5. `token_timing_previous_duration`: a robustness model that also controls for previous-turn
   duration.

The first model estimates a combined density-pacing association. The second removes duration
from the outcome. The third isolates the change associated with current-turn duration. The
fourth then shows the incremental change associated with pre-turn gap and overlap. These are
different estimands, and none supports a causal claim; duration may be a mediator rather than
a conventional confound.

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
- `rq1_timing_model_fit.csv`: sample counts and fit statistics; AIC/BIC are only comparable
  among rows marked as sharing the token outcome.
- `rq1_timing_observations.csv`: observation-level audit data.
- `rq1_timing_data_audit.json`: exclusions, hashes, formulas, timing coverage, and versions.
- `rq1_timing_summary.json`: headline estimates and timing-adjustment interpretation.
- `rq1_timing_comparison.pdf` and `.png`: coefficient paths across the four headline models,
  with duration separated from the incremental gap/overlap adjustment.

Timing attenuation is
`100 * (1 - abs(beta_timing) / abs(beta_token))`. Positive values mean attenuation and
negative values mean amplification. The comparison CSV reports both attenuation from the
token baseline and incremental attenuation from duration-only to the full timing model.
Because the dataset is large, coefficient movement and confidence intervals should be
emphasized over statistical significance alone.

## Tests

```bash
python -m unittest discover \
  -s iclr/rq1_timing_analysis/tests \
  -p 'test_*.py' -v
```
