# Iceberg Extraction-Saturation Diagnostic

This diagnostic tests whether the original extractor's ten-item ceiling mechanically flattens the
iceberg ratio for long turns. It is intentionally isolated from the production extraction pipeline
and from the downstream RQ1 and experiment code.

## Population and sampling

The script reads every nonempty merged turn in `data/*/parsed/*.json`. It recomputes word count
with the original pipeline's Unicode `\w+` rule, assigns all turns to ten equal-frequency length
strata, and deterministically samples 100 turns from each stratum. Tied word counts are ordered by
a seeded hash of the stable turn ID, so reruns select the same 1,000 turns.

The diagnostic prompt preserves the original extraction definitions and schema but omits both
numerical ten-item instructions. The response parser validates and retains every unique item. The
finite model context and generation-token budget are computational limits, not proposition-count
limits; any response that ends at the token limit fails the run and cannot enter the verdict.

## Run

Install the base and LLM dependencies in the project environment. From the repository root, run:

```bash
sbatch diagnostic/run_iceberg_saturation.sh
```

For an interactive two-GPU allocation, the equivalent command is:

```bash
python -u diagnostic/iceberg_saturation.py \
  --input_dir data \
  --output_dir diagnostic/results \
  --download_dir /shared/4/models \
  --tensor_parallel_size 2 \
  --max_tokens 8192 \
  --max_model_len 32768
```

Raw responses are checkpointed after every batch. A rerun resumes only when the sample, prompt,
model, and decoding configuration have the same run signature. If a response is token-truncated,
increase `MAX_TOKENS`, remove `results/iceberg_saturation_raw_outputs.jsonl`, and rerun.

## Outputs

Review these files under `diagnostic/results/`:

- `iceberg_saturation_turns.csv`: one row per sampled turn, with original, uncapped, and
  synthetically recapped counts and ratios.
- `iceberg_saturation_by_decile.csv`: count distributions, rates at or above 11, and ratio
  summaries by word-length decile.
- `iceberg_saturation_summary.json`: input/model/prompt hashes, extraction audit, category
  composition, and the direct verdict.
- `iceberg_saturation.png` and `.pdf`: count, above-cap-rate, and capped-versus-uncapped panels.

The cap-contribution verdict is positive only when a top-decile output contains at least 11
explicit propositions or implicit assumptions and recapping the same new output at ten changes its
iceberg ratio. This is evidence of a mechanical contribution, not proof that the cap is the only
cause of flattening.

## Tests

```bash
python -m unittest discover -s diagnostic/tests -p 'test_*.py' -v
```
