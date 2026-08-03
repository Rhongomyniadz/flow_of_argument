from __future__ import annotations

"""Stage 02: audit the existing Exp1 pair table."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.utils import file_hash, read_json, runtime_versions, stable_hash, write_json
from ..common.progress import run_single


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit pair-level Exp1 assumption effects.")
    value.add_argument("--pairs-csv", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp01_results"))
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--bootstrap-draws", type=int, default=1000)
    value.add_argument("--force", action="store_true")
    return value


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Exp1 columns: {missing}")


def run(args: argparse.Namespace) -> None:
    input_hash = file_hash(args.pairs_csv)
    config = {"pairs_csv": str(args.pairs_csv), "seed": args.seed, "bootstrap_draws": args.bootstrap_draws}
    config_hash = stable_hash(config)
    summary_path = args.output_dir / "summary.json"
    config_path = args.output_dir / "config.json"
    if not args.force and summary_path.exists() and config_path.exists():
        observed_summary = read_json(summary_path)
        observed_config = read_json(config_path)
        if observed_summary.get("input_hash") == input_hash and observed_config.get("config_hash") == config_hash:
            return
    frame = pd.read_csv(args.pairs_csv)
    require_columns(
        frame,
        [
            "episode_id",
            "reciprocal_rank_without_assumptions",
            "reciprocal_rank_with_assumptions",
            "top1_without_assumptions",
            "top1_with_assumptions",
        ],
    )
    frame["mrr_delta"] = (
        pd.to_numeric(frame["reciprocal_rank_with_assumptions"], errors="coerce")
        - pd.to_numeric(frame["reciprocal_rank_without_assumptions"], errors="coerce")
    )
    frame["top1_delta"] = (
        pd.to_numeric(frame["top1_with_assumptions"], errors="coerce")
        - pd.to_numeric(frame["top1_without_assumptions"], errors="coerce")
    )
    frame = frame.dropna(subset=["mrr_delta", "top1_delta"]).copy()
    if frame.empty:
        raise RuntimeError("Exp01 has no valid numeric pair rows after filtering")
    episode = frame.groupby("episode_id", as_index=False).agg(
        pair_count=("mrr_delta", "size"),
        mean_mrr_delta=("mrr_delta", "mean"),
        mean_top1_delta=("top1_delta", "mean"),
    )
    rng = np.random.default_rng(args.seed)
    episode_values = episode["mean_mrr_delta"].to_numpy(dtype=float)
    draws = [float(np.mean(rng.choice(episode_values, size=len(episode_values), replace=True))) for _ in range(args.bootstrap_draws)]
    group_columns = [
        column for column in ("category", "true_next_turn_move_label", "assumption_count", "negative_source")
        if column in frame.columns
    ]
    metric_rows: list[dict[str, Any]] = [
        {
            "group": "overall",
            "value": "all",
            "n_pairs": len(frame),
            "n_episodes": episode["episode_id"].nunique(),
            "mean_mrr_delta": float(frame["mrr_delta"].mean()),
            "mean_top1_delta": float(frame["top1_delta"].mean()),
            "positive_episode_rate": float((episode["mean_mrr_delta"] > 0).mean()),
        }
    ]
    for column in group_columns:
        for group, rows in frame.groupby(column, dropna=False):
            metric_rows.append(
                {
                    "group": column,
                    "value": str(group),
                    "n_pairs": len(rows),
                    "n_episodes": rows["episode_id"].nunique(),
                    "mean_mrr_delta": float(rows["mrr_delta"].mean()),
                    "mean_top1_delta": float(rows["top1_delta"].mean()),
                    "positive_episode_rate": float("nan"),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    episode.to_csv(args.output_dir / "episode_metrics.csv", index=False)
    write_json(args.output_dir / "config.json", {**config, "config_hash": config_hash})
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp01_existing_result_audit",
            "status": "complete",
            "input_hash": input_hash,
            "pair_count": len(frame),
            "episode_count": len(episode),
            "mean_mrr_delta": float(frame["mrr_delta"].mean()),
            "mean_top1_delta": float(frame["top1_delta"].mean()),
            "positive_episode_rate": float((episode["mean_mrr_delta"] > 0).mean()),
            "mrr_delta_ci95_low": float(np.quantile(draws, 0.025)),
            "mrr_delta_ci95_high": float(np.quantile(draws, 0.975)),
            "runtime": runtime_versions(),
        },
    )


def main() -> None:
    args = parser().parse_args()
    run_single(lambda: run(args), "stage 02 audit")


if __name__ == "__main__":
    main()
