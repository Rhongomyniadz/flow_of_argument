from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FONT_SCALE: float = 1.45
TITLE_SIZE: int = 20
LABEL_SIZE: int = 17
TICK_SIZE: int = 14
LEGEND_SIZE: int = 14

sns.set_theme(style="whitegrid", context="paper", font_scale=FONT_SCALE)
plt.rcParams.update(
    {
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
    }
)


class PlotOutputPaths(TypedDict):
    rank_distribution: Path
    headline_metrics: Path
    rank_lift_by_move: Path
    score_lift_by_move: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--move_min_pair_count", type=int, required=True)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required plot input is missing: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def read_json(path: Path) -> dict[str, object]:
    with require_file(path).open("r", encoding="utf-8") as handle:
        payload: object = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing: list[str] = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def coerce_bool_series(series: pd.Series, label: str) -> pd.Series:
    text_values: pd.Series = series.astype(str).str.strip().str.lower()
    valid_mask: pd.Series = text_values.isin({"true", "false"})
    if not bool(valid_mask.all()):
        invalid_values: list[str] = sorted(text_values[~valid_mask].astype(str).unique().tolist())
        raise ValueError(f"{label} must contain only boolean values. Invalid values: {invalid_values}")
    return text_values == "true"


def ci_errorbar_values(
    df: pd.DataFrame,
    mean_column: str,
    low_column: str,
    high_column: str,
    label: str,
) -> np.ndarray:
    means: np.ndarray = df[mean_column].astype(float).to_numpy()
    lows: np.ndarray = df[low_column].astype(float).to_numpy()
    highs: np.ndarray = df[high_column].astype(float).to_numpy()
    errors: np.ndarray = np.vstack([means - lows, highs - means])
    if not bool(np.all(np.isfinite(errors))):
        raise ValueError(f"{label} has non-finite confidence interval values.")
    if bool(np.any(errors < -1e-9)):
        raise ValueError(f"{label} has confidence interval bounds that do not contain the mean.")
    return np.maximum(errors, 0.0)


def build_output_paths(results_dir: Path, output_prefix: str) -> PlotOutputPaths:
    return {
        "rank_distribution": results_dir / f"{output_prefix}_rank_distribution.pdf",
        "headline_metrics": results_dir / f"{output_prefix}_headline_metrics.pdf",
        "rank_lift_by_move": results_dir / f"{output_prefix}_rank_lift_by_move.pdf",
        "score_lift_by_move": results_dir / f"{output_prefix}_score_lift_by_move.pdf",
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rank_distribution(pair_df: pd.DataFrame, output_path: Path) -> None:
    retained_mask: pd.Series = coerce_bool_series(pair_df["canonical_retained"], "Exp1 pair canonical_retained")
    retained: pd.DataFrame = pair_df[retained_mask].copy()
    if retained.empty:
        raise ValueError("Exp1 pair data has no retained pairs to plot.")

    rank_rows: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(
                {
                    "condition": "Without assumptions",
                    "rank": retained["true_rank_without_assumptions"],
                }
            ),
            pd.DataFrame(
                {
                    "condition": "With assumptions",
                    "rank": retained["true_rank_with_assumptions"],
                }
            ),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    sns.histplot(data=rank_rows, x="rank", hue="condition", multiple="dodge", discrete=True, shrink=0.82, ax=ax)
    ax.set_xlabel("True Turn Rank")
    ax.set_ylabel("Retained Pair Count")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_headline_metrics(exp1_summary: dict[str, object], output_path: Path) -> None:
    headline_metrics_raw: object = exp1_summary.get("headline_metrics")
    if not isinstance(headline_metrics_raw, dict):
        raise ValueError("Exp1 summary is missing headline_metrics.")
    headline_metrics: dict[str, object] = headline_metrics_raw
    headline_rows: list[dict[str, str | float]] = [
        {
            "metric": "Top-1",
            "condition": "Without assumptions",
            "value": float(headline_metrics["top1_rate_without_assumptions"]),
        },
        {
            "metric": "Top-1",
            "condition": "With assumptions",
            "value": float(headline_metrics["top1_rate_with_assumptions"]),
        },
        {
            "metric": "MRR",
            "condition": "Without assumptions",
            "value": float(headline_metrics["mrr_without_assumptions"]),
        },
        {
            "metric": "MRR",
            "condition": "With assumptions",
            "value": float(headline_metrics["mrr_with_assumptions"]),
        },
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    sns.barplot(
        data=pd.DataFrame(headline_rows),
        x="metric",
        y="value",
        hue="condition",
        palette=["#4C78A8", "#F58518"],
        ax=ax,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, output_path)


def build_move_plot(move_summary: pd.DataFrame, move_min_pair_count: int) -> pd.DataFrame:
    move_plot: pd.DataFrame = move_summary[move_summary["pair_count"].astype(int) >= move_min_pair_count].copy()
    if move_plot.empty:
        raise ValueError(f"No Exp1 move rows meet the minimum pair count: {move_min_pair_count}")
    return move_plot


def plot_rank_lift_by_move(move_plot: pd.DataFrame, output_path: Path) -> None:
    rank_move_plot: pd.DataFrame = move_plot.sort_values("mean_rank_lift", ascending=True).reset_index(drop=True)
    rank_positions: np.ndarray = np.arange(len(rank_move_plot), dtype=float)
    rank_values: np.ndarray = rank_move_plot["mean_rank_lift"].astype(float).to_numpy()
    rank_labels: list[str] = [
        f"{move_label} (n={pair_count:,})"
        for move_label, pair_count in zip(
            rank_move_plot["true_next_turn_move_label"].astype(str),
            rank_move_plot["pair_count"].astype(int),
            strict=True,
        )
    ]
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    ax.errorbar(
        rank_values,
        rank_positions,
        xerr=ci_errorbar_values(
            rank_move_plot,
            "mean_rank_lift",
            "mean_rank_lift_ci95_low",
            "mean_rank_lift_ci95_high",
            "Exp1 move rank lift",
        ),
        fmt="none",
        ecolor="#6c757d",
        elinewidth=1.2,
        capsize=3,
    )
    ax.scatter(rank_values, rank_positions, color=np.where(rank_values >= 0.0, "#1b9e77", "#d95f02"), s=58, zorder=3)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(rank_positions)
    ax.set_yticklabels(rank_labels)
    ax.set_xlabel("Mean rank lift (positive = assumptions improve rank)")
    ax.set_ylabel("True next-turn move")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_score_lift_by_move(move_plot: pd.DataFrame, output_path: Path) -> None:
    score_move_plot: pd.DataFrame = move_plot.sort_values("mean_score_lift", ascending=True).reset_index(drop=True)
    score_positions: np.ndarray = np.arange(len(score_move_plot), dtype=float)
    score_values: np.ndarray = score_move_plot["mean_score_lift"].astype(float).to_numpy()
    score_labels: list[str] = [
        f"{move_label} (n={pair_count:,})"
        for move_label, pair_count in zip(
            score_move_plot["true_next_turn_move_label"].astype(str),
            score_move_plot["pair_count"].astype(int),
            strict=True,
        )
    ]
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    ax.errorbar(
        score_values,
        score_positions,
        xerr=ci_errorbar_values(
            score_move_plot,
            "mean_score_lift",
            "mean_score_lift_ci95_low",
            "mean_score_lift_ci95_high",
            "Exp1 move score lift",
        ),
        fmt="none",
        ecolor="#6c757d",
        elinewidth=1.2,
        capsize=3,
    )
    ax.scatter(score_values, score_positions, color="#4C78A8", s=58, zorder=3)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(score_positions)
    ax.set_yticklabels(score_labels)
    ax.set_xlabel("Mean score lift")
    ax.set_ylabel("True next-turn move")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_exp1_results(results_dir: Path, output_prefix: str, move_min_pair_count: int) -> PlotOutputPaths:
    exp1_summary: dict[str, object] = read_json(results_dir / "exp1_summary.json")
    pair_df: pd.DataFrame = read_csv(results_dir / "exp1_llm_next_turn_pairs.csv")
    category_summary: pd.DataFrame = read_csv(results_dir / "exp1_llm_next_turn_by_category.csv")
    move_summary: pd.DataFrame = read_csv(results_dir / "exp1_llm_next_turn_by_move.csv")
    require_columns(
        pair_df,
        ["canonical_retained", "true_rank_without_assumptions", "true_rank_with_assumptions"],
        "Exp1 pair data",
    )
    require_columns(
        category_summary,
        [
            "category",
            "pair_count",
            "mean_rank_lift",
            "top1_rate_without_assumptions",
            "top1_rate_with_assumptions",
            "mrr_without_assumptions",
            "mrr_with_assumptions",
        ],
        "Exp1 category data",
    )
    require_columns(
        move_summary,
        [
            "true_next_turn_move_label",
            "pair_count",
            "mean_rank_lift",
            "mean_rank_lift_ci95_low",
            "mean_rank_lift_ci95_high",
            "mean_score_lift",
            "mean_score_lift_ci95_low",
            "mean_score_lift_ci95_high",
        ],
        "Exp1 move data",
    )
    output_paths: PlotOutputPaths = build_output_paths(results_dir, output_prefix)
    move_plot: pd.DataFrame = build_move_plot(move_summary, move_min_pair_count)
    plot_rank_distribution(pair_df, output_paths["rank_distribution"])
    plot_headline_metrics(exp1_summary, output_paths["headline_metrics"])
    plot_rank_lift_by_move(move_plot, output_paths["rank_lift_by_move"])
    plot_score_lift_by_move(move_plot, output_paths["score_lift_by_move"])
    return output_paths


def main() -> None:
    args: argparse.Namespace = parse_args()
    output_paths: PlotOutputPaths = plot_exp1_results(
        results_dir=args.results_dir,
        output_prefix=args.output_prefix,
        move_min_pair_count=args.move_min_pair_count,
    )
    print(json.dumps({name: str(path) for name, path in output_paths.items()}, indent=2))


if __name__ == "__main__":
    main()
