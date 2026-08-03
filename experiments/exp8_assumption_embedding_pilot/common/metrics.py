from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def compose(vectors: list[np.ndarray], masks: list[bool] | None = None) -> np.ndarray:
    if masks is None:
        masks = [True] * len(vectors)
    selected = [normalize(vector) for vector, use in zip(vectors, masks) if use]
    if not selected:
        return np.zeros_like(np.asarray(vectors[0], dtype=np.float32))
    return normalize(np.mean(np.stack(selected), axis=0))


def rank_scores(candidate_ids: list[str], scores: np.ndarray, target_id: str) -> dict[str, float | int]:
    if candidate_ids.count(target_id) != 1:
        raise ValueError(f"Expected exactly one target {target_id}")
    array = np.asarray(scores, dtype=float)
    if array.shape != (len(candidate_ids),):
        raise ValueError("Score shape does not match candidate IDs")
    target_index = candidate_ids.index(target_id)
    order = sorted(range(len(candidate_ids)), key=lambda index: (-array[index], candidate_ids[index]))
    rank = order.index(target_index) + 1
    hardest_negative = max(array[index] for index in range(len(candidate_ids)) if index != target_index)
    return {
        "rank": rank,
        "reciprocal_rank": 1.0 / rank,
        "top1": int(rank == 1),
        "top5": int(rank <= 5),
        "margin": float(array[target_index] - hardest_negative),
    }


def aggregate_rows(rows: list[dict[str, Any]], condition_key: str = "condition") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[condition_key])].append(row)
    output: list[dict[str, Any]] = []
    for condition, group in sorted(groups.items()):
        output.append(
            {
                condition_key: condition,
                "n": len(group),
                "recall_at_1": float(np.mean([float(row["top1"]) for row in group])),
                "recall_at_5": float(np.mean([float(row["top5"]) for row in group])),
                "mrr": float(np.mean([float(row["reciprocal_rank"]) for row in group])),
                "mean_margin": float(np.mean([float(row["margin"]) for row in group])),
            }
        )
    return output


def clustered_delta_interval(
    rows: list[dict[str, Any]],
    first: str,
    second: str,
    metric: str = "reciprocal_rank",
    draws: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[str(row["anchor_id"])][str(row["condition"])] = row
    deltas_by_show: dict[str, list[float]] = defaultdict(list)
    for condition_rows in paired.values():
        if first in condition_rows and second in condition_rows:
            left = condition_rows[first]
            right = condition_rows[second]
            deltas_by_show[str(left["show_id"])].append(float(left[metric]) - float(right[metric]))
    if not deltas_by_show:
        return {"first": first, "second": second, "n_shows": 0, "mean_delta": float("nan")}
    show_ids = np.asarray(sorted(deltas_by_show), dtype=object)
    show_means = np.asarray([np.mean(deltas_by_show[show]) for show in show_ids], dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(1, draws)):
        indices = rng.integers(0, len(show_means), size=len(show_means))
        samples.append(float(np.mean(show_means[indices])))
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "n_shows": len(show_ids),
        "mean_delta": float(np.mean(show_means)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(np.asarray(samples) > 0)),
    }

