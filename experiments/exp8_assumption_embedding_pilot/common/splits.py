from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .utils import stable_hash


def assign_show_splits(episodes: list[dict[str, Any]], seed: int) -> dict[str, str]:
    by_category: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        by_category[str(episode["category"])].add(str(episode["show_id"]))

    assignments: dict[str, str] = {}
    for category in sorted(by_category):
        unresolved = [show for show in sorted(by_category[category]) if show not in assignments]
        unresolved.sort(key=lambda show: stable_hash({"seed": seed, "category": category, "show": show}))
        count = len(unresolved)
        if count == 1:
            cuts = (1, 1)
        elif count == 2:
            cuts = (1, 2)
        else:
            # Keep at least one validation and one test show, including the
            # common three-show pilot case.
            train_end = min(max(1, int(np.floor(count * 0.8))), count - 2)
            validation_end = count - 1
            cuts = (train_end, validation_end)
        for index, show in enumerate(unresolved):
            if index < cuts[0]:
                split = "train"
            elif index < cuts[1]:
                split = "validation"
            else:
                split = "test"
            assignments[show] = split
    return assignments


def assert_show_disjoint(rows: list[dict[str, Any]]) -> None:
    observed: dict[str, str] = {}
    for row in rows:
        show = str(row["show_id"])
        split = str(row["split"])
        previous = observed.setdefault(show, split)
        if previous != split:
            raise ValueError(f"Show {show} appears in both {previous} and {split}")


def balanced_anchor_sample(
    anchors: list[dict[str, Any]],
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [row for row in anchors if row["split"] == split]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_category[str(row["category"])].append(row)
    rng = np.random.default_rng(seed)
    for rows in by_category.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    categories = sorted(by_category)
    while len(selected) < min(limit, len(eligible)):
        progressed = False
        for category in categories:
            if by_category[category]:
                selected.append(by_category[category].pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda row: row["anchor_id"])
