from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left
from typing import Any

from .utils import stable_hash


def _distance(source: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, int, str]:
    return (
        abs(int(source.get("assumption_count", 0)) - int(candidate.get("assumption_count", 0))),
        abs(int(source.get("assumption_token_count", 0)) - int(candidate.get("assumption_token_count", 0))),
        str(candidate["turn_id"]),
    )


def build_control_map(
    turns: list[dict[str, Any]],
    control_type: str,
    source_ids: set[str] | None = None,
) -> dict[str, str]:
    if control_type not in {"same_episode", "same_category", "explicit_matched"}:
        raise ValueError(f"Unknown control type: {control_type}")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        by_episode[str(turn["episode_id"])].append(turn)
        by_category[str(turn["category"])].append(turn)
    for rows in by_category.values():
        rows.sort(key=lambda row: (int(row.get("assumption_token_count", 0)), str(row["turn_id"])))
    category_tokens = {
        category: [int(row.get("assumption_token_count", 0)) for row in rows]
        for category, rows in by_category.items()
    }
    mapping: dict[str, str] = {}
    sources = turns if source_ids is None else [row for row in turns if str(row["turn_id"]) in source_ids]
    for source in sources:
        if control_type == "explicit_matched":
            if int(source.get("explicit_count", 0)) > 0:
                mapping[source["turn_id"]] = source["turn_id"]
            continue
        if control_type == "same_episode":
            candidates = [
                row for row in by_episode[str(source["episode_id"])]
                if row["turn_id"] != source["turn_id"]
                and abs(int(row["turn_idx"]) - int(source["turn_idx"])) >= 3
                and int(row.get("assumption_count", 0)) > 0
            ]
        else:
            rows = by_category[str(source["category"])]
            tokens = category_tokens[str(source["category"])]
            center = bisect_left(tokens, int(source.get("assumption_token_count", 0)))
            radius = 64
            candidates = [
                row for row in rows[max(0, center - radius):min(len(rows), center + radius)]
                if row["show_id"] != source["show_id"] and int(row.get("assumption_count", 0)) > 0
            ]
            if not candidates:
                candidates = [
                    row for row in rows
                    if row["show_id"] != source["show_id"] and int(row.get("assumption_count", 0)) > 0
                ][:128]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (_distance(source, row), stable_hash({"source": source["turn_id"], "candidate": row["turn_id"]})))
        mapping[source["turn_id"]] = candidates[0]["turn_id"]
    return mapping
