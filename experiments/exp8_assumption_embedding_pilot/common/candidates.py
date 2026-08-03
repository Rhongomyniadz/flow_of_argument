from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .utils import stable_hash


def build_anchors(episodes: list[dict[str, Any]], splits: dict[str, str], candidate_count: int = 25) -> list[dict[str, Any]]:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    all_turns = [turn for episode in episodes for turn in episode["turns"]]
    lookup = {turn["turn_id"]: turn for turn in all_turns}
    by_episode: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    global_ids: list[str] = []
    for turn in all_turns:
        by_episode[str(turn["episode_id"])].append(turn["turn_id"])
        by_category[str(turn["category"])].append(turn["turn_id"])
        global_ids.append(turn["turn_id"])

    anchors: list[dict[str, Any]] = []
    def sample_pool(
        pool: list[str],
        *,
        rng: np.random.Generator,
        used: set[str],
        limit: int,
        predicate: Any | None = None,
    ) -> list[str]:
        if not pool or limit < 1:
            return []
        selected: list[str] = []
        attempts = min(max(limit * 50, 200), max(len(pool) * 2, 200))
        for _ in range(attempts):
            turn_id = pool[int(rng.integers(0, len(pool)))]
            if turn_id in used or (predicate is not None and not predicate(turn_id)):
                continue
            used.add(turn_id)
            selected.append(turn_id)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            for turn_id in pool:
                if turn_id in used or (predicate is not None and not predicate(turn_id)):
                    continue
                used.add(turn_id)
                selected.append(turn_id)
                if len(selected) >= limit:
                    break
        return selected

    for episode in episodes:
        turns = episode["turns"]
        for position, turn in enumerate(turns[:-1]):
            target = turns[position + 1]
            if not turn["turn_text"] or not target["turn_text"]:
                continue
            anchor_id = turn["turn_id"]
            rng = np.random.default_rng(int(stable_hash(anchor_id, 16), 16))
            used = {anchor_id, target["turn_id"]}
            candidates: list[str] = [target["turn_id"]]

            same_episode = [
                turn_id
                for turn_id in by_episode[str(turn["episode_id"])]
                if turn_id not in used and abs(int(lookup[turn_id]["turn_idx"]) - int(turn["turn_idx"])) >= 3
            ]
            rng.shuffle(same_episode)
            for turn_id in same_episode[:8]:
                candidates.append(turn_id)
                used.add(turn_id)

            candidates.extend(
                sample_pool(
                    by_category[str(turn["category"])],
                    rng=rng,
                    used=used,
                    limit=max(0, min(candidate_count, 17) - len(candidates)),
                    predicate=lambda turn_id: lookup[turn_id]["show_id"] != turn["show_id"],
                )
            )
            candidates.extend(
                sample_pool(
                    global_ids,
                    rng=rng,
                    used=used,
                    limit=max(0, candidate_count - len(candidates)),
                )
            )
            if len(candidates) < 2:
                continue
            rng.shuffle(candidates)
            if candidates.count(target["turn_id"]) != 1 or len(candidates) != len(set(candidates)):
                raise AssertionError(f"Invalid candidate pool for {anchor_id}")
            history_ids = [item["turn_id"] for item in turns[max(0, position - 3):position]]
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "target_id": target["turn_id"],
                    "candidate_ids": candidates,
                    "history_ids": history_ids,
                    "category": turn["category"],
                    "show_id": turn["show_id"],
                    "episode_id": turn["episode_id"],
                    "turn_idx": turn["turn_idx"],
                    "split": splits[str(turn["show_id"])],
                    "assumption_count": turn["assumption_count"],
                    "assumption_token_count": turn["assumption_token_count"],
                }
            )
    return anchors


def validate_anchor(anchor: dict[str, Any]) -> None:
    candidates = list(anchor["candidate_ids"])
    target = anchor["target_id"]
    if candidates.count(target) != 1:
        raise ValueError(f"Anchor {anchor['anchor_id']} has {candidates.count(target)} true candidates")
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"Anchor {anchor['anchor_id']} has duplicate candidates")
    if anchor["anchor_id"] in candidates:
        raise ValueError(f"Anchor {anchor['anchor_id']} appears in its own candidates")
