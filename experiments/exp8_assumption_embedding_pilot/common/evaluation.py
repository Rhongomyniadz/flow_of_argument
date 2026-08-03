from __future__ import annotations

from typing import Any

import numpy as np

from .embeddings import EmbeddingStore, component_vectors
from .metrics import compose, normalize, rank_scores
from .utils import stable_hash


FROZEN_CONDITIONS = (
    "current",
    "current_history",
    "history_explicit",
    "history_assumption",
    "full",
    "shuffled",
)


def query_for_condition(
    anchor: dict[str, Any],
    store: EmbeddingStore,
    condition: str,
    *,
    donor_id: str | None = None,
    explicit_as_assumption: bool = False,
    dimension: int | None = None,
) -> np.ndarray:
    parts = component_vectors(
        anchor,
        store,
        assumption_source_id=donor_id,
        explicit_as_assumption=explicit_as_assumption,
        dimension=dimension,
    )
    current = parts["current"]
    history = parts["history"]
    explicit = parts["explicit"]
    assumption = parts["assumption"]
    history_mask = bool(anchor.get("history_ids"))
    explicit_mask = bool(parts["explicit_mask"])
    assumption_mask = bool(parts["assumption_mask"])
    if condition == "current":
        return normalize(current)
    if condition == "current_history":
        return compose([current, history], [True, history_mask])
    if condition == "history_explicit":
        return compose([current, history, explicit], [True, history_mask, explicit_mask])
    if condition == "history_assumption":
        return compose([current, history, assumption], [True, history_mask, assumption_mask])
    if condition in {"full", "shuffled", "control"}:
        return compose(
            [current, history, explicit, assumption],
            [True, history_mask, explicit_mask, assumption_mask],
        )
    raise ValueError(f"Unknown condition: {condition}")


def evaluate_anchor(
    anchor: dict[str, Any],
    store: EmbeddingStore,
    condition: str,
    *,
    donor_id: str | None = None,
    explicit_as_assumption: bool = False,
    dimension: int | None = None,
) -> dict[str, Any]:
    query = query_for_condition(
        anchor,
        store,
        condition,
        donor_id=donor_id,
        explicit_as_assumption=explicit_as_assumption,
        dimension=dimension,
    )
    candidate_ids = [str(value) for value in anchor["candidate_ids"]]
    documents = []
    for candidate_id in candidate_ids:
        vector = store.get(candidate_id)["document"]
        documents.append(normalize(vector[:dimension] if dimension else vector))
    scores = np.stack(documents) @ normalize(query)
    metrics = rank_scores(candidate_ids, scores, str(anchor["target_id"]))
    return {
        "anchor_id": anchor["anchor_id"],
        "condition": condition,
        "category": anchor["category"],
        "show_id": anchor["show_id"],
        "episode_id": anchor["episode_id"],
        "target_id": anchor["target_id"],
        "candidate_count": len(candidate_ids),
        "candidate_pool_hash": stable_hash(candidate_ids),
        **metrics,
    }
