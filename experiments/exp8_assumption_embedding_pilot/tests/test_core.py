from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..common.candidates import build_anchors, validate_anchor
from ..common.controls import build_control_map
from ..common.metrics import aggregate_rows, clustered_delta_interval, rank_scores
from ..common.splits import assert_show_disjoint, assign_show_splits
from ..common.utils import make_manifest, patch_directory, validate_patch_manifests, write_json


def episode(category: str, show: str, episode_id: str, turn_count: int = 6) -> dict:
    turns = []
    for index in range(turn_count):
        turns.append(
            {
                "turn_id": f"{category}:{episode_id}:{index}",
                "category": category,
                "show_id": show,
                "episode_id": episode_id,
                "turn_idx": index,
                "turn_text": f"Turn {index} from {show}",
                "explicit": [{"text": f"explicit {index}", "confidence": 0.8}],
                "assumptions": [{"text": f"assumption {index}", "confidence": 0.7}],
                "explicit_count": 1,
                "assumption_count": 1,
                "explicit_token_count": 2,
                "assumption_token_count": 2,
            }
        )
    return {
        "episode_key": f"{category}:{episode_id}",
        "category": category,
        "show_id": show,
        "episode_id": episode_id,
        "turns": turns,
    }


class SplitAndCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = [episode("news", f"show-{index}", f"ep-{index}") for index in range(5)]

    def test_show_disjoint_and_has_validation(self) -> None:
        assignment = assign_show_splits(self.episodes, seed=42)
        self.assertIn("validation", set(assignment.values()))
        self.assertIn("test", set(assignment.values()))
        anchors = build_anchors(self.episodes, assignment, candidate_count=8)
        assert_show_disjoint(anchors)
        self.assertEqual(len({anchor["show_id"] for anchor in anchors}), 5)

    def test_history_never_contains_current_or_future(self) -> None:
        assignment = assign_show_splits(self.episodes, seed=42)
        anchors = build_anchors(self.episodes, assignment, candidate_count=8)
        for anchor in anchors:
            validate_anchor(anchor)
            history_indices = [int(value.rsplit(":", 1)[1]) for value in anchor["history_ids"]]
            self.assertTrue(all(index < int(anchor["turn_idx"]) for index in history_indices))
            self.assertNotIn(anchor["anchor_id"], anchor["history_ids"])
            self.assertEqual(anchor["candidate_ids"].count(anchor["target_id"]), 1)

    def test_candidate_sharding_is_deterministic(self) -> None:
        assignment = assign_show_splits(self.episodes, seed=42)
        first = build_anchors(self.episodes, assignment, candidate_count=8)
        second = build_anchors(self.episodes, assignment, candidate_count=8)
        self.assertEqual(first, second)


class ControlAndMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        episodes = [episode("news", "show-a", "ep-a"), episode("news", "show-b", "ep-b")]
        self.turns = [turn for value in episodes for turn in value["turns"]]

    def test_counterfactual_matching(self) -> None:
        source = self.turns[0]
        same_episode = build_control_map(self.turns, "same_episode", {source["turn_id"]})
        donor = next(turn for turn in self.turns if turn["turn_id"] == same_episode[source["turn_id"]])
        self.assertEqual(donor["episode_id"], source["episode_id"])
        self.assertGreaterEqual(abs(donor["turn_idx"] - source["turn_idx"]), 3)
        same_category = build_control_map(self.turns, "same_category", {source["turn_id"]})
        donor = next(turn for turn in self.turns if turn["turn_id"] == same_category[source["turn_id"]])
        self.assertEqual(donor["category"], source["category"])
        self.assertNotEqual(donor["show_id"], source["show_id"])

    def test_metric_aggregation_and_cluster_bootstrap(self) -> None:
        ranked = rank_scores(["a", "b", "c"], np.asarray([0.1, 0.8, 0.2]), "b")
        self.assertEqual(ranked["rank"], 1)
        rows = []
        for anchor, show in (("a", "s1"), ("b", "s2")):
            rows.extend(
                [
                    {"anchor_id": anchor, "show_id": show, "condition": "full", "top1": 1, "top5": 1, "reciprocal_rank": 1.0, "margin": 0.2},
                    {"anchor_id": anchor, "show_id": show, "condition": "base", "top1": 0, "top5": 1, "reciprocal_rank": 0.5, "margin": -0.1},
                ]
            )
        self.assertEqual(len(aggregate_rows(rows)), 2)
        delta = clustered_delta_interval(rows, "full", "base", draws=20, seed=42)
        self.assertAlmostEqual(float(delta["mean_delta"]), 0.5)
        self.assertEqual(delta["n_shows"], 2)


class PatchValidationTests(unittest.TestCase):
    def test_patch_completeness_and_hash_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(2):
                write_json(
                    patch_directory(root, index, 2) / "patch_manifest.json",
                    make_manifest(
                        stage="test",
                        patch_index=index,
                        num_patches=2,
                        row_count=1,
                        input_hash="input",
                        split_hash="split",
                        config={"a": 1},
                    ),
                )
            self.assertEqual(len(validate_patch_manifests(root, "test", 2)), 2)
            (patch_directory(root, 1, 2) / "patch_manifest.json").unlink()
            with self.assertRaises(RuntimeError):
                validate_patch_manifests(root, "test", 2)


if __name__ == "__main__":
    unittest.main()

