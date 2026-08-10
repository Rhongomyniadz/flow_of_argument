from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import deduplicate_data as cleaner


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def statements(prefix: str, count: int, offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "text": f"{prefix} statement {index}",
            "confidence": offset + index / 100,
        }
        for index in range(count)
    ]


def turn(
    index: int,
    speaker: str,
    count: int,
    *,
    explicit: list[dict[str, object]] | None = None,
    assumptions: list[dict[str, object]] | None = None,
    move: str = "Assert / Elaborate",
) -> dict[str, object]:
    text = words(f"t{index}_", count)
    return {
        "turn_idx": index,
        "speaker_id": speaker,
        "speaker": speaker,
        "turn_text": text,
        "transcript": text,
        "start_time": float(index * 10),
        "end_time": float(index * 10 + 5),
        "duration": 5.0,
        "wordCount": count,
        "word_count": count,
        "explicit_propositions": explicit or [],
        "assumptions": assumptions or [],
        "conversation_move_label": move,
    }


class CleanEpisodeTests(unittest.TestCase):
    def test_filter_merge_rank_cap_and_abab(self) -> None:
        episode = [
            turn(0, "A", 50, explicit=statements("first", 12)),
            turn(1, "B", 49),  # deleted; its neighboring A turns then merge
            turn(
                2,
                "A",
                55,
                explicit=statements("second", 4, offset=1.0),
                assumptions=statements("assumption_a", 7),
                move="Answer",
            ),
            turn(3, "B", 50, assumptions=statements("assumption_b1", 7)),
            turn(4, "B", 51, assumptions=statements("assumption_b2", 7)),
            turn(5, "A", 50),
        ]

        cleaned = cleaner.clean_episode(episode, min_words=50, max_statements=10)
        self.assertIsNotNone(cleaned)
        assert cleaned is not None
        self.assertEqual([cleaner.speaker_value(row) for row in cleaned.turns], ["A", "B", "A"])
        self.assertEqual(cleaned.stats.short_turns_removed, 1)
        self.assertEqual(cleaned.stats.merge_groups, 2)
        self.assertEqual(cleaned.stats.turns_absorbed_by_merging, 2)
        self.assertEqual([row["turn_idx"] for row in cleaned.turns], [0, 1, 2])
        self.assertEqual(cleaned.turns[0]["conversation_move_label"], "Answer")
        self.assertEqual(cleaned.turns[0]["merged_from_turn_indices"], [0, 2])
        self.assertEqual(cleaned.turns[1]["merged_from_turn_indices"], [3, 4])
        self.assertEqual(cleaned.turns[2]["merged_from_turn_indices"], [5])
        self.assertEqual(len(cleaned.turns[0]["explicit_propositions"]), 10)
        self.assertEqual(len(cleaned.turns[1]["assumptions"]), 10)
        explicit_confidences = [
            item["confidence"] for item in cleaned.turns[0]["explicit_propositions"]
        ]
        self.assertEqual(explicit_confidences, sorted(explicit_confidences, reverse=True))
        self.assertTrue(all(cleaner.word_count(cleaner.turn_text(row)) >= 50 for row in cleaned.turns))

    def test_rejects_non_two_speaker_episode(self) -> None:
        episode = [turn(0, "A", 50), turn(1, "B", 50), turn(2, "C", 50)]
        with self.assertRaisesRegex(cleaner.EpisodeRejected, "Found 3 speaker"):
            cleaner.clean_episode(episode, min_words=50, max_statements=10)

    def test_object_root_is_preserved(self) -> None:
        source = {
            "episode_id": "object-episode",
            "metadata": {"source": "fixture"},
            "turns": [turn(0, "A", 50), turn(1, "B", 50)],
        }
        cleaned = cleaner.clean_episode(source, min_words=50, max_statements=10)
        self.assertIsNotNone(cleaned)
        assert cleaned is not None
        serialized = cleaner.serialize_episode(cleaned)
        self.assertEqual(serialized["episode_id"], "object-episode")
        self.assertEqual(serialized["metadata"], {"source": "fixture"})
        self.assertEqual(len(serialized["turns"]), 2)
        self.assertEqual(
            [row["merged_from_turn_indices"] for row in serialized["turns"]],
            [[0], [1]],
        )


class EndToEndTests(unittest.TestCase):
    def test_duplicate_files_keep_canonical_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "data"
            output_root = root / "cleaned"
            category = input_root / "conversation_moves_labeled" / "commentary"
            category.mkdir(parents=True)

            base = [
                turn(0, "A", 50, assumptions=statements("base", 3)),
                turn(1, "B", 50),
                turn(2, "A", 50),
            ]
            duplicate = json.loads(json.dumps(base))
            duplicate[0]["assumptions"] = statements("different_sampling", 8)
            (category / "episode.json").write_text(
                json.dumps(base), encoding="utf-8"
            )
            (category / "episode_2.json").write_text(
                json.dumps(duplicate), encoding="utf-8"
            )
            # A non-episode derivative artifact must be audited but not copied.
            pairs = input_root / "implicature_flow" / "entailment_pairs_1to10"
            pairs.mkdir(parents=True)
            (pairs / "episode.json").write_text(
                json.dumps({"episode_id": "episode", "pairs": []}),
                encoding="utf-8",
            )

            args = argparse.Namespace(
                input_root=input_root,
                output_root=output_root,
                min_words=50,
                max_statements=10,
                overwrite=False,
                verbose=False,
            )
            manifest = cleaner.run(args)

            self.assertTrue(
                (output_root / "conversation_moves_labeled" / "commentary" / "episode.json").is_file()
            )
            self.assertFalse(
                (output_root / "conversation_moves_labeled" / "commentary" / "episode_2.json").exists()
            )
            self.assertEqual(manifest["status_counts"]["written"], 1)
            self.assertEqual(
                manifest["status_counts"]["duplicate_episode_removed"], 1
            )
            self.assertEqual(manifest["status_counts"]["non_episode_json"], 1)
            audit_rows = [
                json.loads(line)
                for line in (output_root / cleaner.AUDIT_NAME).read_text(encoding="utf-8").splitlines()
            ]
            duplicate_row = next(
                row for row in audit_rows if row["status"] == "duplicate_episode_removed"
            )
            self.assertEqual(
                duplicate_row["duplicate_of"],
                "conversation_moves_labeled/commentary/episode.json",
            )


if __name__ == "__main__":
    unittest.main()
