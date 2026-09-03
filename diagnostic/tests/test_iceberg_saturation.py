from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from diagnostic import iceberg_saturation as saturation


def source_turn(index: int, word_count: int, prefix: str) -> saturation.SourceTurn:
    return saturation.SourceTurn(
        turn_id=f"{prefix}:episode:{index}",
        category=prefix,
        episode_id="episode",
        turn_idx=index,
        source_path=f"{prefix}/parsed/episode.json",
        turn_text=f"turn {index}",
        word_count=word_count,
        original_explicit_count=10,
        original_implicit_count=10,
    )


def extraction_items(prefix: str, count: int) -> list[dict[str, object]]:
    return [
        {"text": f"{prefix} {index}", "confidence": 1.0 - index / 100.0}
        for index in range(count)
    ]


class SamplingTests(unittest.TestCase):
    def test_samples_exactly_one_hundred_turns_per_decile_deterministically(self) -> None:
        turns = [source_turn(index, index + 1, "business") for index in range(2_000)]
        assignments = saturation.assign_length_deciles(turns, 10, 42)
        first = saturation.sample_from_deciles(assignments, 10, 100, 42)
        second = saturation.sample_from_deciles(assignments, 10, 100, 42)

        self.assertEqual(len(first), 1_000)
        self.assertEqual(
            Counter(turn.length_decile for turn in first),
            Counter({decile: 100 for decile in range(1, 11)}),
        )
        self.assertEqual(
            [turn.source.turn_id for turn in first],
            [turn.source.turn_id for turn in second],
        )

    def test_tied_lengths_are_split_into_balanced_deterministic_deciles(self) -> None:
        turns = [source_turn(index, 5, "commentary") for index in range(103)]
        first = saturation.assign_length_deciles(turns, 10, 42)
        second = saturation.assign_length_deciles(list(reversed(turns)), 10, 42)
        counts = Counter(turn.length_decile for turn in first)

        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(
            [(turn.source.turn_id, turn.length_decile) for turn in first],
            [(turn.source.turn_id, turn.length_decile) for turn in second],
        )

    def test_duplicate_turn_ids_fail_before_sampling(self) -> None:
        duplicate = source_turn(1, 10, "news")
        with self.assertRaisesRegex(saturation.InputValidationError, "Duplicate"):
            saturation.ensure_unique_turn_ids([duplicate, duplicate])


class ParsingTests(unittest.TestCase):
    def test_uncapped_parser_retains_more_than_ten_items_and_deduplicates(self) -> None:
        explicit = extraction_items("Explicit", 12)
        assumptions = extraction_items("Implicit", 12)
        explicit.append({"text": "Explicit 0", "confidence": 0.1})
        assumptions.append({"text": "Explicit 1", "confidence": 0.5})
        raw = json.dumps(
            {
                "explicit_propositions": explicit,
                "assumptions": assumptions,
            }
        )

        parsed = saturation.parse_uncapped_extraction(raw, "test response")

        self.assertEqual(len(parsed.explicit_propositions), 12)
        self.assertEqual(len(parsed.assumptions), 12)
        self.assertGreater(len(parsed.explicit_propositions), saturation.ITEM_CAP)
        self.assertGreater(len(parsed.assumptions), saturation.ITEM_CAP)

    def test_parser_accepts_json_code_fence_without_truncating(self) -> None:
        raw = "```json\n" + json.dumps(
            {
                "explicit_propositions": extraction_items("Explicit", 11),
                "assumptions": extraction_items("Implicit", 0),
            }
        ) + "\n```"

        parsed = saturation.parse_uncapped_extraction(raw, "fenced response")

        self.assertEqual(len(parsed.explicit_propositions), 11)

    def test_malformed_response_raises_specific_error(self) -> None:
        with self.assertRaisesRegex(saturation.ExtractionParseError, "required lists"):
            saturation.parse_uncapped_extraction('{"wrong": []}', "bad response")

    def test_invalid_confidence_raises_specific_error(self) -> None:
        raw = json.dumps(
            {
                "explicit_propositions": [{"text": "Claim", "confidence": 1.5}],
                "assumptions": [],
            }
        )
        with self.assertRaisesRegex(saturation.ExtractionParseError, "between 0 and 1"):
            saturation.parse_uncapped_extraction(raw, "bad confidence")

    def test_token_truncation_is_fatal(self) -> None:
        record = saturation.GenerationRecord(
            turn_id="business:episode:1",
            raw_output="{}",
            finish_reason="length",
            prompt_token_count=100,
            output_token_count=8192,
            run_signature="signature",
        )
        with self.assertRaisesRegex(saturation.GenerationTruncatedError, "token limit"):
            saturation.validate_generation_record(record)


class MetricTests(unittest.TestCase):
    def test_capping_and_iceberg_ratio_are_computed_from_the_same_extraction(self) -> None:
        turn = saturation.AssignedTurn(
            source=source_turn(1, 500, "religion"),
            length_decile=10,
        )
        generation = saturation.GenerationRecord(
            turn_id=turn.source.turn_id,
            raw_output="{}",
            finish_reason="stop",
            prompt_token_count=100,
            output_token_count=200,
            run_signature="signature",
        )
        extraction = saturation.ExtractionResult(
            explicit_propositions=tuple(
                saturation.ExtractionItem(text=f"E {index}", confidence=0.9)
                for index in range(12)
            ),
            assumptions=tuple(
                saturation.ExtractionItem(text=f"I {index}", confidence=0.9)
                for index in range(3)
            ),
        )

        result = saturation.build_turn_result(turn, generation, extraction, 10)

        self.assertEqual(result["uncapped_explicit_count"], 12)
        self.assertEqual(result["recapped_explicit_count"], 10)
        self.assertAlmostEqual(float(result["uncapped_iceberg_ratio"]), 3.0)
        self.assertAlmostEqual(float(result["recapped_iceberg_ratio"]), 2.5)
        self.assertTrue(result["ratio_changed_by_cap"])

    def test_top_decile_verdict_reports_explicit_and_implicit_separately(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "length_decile": 10,
                    "word_count": 500,
                    "explicit_ge_11": True,
                    "implicit_ge_11": False,
                    "either_ge_11": True,
                    "ratio_changed_by_cap": True,
                    "uncapped_explicit_count": 12,
                    "uncapped_implicit_count": 4,
                },
                {
                    "length_decile": 10,
                    "word_count": 800,
                    "explicit_ge_11": False,
                    "implicit_ge_11": True,
                    "either_ge_11": True,
                    "ratio_changed_by_cap": True,
                    "uncapped_explicit_count": 8,
                    "uncapped_implicit_count": 13,
                },
            ]
        )

        verdict = saturation.diagnostic_verdict(frame, 10, 10)

        self.assertEqual(verdict["explicit_ge_11_count"], 1)
        self.assertEqual(verdict["implicit_ge_11_count"], 1)
        self.assertTrue(verdict["cap_contribution_to_flattening_supported"])

    def test_negative_ratio_counts_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            saturation.iceberg_ratio(-1, 2)

    def test_json_records_convert_missing_numeric_values_to_null(self) -> None:
        records = saturation.json_records(pd.DataFrame([{"rate": float("nan")}]))

        self.assertIsNone(records[0]["rate"])

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib is not installed in the test environment",
    )
    def test_summary_and_both_plot_formats_are_rendered_for_complete_sample(self) -> None:
        rows: list[dict[str, object]] = []
        for decile in range(1, 11):
            for index in range(100):
                explicit_count = decile + 2
                implicit_count = 3
                uncapped_ratio = saturation.iceberg_ratio(explicit_count, implicit_count)
                recapped_ratio = saturation.iceberg_ratio(min(explicit_count, 10), 3)
                rows.append(
                    {
                        "turn_id": f"business:episode-{decile}:{index}",
                        "category": "business",
                        "word_count": decile * 100 + index,
                        "length_decile": decile,
                        "original_explicit_count": 10,
                        "original_implicit_count": 10,
                        "uncapped_explicit_count": explicit_count,
                        "uncapped_implicit_count": implicit_count,
                        "explicit_ge_11": explicit_count >= 11,
                        "implicit_ge_11": False,
                        "either_ge_11": explicit_count >= 11,
                        "uncapped_iceberg_ratio": uncapped_ratio,
                        "recapped_iceberg_ratio": recapped_ratio,
                        "ratio_changed_by_cap": not math.isclose(
                            uncapped_ratio,
                            recapped_ratio,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ),
                        "absolute_ratio_difference": abs(uncapped_ratio - recapped_ratio),
                        "prompt_token_count": 100,
                        "output_token_count": 200,
                        "finish_reason": "stop",
                    }
                )
        turns = pd.DataFrame(rows)
        deciles = saturation.build_decile_summary_frame(turns, 10, 10)
        config = saturation.ModelConfig(
            model_name=saturation.DEFAULT_MODEL_NAME,
            download_dir=None,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            batch_size=32,
            max_tokens=8192,
            max_model_len=32768,
            temperature=0.6,
            top_p=0.95,
            min_p=0.1,
            top_k=20,
            repetition_penalty=1.1,
            seed=42,
        )
        summary = saturation.build_summary(
            turns,
            deciles,
            {"input_file_count": 1},
            config,
            "sample-hash",
            "run-signature",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            saturation.write_json(output_dir / "summary.json", summary)
            png_path, pdf_path = saturation.save_plot(deciles, output_dir)

            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
