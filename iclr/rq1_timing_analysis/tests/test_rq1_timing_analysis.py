from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "iclr" / "rq1_timing_analysis" / "rq1_timing_analysis.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_analysis_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("rq1_timing_analysis_module", SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load module specification from {SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


analysis = load_analysis_module()


def analysis_dependencies_available() -> bool:
    return (
        importlib.util.find_spec("statsmodels") is not None
        and importlib.util.find_spec("matplotlib") is not None
        and importlib.util.find_spec("tqdm") is not None
    )


def representation_items(prefix: str, count: int) -> list[dict[str, object]]:
    return [{"text": f"{prefix}-{index}"} for index in range(count)]


def write_model_episodes(input_dir: Path, episode_count: int, turn_count: int) -> None:
    stance_pattern = (-3, -1, 2, 4, 0, -2, 3, 1, -4, 2, 0, 5, -1, 3, -2, 1)
    words = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    for episode_index in range(episode_count):
        category = "alpha" if episode_index % 2 == 0 else "beta"
        previous_end = 0.0
        turns: list[dict[str, object]] = []
        for turn_index in range(turn_count):
            if turn_index == 0:
                start = 0.0
            else:
                gap = -0.15 if (turn_index + episode_index) % 7 == 0 else 0.2 * ((turn_index + 1) % 3)
                start = previous_end + gap
            duration = 1.5 + 0.4 * ((turn_index + episode_index) % 5)
            end = start + duration
            previous_end = end
            explicit_count = 1 + ((turn_index * 2 + episode_index) % 6)
            assumption_count = (turn_index + episode_index * 2) % 5
            word_count = 7 + ((turn_index * 3 + episode_index) % 9)
            turns.append(
                {
                    "category": category,
                    "episode_id": f"episode-{episode_index:03d}",
                    "turn_idx": turn_index,
                    "speaker_id": "A" if turn_index % 2 == 0 else "B",
                    "turn_type_label": "Substantive",
                    "turn_text": " ".join(words.split()[:word_count]),
                    "stance_pt": stance_pattern[(turn_index + episode_index) % len(stance_pattern)],
                    "start_time": start,
                    "end_time": end,
                    "explicit_propositions": representation_items("e", explicit_count),
                    "assumptions": representation_items("a", assumption_count),
                }
            )
        (input_dir / f"episode-{episode_index:03d}.json").write_text(
            json.dumps(turns),
            encoding="utf-8",
        )


class FeatureConstructionTests(unittest.TestCase):
    def test_list_root_exact_features_gaps_and_overlap(self) -> None:
        path = FIXTURES / "list_root_episode.json"
        raw_turns = analysis.load_episode(path)
        observations, exclusions, timing_sources = analysis.build_episode_observations(
            raw_turns,
            "alpha",
            "list-episode",
            path,
        )
        self.assertEqual(len(observations), 3)
        self.assertEqual(exclusions, {})
        self.assertEqual(timing_sources, {"start_end": 5})

        first = observations[0]
        self.assertEqual(first["current_turn_raw_index"], 2)
        self.assertAlmostEqual(first["delta_stance"], 0.4)
        self.assertAlmostEqual(first["lag_delta_stance"], 0.2)
        self.assertAlmostEqual(first["agree_move"], 0.4)
        self.assertEqual(first["disagree_move"], 0.0)
        self.assertAlmostEqual(first["lag_agree_move"], 0.2)
        self.assertEqual(first["lag_disagree_move"], 0.0)
        self.assertAlmostEqual(first["previous_iceberg_ratio"], 1.0)
        self.assertAlmostEqual(first["iceberg_ratio"], 2.0)
        self.assertAlmostEqual(
            first["delta_log_iceberg_ratio"],
            math.log1p(2.0) - math.log1p(1.0),
        )
        self.assertAlmostEqual(first["previous_density_per_token"], 0.125)
        self.assertAlmostEqual(first["density_per_token"], 0.2)
        self.assertAlmostEqual(
            first["delta_log_density_per_token"],
            math.log1p(0.2) - math.log1p(0.125),
        )
        self.assertEqual(first["raw_gap"], 1.0)
        self.assertEqual(first["pre_turn_gap"], 1.0)
        self.assertAlmostEqual(first["log_gap"], math.log(2.0))
        self.assertEqual(first["overlap"], 0)
        self.assertAlmostEqual(first["timeline_position"], 8.0 / 14.0)

        overlap = observations[1]
        self.assertEqual(overlap["raw_gap"], -0.5)
        self.assertEqual(overlap["pre_turn_gap"], 0.0)
        self.assertEqual(overlap["log_gap"], 0.0)
        self.assertEqual(overlap["overlap"], 1)
        zero_gap = observations[2]
        self.assertEqual(zero_gap["raw_gap"], 0.0)
        self.assertEqual(zero_gap["overlap"], 0)

    def test_object_root_endpoint_derivation_and_explicit_exclusions(self) -> None:
        path = FIXTURES / "object_root_episode.json"
        raw_turns = analysis.load_episode(path)
        observations, exclusions, timing_sources = analysis.build_episode_observations(
            raw_turns,
            "beta",
            "object-episode",
            path,
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["previous_timestamp_source"], "start_duration")
        self.assertEqual(observations[0]["raw_gap"], 0.0)
        self.assertEqual(timing_sources["end_duration"], 1)
        self.assertEqual(timing_sources["start_duration"], 1)
        self.assertEqual(exclusions["invalid_current_timing:nonpositive_endpoint_duration"], 1)
        self.assertEqual(exclusions["invalid_previous_timing:nonpositive_endpoint_duration"], 1)
        self.assertEqual(exclusions["same_speaker_boundary"], 1)
        self.assertEqual(exclusions["non_substantive_window"], 3)

    def test_invalid_endpoints_are_not_replaced_by_duration(self) -> None:
        timing = analysis.parse_timing({"start_time": 4.0, "end_time": 4.0, "duration": 3.0})
        self.assertFalse(timing["valid"])
        self.assertEqual(timing["error"], "nonpositive_endpoint_duration")
        self.assertIsNone(timing["duration"])

    def test_non_substantive_turn_breaks_raw_adjacency(self) -> None:
        path = FIXTURES / "list_root_episode.json"
        turns = analysis.load_episode(path)[:4]
        changed = [dict(turn) for turn in turns]
        changed[1]["turn_type_label"] = "Backchannel"
        observations, exclusions, _ = analysis.build_episode_observations(
            changed,
            "alpha",
            "broken-adjacency",
            path,
        )
        self.assertEqual(observations, [])
        self.assertEqual(exclusions, {"non_substantive_window": 2})

    def test_attenuation_formula_and_amplification(self) -> None:
        self.assertAlmostEqual(analysis.attenuation_percent(-0.01, -0.008), 20.0)
        self.assertAlmostEqual(analysis.attenuation_percent(0.01, 0.012), -20.0)
        with self.assertRaisesRegex(ZeroDivisionError, "zero baseline coefficient"):
            analysis.attenuation_percent(0.0, 0.1)

    def test_bulk_observation_csv_is_the_only_new_ignored_result(self) -> None:
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(
            "/iclr/rq1_timing_analysis/results/rq1_timing_observations.csv",
            ignore_text,
        )
        self.assertNotIn("/iclr/rq1_timing_analysis/results/*.csv", ignore_text)
        self.assertNotIn("/iclr/rq1_timing_analysis/results/*.png", ignore_text)


@unittest.skipUnless(
    analysis_dependencies_available(),
    "statsmodels, matplotlib, and tqdm are required for regression integration tests",
)
class RegressionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        write_model_episodes(self.input_dir, 30, 14)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_clustered_models_share_the_exact_sample(self) -> None:
        frame, audit = analysis.build_dataset(
            self.input_dir,
            [],
            "parsed",
            None,
            False,
        )
        self.assertEqual(len(frame), 360)
        self.assertEqual(frame["episode"].nunique(), 30)
        self.assertEqual(audit["retained_observation_count"], 360)
        results = analysis.fit_models(frame)
        self.assertEqual(set(results), set(analysis.MODEL_ORDER))
        self.assertTrue(all(int(result.nobs) == 360 for result in results.values()))
        baseline_terms = set(results[analysis.RATIO_MODEL].params.index)
        self.assertIn("lag_agree_move", baseline_terms)
        self.assertIn("lag_disagree_move", baseline_terms)
        self.assertNotIn("lag_delta_stance", baseline_terms)
        duration_terms = set(results[analysis.DURATION_MODEL].params.index)
        self.assertIn("log_duration", duration_terms)
        self.assertNotIn("log_gap", duration_terms)
        self.assertNotIn("overlap", duration_terms)
        timing_terms = set(results[analysis.TIMING_MODEL].params.index)
        self.assertIn("log_duration", timing_terms)
        self.assertIn("log_gap", timing_terms)
        self.assertIn("overlap", timing_terms)
        coefficients = analysis.coefficient_frame(results)
        for model_name in analysis.MODEL_ORDER:
            for term in analysis.STANCE_TERMS:
                selected = coefficients[
                    (coefficients["model_name"] == model_name)
                    & (coefficients["term"] == term)
                ]
                self.assertEqual(len(selected), 1)
                self.assertTrue(math.isfinite(float(selected.iloc[0]["clustered_se"])))
        model_fit = analysis.model_fit_frame(results, frame)
        self.assertEqual(model_fit["transition_count"].nunique(), 1)
        self.assertEqual(model_fit["episode_count"].nunique(), 1)
        self.assertEqual(set(model_fit["response_variable"]), {"delta_log_iceberg_ratio"})
        self.assertTrue(model_fit["aic_bic_comparable_to_other_models"].all())

    def test_end_to_end_writes_every_artifact_and_hash(self) -> None:
        output_dir = self.root / "output"
        args = argparse.Namespace(
            data_dir=self.input_dir,
            output_dir=output_dir,
            categories=None,
            category_data_subdir="parsed",
            max_episodes=None,
            plot_dpi=100,
            no_tqdm=True,
        )
        summary = analysis.run_analysis(args)
        paths = analysis.output_paths(output_dir)
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths.values()))
        self.assertEqual(summary["transition_count"], 360)
        self.assertEqual(summary["episode_count"], 30)
        self.assertEqual(
            set(summary["artifact_sha256"]),
            {path.name for key, path in paths.items() if key != "summary"},
        )
        comparison = pd.read_csv(paths["stance_comparison"])
        self.assertEqual(len(comparison), 6)
        self.assertEqual(set(comparison["model_name"]), set(analysis.HEADLINE_MODELS))
        duration_rows = comparison[comparison["model_name"] == analysis.DURATION_MODEL]
        self.assertEqual(len(duration_rows), 2)
        self.assertTrue(duration_rows["attenuation_from_baseline_percent"].notna().all())
        self.assertTrue(duration_rows["timing_attenuation_percent"].isna().all())
        self.assertTrue(duration_rows["incremental_attenuation_percent"].notna().all())
        self.assertTrue(
            all(
                f"{analysis.DURATION_MODEL}:{term}" in summary["headline_coefficients"]
                for term in analysis.STANCE_TERMS
            )
        )
        audit = json.loads(paths["data_audit"].read_text(encoding="utf-8"))
        self.assertTrue(audit["common_model_sample_verified"])
        self.assertEqual(audit["retained_observation_count"], 360)
        self.assertEqual(audit["observation_csv_sha256"], analysis.file_sha256(paths["observations"]))


if __name__ == "__main__":
    unittest.main()
