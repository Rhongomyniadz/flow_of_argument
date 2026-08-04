from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iclr.exp1_representation_baselines import exp1_representation_baselines as baseline


FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_turn(category: str, episode: str, index: int) -> dict:
    return {
        "category": category,
        "episode_id": episode,
        "turn_idx": index,
        "start_time": float(index),
        "turn_type_label": "Substantive",
        "conversation_move_label": "Assert / Elaborate" if index % 2 == 0 else "Answer",
        "turn_text": f"{category} {episode} turn {index}",
        "explicit_propositions": [
            {"text": f"explicit {category} {episode} {index}"},
            {"text": f"explicit {category} {episode} {index}"},
        ],
        "assumptions": [
            {"text": f"assumption {category} {episode} {index}"},
            f"assumption {category} {episode} {index}",
        ],
    }


def write_synthetic_corpus(root: Path) -> None:
    for category in ("alpha", "beta"):
        category_dir = root / category
        category_dir.mkdir(parents=True)
        for episode_number in range(3):
            episode_id = f"{category}-episode-{episode_number}"
            turns = [synthetic_turn(category, episode_id, index) for index in range(6)]
            if category == "alpha" and episode_number == 0:
                turns[0]["assumptions"] = []
                turns[1]["explicit_propositions"] = []
            if category == "alpha" and episode_number == 1:
                # A procedural turn breaks immediate adjacency between turn 1 and turn 3.
                turns.insert(
                    2,
                    {
                        "category": category,
                        "episode_id": episode_id,
                        "turn_idx": 2,
                        "start_time": 2.0,
                        "turn_type_label": "Backchannel",
                        "conversation_move_label": "Backchannel",
                        "turn_text": "Mm-hm.",
                        "explicit_propositions": [],
                        "assumptions": [],
                    },
                )
                for position, turn in enumerate(turns[3:], start=3):
                    turn["turn_idx"] = position
                    turn["start_time"] = float(position)
            if category == "beta" and episode_number == 2:
                # Same-episode donor assumptions are either empty or exactly equal.
                for turn in turns:
                    turn["assumptions"] = ["constant beta control"]
            value: object = turns if episode_number % 2 == 0 else {"turns": turns}
            (category_dir / f"{episode_id}.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def make_args(input_dir: Path, output_dir: Path, *extra: str):
    argv = [
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--bootstrap_draws", "5",
        "--prompt_batch_size", "128",
        *extra,
    ]
    args = baseline.parse_args(argv)
    baseline.validate_args(args)
    return args


class LoadingAndRepresentationTests(unittest.TestCase):
    def test_list_and_object_roots_and_chronological_order(self) -> None:
        list_turns, list_pairs = baseline.build_episode_records(
            "news", FIXTURES / "list_root_episode.json", history_turns=3
        )
        object_turns, object_pairs = baseline.build_episode_records(
            "news", FIXTURES / "object_root_episode.json", history_turns=3
        )
        self.assertEqual([turn["turn_idx"] for turn in list_turns], [0, 1])
        self.assertEqual(list_turns[0]["explicit_texts"], ["Explicit first."])
        self.assertEqual(list_turns[0]["assumption_texts"], ["Assumption first."])
        self.assertEqual(len(list_pairs), 1)
        self.assertEqual(len(object_turns), 2)
        self.assertEqual(len(object_pairs), 1)

    def test_non_substantive_turn_breaks_pair_but_not_substantive_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.json"
            turns = [
                synthetic_turn("news", "break", 0),
                {
                    **synthetic_turn("news", "break", 1),
                    "turn_type_label": "Backchannel",
                },
                synthetic_turn("news", "break", 2),
                synthetic_turn("news", "break", 3),
            ]
            path.write_text(json.dumps(turns), encoding="utf-8")
            records, pairs = baseline.build_episode_records("news", path, history_turns=3)
        self.assertEqual(len(records), 3)
        self.assertEqual([(pair["source_turn_idx"], pair["true_next_turn_idx"]) for pair in pairs], [(2, 3)])
        source_two = next(turn for turn in records if turn["turn_idx"] == 2)
        self.assertEqual(source_two["history_turn_ids"], ["news:break:0"])

    def test_all_representation_fields_and_placeholders(self) -> None:
        pair = {
            "pair_id": "p",
            "source_turn_text": "raw",
            "source_explicit_texts": [],
            "source_assumption_texts": [],
            "history_turn_texts": [],
            "donors": {
                "explicit_plus_shuffled_assumptions": {
                    "donor_assumptions": ["wrong"],
                    "control_unavailable_reason": None,
                }
            },
        }
        self.assertIn(baseline.EMPTY_EXPLICIT, baseline.format_representation(pair, "explicit_only"))
        self.assertIn(baseline.EMPTY_ASSUMPTIONS, baseline.format_representation(pair, "assumptions_only"))
        self.assertIn(baseline.EMPTY_HISTORY, baseline.format_representation(pair, "raw_turn_with_history"))
        real = baseline.format_representation(pair, "explicit_plus_assumptions")
        corrupt = baseline.format_representation(pair, "explicit_plus_shuffled_assumptions")
        self.assertIn("[Implicit assumptions]", real)
        self.assertIn("[Implicit assumptions]", corrupt)
        self.assertNotIn("shuffled", corrupt.casefold())
        self.assertNotIn("wrong", real)


class PreparationAndControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        write_synthetic_corpus(self.input_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self, seed: int = 42, output_name: str = "output"):
        output = self.root / output_name
        args = make_args(
            self.input_dir,
            output,
            "--prepare_only",
            "--seed", str(seed),
        )
        baseline.prepare_dataset(args)
        return args, baseline.read_jsonl(baseline.prepared_path(args))

    def test_candidate_invariance_hash_and_leakage(self) -> None:
        args, pairs = self.prepare()
        complete = [pair for pair in pairs if pair["candidate_pool_complete"]]
        self.assertTrue(complete)
        for pair in complete:
            candidates = pair["candidates"]
            ids = [candidate["candidate_turn_id"] for candidate in candidates]
            self.assertEqual(len(ids), baseline.EXPECTED_CANDIDATE_COUNT)
            self.assertEqual(len(set(ids)), baseline.EXPECTED_CANDIDATE_COUNT)
            self.assertEqual(pair["candidate_pool_sha256"], baseline.stable_hash(ids))
            self.assertFalse(set(ids).intersection(pair["history_turn_ids"]))
            for condition in args.conditions:
                self.assertEqual(pair["candidate_pool_sha256"], baseline.stable_hash(ids))

    def test_donors_are_deterministic_valid_and_seed_sensitive(self) -> None:
        _, first = self.prepare(seed=42, output_name="first")
        _, second = self.prepare(seed=42, output_name="second")
        _, third = self.prepare(seed=43, output_name="third")
        first_map = {
            (pair["pair_id"], condition): pair["donors"][condition].get("donor_turn_id")
            for pair in first
            for condition in baseline.CONTROL_CONDITIONS
        }
        second_map = {
            (pair["pair_id"], condition): pair["donors"][condition].get("donor_turn_id")
            for pair in second
            for condition in baseline.CONTROL_CONDITIONS
        }
        third_map = {
            (pair["pair_id"], condition): pair["donors"][condition].get("donor_turn_id")
            for pair in third
            for condition in baseline.CONTROL_CONDITIONS
        }
        self.assertEqual(first_map, second_map)
        self.assertTrue(any(first_map[key] != third_map[key] for key in first_map))
        for pair in first:
            candidate_ids = {candidate["candidate_turn_id"] for candidate in pair["candidates"]}
            shuffled = pair["donors"]["explicit_plus_shuffled_assumptions"]
            if shuffled["donor_turn_id"]:
                self.assertNotEqual(shuffled["donor_episode_id"], pair["episode_id"])
                self.assertNotIn(shuffled["donor_turn_id"], candidate_ids)
            wrong = pair["donors"]["explicit_plus_wrong_episode_assumptions"]
            if wrong["donor_turn_id"]:
                self.assertEqual(wrong["donor_episode_id"], pair["episode_id"])
                donor_idx = int(str(wrong["donor_turn_id"]).rsplit(":", 1)[1])
                self.assertGreaterEqual(abs(donor_idx - int(pair["source_turn_idx"])), 3)

    def test_missing_same_episode_control_is_explicit(self) -> None:
        _, pairs = self.prepare()
        beta_control_pairs = [
            pair
            for pair in pairs
            if pair["episode_id"] == "beta-episode-2" and pair["candidate_pool_complete"]
        ]
        self.assertTrue(beta_control_pairs)
        self.assertTrue(
            any(
                not pair["conditions"]["explicit_plus_wrong_episode_assumptions"]["available"]
                for pair in beta_control_pairs
            )
        )


class ParsingRankingAndGoldenTests(unittest.TestCase):
    def test_legacy_golden_contract(self) -> None:
        golden = json.loads((FIXTURES / "legacy_contract_golden.json").read_text(encoding="utf-8"))
        for case in golden["parser_cases"]:
            parsed = baseline.parse_llm_score(case["raw"])
            self.assertEqual(parsed["score"], case["score"])
            self.assertEqual(parsed["parse_success"], case["parse_success"])
            self.assertEqual(parsed["parse_error"], case["parse_error"])
        ranked = baseline.rank_condition_scores(golden["ranking_rows"])
        self.assertEqual([row["candidate_turn_id"] for row in ranked], golden["expected_rank_order"])
        positive = next(row for row in ranked if row["is_true_next_turn"])
        self.assertEqual(positive["rank"], golden["expected_true_rank"])
        self.assertEqual(1.0 / positive["rank"], golden["expected_true_reciprocal_rank"])

    def test_pair_id_matches_legacy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "golden.json"
            path.write_text(
                json.dumps([
                    synthetic_turn("news", "golden-episode", 0),
                    synthetic_turn("news", "golden-episode", 1),
                ]),
                encoding="utf-8",
            )
            _, pairs = baseline.build_episode_records("news", path, history_turns=3)
        self.assertEqual(pairs[0]["pair_id"], "news:golden-episode:0:1")

    def test_slurm_runner_contract_and_isolation(self) -> None:
        runner_path = EXPERIMENT_ROOT / "run_exp1_representation_baselines.sh"
        raw = runner_path.read_bytes()
        text = raw.decode("utf-8")
        self.assertNotIn(b"\r\n", raw)
        for expected in (
            "#SBATCH --job-name=exp1_repr_baselines",
            "#SBATCH --gres=gpu:A6000:2",
            "iclr/exp1_representation_baselines/_log/",
            "EXP1_BASELINE_STAGE",
            "prepare)",
            "patch)",
            "merge)",
            "analysis)",
            "ALLOW_FULL_RUN",
            "TENSOR_PARALLEL_SIZE=2",
            "Submit this runner with sbatch, not bash",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("experiments/exp1_relevance_bridge/run_exp1.sh", text)

        source = (EXPERIMENT_ROOT / "exp1_representation_baselines.py").read_text(encoding="utf-8")
        self.assertIn('tensor_parallel_size=args.tensor_parallel_size', source)
        self.assertIn('distributed_executor_backend="mp"', source)


class EndToEndAndPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        write_synthetic_corpus(self.input_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fake_score_pipeline_outputs_and_paired_orientation(self) -> None:
        output = self.root / "single"
        args = make_args(self.input_dir, output, "--dry_run")
        baseline.prepare_dataset(args)
        score_manifest = baseline.score_dataset(args)
        summary = baseline.analyze_dataset(args)
        self.assertEqual(score_manifest["expected_task_count"], score_manifest["valid_task_count"])
        for path in baseline.final_paths(output).values():
            self.assertTrue(path.exists(), path)
        long_df = pd.read_csv(baseline.final_paths(output)["metrics_long"])
        self.assertTrue((long_df.loc[long_df["full_retained"] == True, "true_rank"] == 1).all())
        pairwise = pd.read_csv(baseline.final_paths(output)["pairwise"])
        self.assertTrue((pairwise.loc[pairwise["paired_sample_size"] > 0, "mean_improvement"].fillna(0) == 0).all())
        self.assertIn("output_hashes", summary)
        self.assertNotIn("summary", summary["output_hashes"])
        self.assertEqual(len(summary["strict_control_paper_table"]), len(baseline.DEFAULT_CONDITIONS))

        resume_args = make_args(self.input_dir, output, "--dry_run", "--score_only")
        resumed = baseline.score_dataset(resume_args)
        self.assertEqual(resumed["attempted_this_run"], 0)

        score_file = baseline.final_paths(output)["scores"]
        score_rows = baseline.read_jsonl(score_file)
        invalid = dict(
            score_rows[0],
            score=None,
            parse_success=False,
            parse_error="missing_json_object",
            raw_output="invalid",
        )
        baseline.write_jsonl(score_file, [invalid, *score_rows[1:]])
        retried = baseline.score_dataset(resume_args)
        self.assertEqual(retried["attempted_this_run"], 1)
        self.assertEqual(retried["valid_task_count"], retried["expected_task_count"])

    def test_incomplete_candidate_pool_still_produces_auditable_outputs(self) -> None:
        input_dir = self.root / "small-input" / "news"
        input_dir.mkdir(parents=True)
        turns = [
            synthetic_turn("news", "small", 0),
            synthetic_turn("news", "small", 1),
        ]
        (input_dir / "small.json").write_text(json.dumps(turns), encoding="utf-8")
        output = self.root / "small-output"
        args = make_args(self.root / "small-input", output, "--dry_run")
        prepared = baseline.prepare_dataset(args)
        scored = baseline.score_dataset(args)
        analyzed = baseline.analyze_dataset(args)
        self.assertEqual(prepared["candidate_complete_pair_count"], 0)
        self.assertEqual(scored["expected_task_count"], 0)
        self.assertEqual(analyzed["candidate_complete_pair_count"], 0)
        self.assertTrue(baseline.final_paths(output)["scores"].exists())

    def test_patch_merge_duplicate_and_missing_patch_contracts(self) -> None:
        output = self.root / "patched"
        prepare_args = make_args(self.input_dir, output, "--prepare_only")
        baseline.prepare_dataset(prepare_args)
        for patch_index in (0, 1):
            args = make_args(
                self.input_dir,
                output,
                "--dry_run",
                "--score_only",
                "--num_patches", "2",
                "--episodes_per_patch", "3",
                "--patch_index", str(patch_index),
            )
            baseline.score_dataset(args)
        merge_args = make_args(
            self.input_dir,
            output,
            "--merge_patches_only",
            "--dry_run",
            "--num_patches", "2",
            "--episodes_per_patch", "3",
        )
        merged = baseline.merge_patch_scores(merge_args)
        self.assertGreater(merged["merged_score_count"], 0)
        merged_rows = baseline.read_jsonl(baseline.final_paths(output)["scores"])
        merged_order = [
            (row["pair_id"], row["condition"], int(row["candidate_order"]))
            for row in merged_rows
        ]
        self.assertEqual(merged_order, sorted(merged_order))

        patch_zero_scores = baseline.score_path(baseline.patch_dir(output, 0, 2))
        duplicate = baseline.read_jsonl(patch_zero_scores)[0]
        baseline.append_jsonl(patch_zero_scores, [duplicate])
        merged_again = baseline.merge_patch_scores(merge_args)
        self.assertEqual(merged_again["merged_score_count"], merged["merged_score_count"])

        conflicting = dict(duplicate, rationale="conflicting duplicate")
        baseline.append_jsonl(patch_zero_scores, [conflicting])
        with self.assertRaisesRegex(RuntimeError, "Conflicting duplicate"):
            baseline.merge_patch_scores(merge_args)

        missing_output = self.root / "missing"
        missing_args = make_args(
            self.input_dir,
            missing_output,
            "--merge_patches_only",
            "--num_patches", "2",
        )
        with self.assertRaisesRegex(RuntimeError, "Missing patch output"):
            baseline.merge_patch_scores(missing_args)


if __name__ == "__main__":
    unittest.main()
