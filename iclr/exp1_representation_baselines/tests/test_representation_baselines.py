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
        "merged_from_turn_indices": [index],
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


def build_records(category: str, path: Path, history_turns: int):
    return baseline.build_episode_records(
        category,
        path,
        history_turns,
        baseline.DEFAULT_SOURCE_TAIL_WORDS,
        baseline.DEFAULT_CANDIDATE_HEAD_WORDS,
        baseline.DEFAULT_ASSUMPTION_BUDGET,
        list(baseline.DEFAULT_FUTURE_HORIZONS),
    )


class LoadingAndRepresentationTests(unittest.TestCase):
    def test_future_horizons_use_exact_turn_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "horizons.json"
            turns = [synthetic_turn("news", "horizons", index) for index in range(6)]
            path.write_text(json.dumps(turns), encoding="utf-8")
            _, pairs, _ = build_records("news", path, history_turns=3)
        counts = pd.Series([pair["future_horizon"] for pair in pairs]).value_counts().to_dict()
        self.assertEqual(counts, {1: 5, 3: 3, 5: 1})
        horizon_three = next(
            pair
            for pair in pairs
            if pair["source_turn_idx"] == 0 and pair["future_horizon"] == 3
        )
        self.assertEqual(horizon_three["true_next_turn_idx"], 3)

    def test_list_and_object_roots_and_chronological_order(self) -> None:
        list_turns, list_pairs, _ = build_records(
            "news", FIXTURES / "list_root_episode.json", history_turns=3
        )
        object_turns, object_pairs, _ = build_records(
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
            records, pairs, _ = build_records("news", path, history_turns=3)
        self.assertEqual(len(records), 3)
        self.assertEqual([(pair["source_turn_idx"], pair["true_next_turn_idx"]) for pair in pairs], [(2, 3)])
        source_two = next(turn for turn in records if turn["turn_idx"] == 2)
        self.assertEqual(source_two["history_turn_ids"], ["news:break:0"])

    def test_all_representation_fields_and_placeholders(self) -> None:
        pair = {
            "pair_id": "p",
            "source_turn_text": "raw",
            "source_tail_text": "raw",
            "source_explicit_texts": [],
            "source_assumption_texts": [],
            "source_all_assumption_texts": [],
            "history_turn_texts": [],
            "donors": {
                "raw_history_different_episode_assumptions": {
                    "matched_donor_assumptions": ["wrong"],
                    "donor_assumptions": ["wrong"],
                    "control_unavailable_reason": None,
                }
            },
        }
        self.assertIn(baseline.EMPTY_EXPLICIT, baseline.format_representation(pair, "explicit_only"))
        self.assertIn(baseline.EMPTY_ASSUMPTIONS, baseline.format_representation(pair, "assumptions_only"))
        self.assertIn(baseline.EMPTY_HISTORY, baseline.format_representation(pair, "raw_turn_with_history"))
        real = baseline.format_representation(pair, "explicit_plus_assumptions")
        first_one = baseline.format_representation(pair, "explicit_plus_top1_assumption")
        first_three = baseline.format_representation(pair, "explicit_plus_top3_assumptions")
        raw_plus = baseline.format_representation(pair, "raw_turn_plus_assumptions")
        corrupt = baseline.format_representation(pair, "explicit_plus_different_episode_assumptions")
        self.assertIn("[All extracted implicit assumptions]", real)
        self.assertIn("first 1", first_one)
        self.assertIn("first 3", first_three)
        self.assertIn("[Final local window of the current turn]", raw_plus)
        self.assertIn("[Implicit assumptions]", corrupt)
        self.assertNotIn("different_episode", corrupt.casefold())
        self.assertNotIn("wrong", real)

    def test_assumption_budget_conditions_preserve_extraction_order(self) -> None:
        pair = {
            "pair_id": "budget",
            "source_turn_text": "raw",
            "source_tail_text": "raw",
            "source_explicit_texts": ["explicit"],
            "source_assumption_texts": ["first", "second", "third", "fourth"],
            "source_all_assumption_texts": ["first", "second", "third", "fourth"],
            "history_turn_texts": [],
            "donors": {},
        }
        first_one = baseline.format_representation(pair, "explicit_plus_top1_assumption")
        first_three = baseline.format_representation(pair, "explicit_plus_top3_assumptions")
        self.assertIn("first", first_one)
        self.assertNotIn("second", first_one)
        self.assertIn("third", first_three)
        self.assertNotIn("fourth", first_three)

    def test_raw_history_turn_count_tracks_budgeted_dialogue_tokens(self) -> None:
        context_turns = [
            {
                "turn_id": f"turn-{index}",
                "turn_idx": index,
                "turn_text": " ".join(f"w{index}_{word}" for word in range(90)),
                "explicit_texts": [],
                "assumption_texts": [],
            }
            for index in range(4)
        ]
        pair = {"pair_id": "turn-budget", "context_turns": context_turns}
        counts = [
            baseline.raw_history_included_turn_count(pair, budget)
            for budget in (128, 256, 512)
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertGreaterEqual(counts[0], 1)
        self.assertEqual(counts[-1], 4)
        for budget in (128, 256, 512):
            self.assertLessEqual(
                baseline.whitespace_token_count(baseline.format_raw_history(pair, budget)),
                budget,
            )

    def test_matched_history_conditions_share_identical_raw_prefix_and_aux_length(self) -> None:
        context_turns = [
            {
                "turn_id": f"turn-{index}",
                "turn_idx": index,
                "substantive_position": index + 1,
                "turn_text": " ".join(f"raw{index}_{word}" for word in range(80)),
                "explicit_texts": [f"explicit state {index} with several words", "repeated explicit state"],
                "assumption_texts": [f"implicit state {index} with several words", "repeated assumption"],
            }
            for index in range(4)
        ]
        pair = {
            "pair_id": "matched",
            "context_turns": context_turns,
            "source_assumption_texts": ["current implicit assumption has six useful content words"],
            "donors": {
                "raw_history_different_episode_assumptions": {
                    "matched_donor_assumptions": ["different episode implicit assumption has six useful content words"],
                    "matched_auxiliary_word_count": 8,
                    "control_unavailable_reason": None,
                },
                "raw_history_same_episode_random_turn_assumptions": {
                    "matched_donor_assumptions": ["same episode random turn assumption also has useful content words"],
                    "matched_auxiliary_word_count": 8,
                    "control_unavailable_reason": None,
                },
            },
        }
        # Make donor audit lengths match the actual true-assumption content budget.
        target_words = baseline.auxiliary_word_budget(pair)
        for donor in pair["donors"].values():
            donor["matched_donor_assumptions"], donor["matched_auxiliary_word_count"] = \
                baseline.truncate_text_items_to_words(donor["matched_donor_assumptions"], target_words)
        raw = baseline.format_representation(pair, baseline.matched_condition_id("raw_history", 128))
        self.assertLessEqual(baseline.whitespace_token_count(raw), 128)
        augmented_counts = []
        for base_condition in baseline.MATCHED_BASE_CONDITIONS:
            condition = baseline.matched_condition_id(base_condition, 128)
            representation = baseline.format_representation(pair, condition)
            if base_condition != "raw_history":
                self.assertTrue(representation.startswith(raw + "\n\n"))
                augmented_counts.append(baseline.whitespace_token_count(representation))
        self.assertEqual(len(set(augmented_counts)), 1)


    def test_confidence_ranking_and_nonconsecutive_merge_provenance(self) -> None:
        assumptions = [
            {
                "text": "The speaker expects the budget question to be answered.",
                "confidence": 0.2,
            },
            {"text": "Highest-confidence unrelated assumption.", "confidence": 0.95},
            {"text": "First tied assumption.", "confidence": 0.7},
            {"text": "Second tied assumption.", "confidence_score": 0.7},
            "Legacy unscored assumption.",
            "Highest-confidence unrelated assumption.",
        ]
        ranked = baseline.rank_assumptions_by_confidence(assumptions)
        self.assertEqual(
            ranked,
            [
                "Highest-confidence unrelated assumption.",
                "First tied assumption.",
                "Second tied assumption.",
                "The speaker expects the budget question to be answered.",
                "Legacy unscored assumption.",
            ],
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            baseline.rank_assumptions_by_confidence([{"text": "Invalid", "confidence": 1.1}])
        with self.assertRaisesRegex(TypeError, "must be a number"):
            baseline.rank_assumptions_by_confidence([{"text": "Invalid", "confidence": "high"}])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.json"
            source = synthetic_turn("news", "provenance", 0)
            source["merged_from_turn_indices"] = [0, 2]
            source["assumptions"] = assumptions
            target = synthetic_turn("news", "provenance", 1)
            target["merged_from_turn_indices"] = [3]
            path.write_text(json.dumps([source, target]), encoding="utf-8")
            turns, pairs, counts = build_records("news", path, history_turns=3)
        self.assertEqual(turns[0]["assumption_texts"], ranked[:3])
        self.assertEqual(pairs, [])
        self.assertEqual(counts["boundary_invalid_merged_group_count"], 1)


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

    def test_prepare_rejects_cleaned_data_without_original_provenance(self) -> None:
        path = self.input_dir / "alpha" / "alpha-episode-0.json"
        turns = json.loads(path.read_text(encoding="utf-8"))
        for turn in turns:
            turn.pop("merged_from_turn_indices")
        path.write_text(json.dumps(turns), encoding="utf-8")
        args = make_args(self.input_dir, self.root / "missing-provenance", "--prepare_only")
        with self.assertRaisesRegex(RuntimeError, "Regenerate data_cleaned"):
            baseline.prepare_dataset(args)

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
        self.assertEqual(set(first_map), set(third_map))
        for pair in first:
            candidate_ids = {candidate["candidate_turn_id"] for candidate in pair["candidates"]}
            different_episode = pair["donors"]["raw_history_different_episode_assumptions"]
            if different_episode["donor_turn_id"]:
                self.assertNotEqual(different_episode["donor_episode_id"], pair["episode_id"])
                self.assertNotIn(different_episode["donor_turn_id"], candidate_ids)
            same_episode_random_turn = pair["donors"]["raw_history_same_episode_random_turn_assumptions"]
            if same_episode_random_turn["donor_turn_id"]:
                self.assertEqual(same_episode_random_turn["donor_episode_id"], pair["episode_id"])
                donor_idx = int(str(same_episode_random_turn["donor_turn_id"]).rsplit(":", 1)[1])
                self.assertGreaterEqual(int(pair["source_turn_idx"]) - donor_idx, 3)

    def test_missing_same_episode_control_is_explicit(self) -> None:
        args = make_args(
            self.input_dir,
            self.root / "controls",
            "--prepare_only",
            "--conditions",
            *baseline.ALL_CONDITIONS,
        )
        baseline.prepare_dataset(args)
        pairs = baseline.read_jsonl(baseline.prepared_path(args))
        beta_control_pairs = [
            pair
            for pair in pairs
            if pair["episode_id"] == "beta-episode-2" and pair["candidate_pool_complete"]
        ]
        self.assertTrue(beta_control_pairs)
        self.assertTrue(
            any(
                not pair["conditions"][baseline.matched_condition_id("raw_history_same_episode_random_turn_assumptions", 256)]["available"]
                for pair in beta_control_pairs
            )
        )


class ParsingAccuracyAndGoldenTests(unittest.TestCase):
    def test_default_output_directory_is_model_scoped(self) -> None:
        first = baseline.parse_args(["--model_name", "Qwen/First-Model"])
        second = baseline.parse_args(["--model_name", "Other/Second-Model"])
        baseline.validate_args(first)
        baseline.validate_args(second)
        self.assertEqual(first.output_dir, baseline.DEFAULT_OUTPUT_ROOT / "Qwen__First-Model")
        self.assertEqual(second.output_dir, baseline.DEFAULT_OUTPUT_ROOT / "Other__Second-Model")
        self.assertNotEqual(first.output_dir, second.output_dir)

    def test_default_conditions_are_the_diagnostic_decomposition(self) -> None:
        args = baseline.parse_args([])
        baseline.validate_args(args)
        self.assertEqual(args.source_tail_words, 100)
        self.assertEqual(args.candidate_head_words, 100)
        self.assertEqual(args.future_horizons, [1, 3, 5])
        self.assertEqual(args.representation_budgets, [256])
        self.assertEqual(args.conditions, list(baseline.DEFAULT_CONDITIONS))

    def test_forced_choice_parser_and_binary_accuracy(self) -> None:
        golden = json.loads((FIXTURES / "legacy_contract_golden.json").read_text(encoding="utf-8"))
        for case in golden["parser_cases"]:
            parsed = baseline.parse_llm_choice(case["raw"])
            self.assertEqual(parsed["choice"], case["choice"])
            self.assertEqual(parsed["parse_success"], case["parse_success"])
            self.assertEqual(parsed["parse_error"], case["parse_error"])

        accuracy_fixture = golden["binary_accuracy"]
        rows = []
        for index in range(24):
            preferences = accuracy_fixture["preference_overrides"].get(
                str(index),
                accuracy_fixture["default_preferences"],
            )
            negative_order = accuracy_fixture["negative_candidate_orders"][index]
            for order, preference in zip(("positive_first", "positive_second"), preferences):
                rows.append(
                    {
                        "comparison_id": f"comparison-{index}",
                        "presentation_order": order,
                        "positive_candidate_order": accuracy_fixture["positive_candidate_order"],
                        "negative_candidate_order": negative_order,
                        "choice": "A",
                        "positive_preference": preference,
                        "parse_success": True,
                    }
                )
        metrics = baseline.aggregate_pairwise_condition(rows)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["accuracy"], accuracy_fixture["expected_accuracy"])
        self.assertEqual(
            metrics["order_consistency_rate"],
            accuracy_fixture["expected_order_consistency_rate"],
        )

    def test_retry_prompt_preserves_forced_choice_contract(self) -> None:
        prompt = baseline.build_retry_prompt(
            "source",
            "candidate A",
            "candidate B",
            3,
            "invalid",
            "invalid_json",
        )
        self.assertIn("Candidate A", prompt)
        self.assertIn("Candidate B", prompt)
        self.assertIn("valid JSON object", prompt)
        self.assertIn("Previous parse error", prompt)

    def test_pair_id_matches_legacy_contract(self) -> None:
        golden = json.loads((FIXTURES / "legacy_contract_golden.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "golden.json"
            path.write_text(
                json.dumps([
                    synthetic_turn("news", "golden-episode", 0),
                    synthetic_turn("news", "golden-episode", 1),
                ]),
                encoding="utf-8",
            )
            _, pairs, _ = build_records("news", path, history_turns=3)
        self.assertEqual(pairs[0]["pair_id"], golden["pair_id"])

    def test_diagnostic_gate_requires_replication_before_full_corpus(self) -> None:
        pairwise = pd.DataFrame(
            [
                {
                    "analysis_subset": "sparse_explicit",
                    "future_horizon": 1,
                    "target_condition": "explicit_plus_top3_assumptions",
                    "baseline_condition": "explicit_only",
                    "metric": "accuracy",
                    "mean_improvement": 0.03,
                    "ci95_low": 0.01,
                    "ci95_high": 0.05,
                },
                {
                    "analysis_subset": "sparse_explicit",
                    "future_horizon": 1,
                    "target_condition": "explicit_plus_top3_assumptions",
                    "baseline_condition": "explicit_plus_different_episode_assumptions",
                    "metric": "accuracy",
                    "mean_improvement": 0.02,
                    "ci95_low": 0.005,
                    "ci95_high": 0.04,
                },
                {
                    "analysis_subset": "sparse_explicit",
                    "future_horizon": 1,
                    "target_condition": "explicit_plus_top3_assumptions",
                    "baseline_condition": "explicit_plus_same_episode_random_turn_assumptions",
                    "metric": "accuracy",
                    "mean_improvement": 0.02,
                    "ci95_low": 0.004,
                    "ci95_high": 0.04,
                },
                {
                    "analysis_subset": "assumption_eligible",
                    "future_horizon": 1,
                    "target_condition": "raw_turn_plus_assumptions",
                    "baseline_condition": "raw_turn",
                    "metric": "accuracy",
                    "mean_improvement": 0.02,
                    "ci95_low": 0.005,
                    "ci95_high": 0.04,
                },
            ]
        )
        long_rows = []
        for category in ("alpha", "beta"):
            for condition, value in (
                ("explicit_only", 0.4),
                ("explicit_plus_top3_assumptions", 0.5),
                ("explicit_plus_different_episode_assumptions", 0.45),
                ("explicit_plus_same_episode_random_turn_assumptions", 0.44),
                ("raw_turn", 0.6),
                ("raw_turn_plus_assumptions", 0.7),
            ):
                long_rows.append(
                    {
                        "pair_id": f"{category}:pair",
                        "category": category,
                        "condition": condition,
                        "future_horizon": 1,
                        "assumption_eligible": True,
                        "sparse_explicit": True,
                        "accuracy": value,
                    }
                )
        coverage = pd.DataFrame([{"retained_pair_rate": 0.99}, {"retained_pair_rate": 1.0}])
        gate = baseline.diagnostic_gate(pairwise, pd.DataFrame(long_rows), coverage)
        self.assertTrue(gate["ready_for_cross_model_smoke"])
        self.assertFalse(gate["ready_for_full_corpus"])
        self.assertEqual(gate["interpretation"], "assumptions_add_signal_beyond_raw_lexical_context")

    def test_slurm_runner_contract_and_isolation(self) -> None:
        runner_path = EXPERIMENT_ROOT / "run_exp1_representation_baselines.sh"
        raw = runner_path.read_bytes()
        text = raw.decode("utf-8")
        self.assertNotIn(b"\r\n", raw)
        for expected in (
            "#SBATCH --job-name=exp1_repr_diagnostic",
            "#SBATCH --gres=gpu:A6000:2",
            "iclr/exp1_representation_baselines/_log/exp1_repr_diagnostic_",
            "EXP1_BASELINE_STAGE",
            "patch)",
            "merge)",
            "analysis)",
            "Prepared data is missing",
            "prepare_exp1_representation.py",
            "ALLOW_FULL_RUN",
            "TENSOR_PARALLEL_SIZE=2",
            "Submit this runner with sbatch, not bash",
            'MODEL_OUTPUT_NAME="${MODEL_NAME//\\//__}"',
            "raw_history",
            "raw_history_true_assumptions",
            "raw_history_different_episode_assumptions",
            "raw_history_same_episode_random_turn_assumptions",
            "raw_history_explicit",
            "REPRESENTATION_BUDGETS_CSV",
            'REPRESENTATION_BUDGETS_CSV="${REPRESENTATION_BUDGETS_CSV:-256}"',
            "SOURCE_TAIL_WORDS",
            "CANDIDATE_HEAD_WORDS",
            'FUTURE_HORIZONS_CSV="${FUTURE_HORIZONS_CSV:-1,3,5}"',
            'SOURCE_TAIL_WORDS="${SOURCE_TAIL_WORDS:-100}"',
            'CANDIDATE_HEAD_WORDS="${CANDIDATE_HEAD_WORDS:-100}"',
            "ASSUMPTION_BUDGET",
            "AUDIT_SAMPLE_SIZE_PER_OUTCOME",
            'MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"',
            'TEMPERATURE="${TEMPERATURE:-0.0}"',
            'TOP_P="${TOP_P:-1.0}"',
            'MIN_P="${MIN_P:-0.0}"',
            'TOP_K="${TOP_K:-0}"',
            'REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"',
            'PLOT_DPI="${PLOT_DPI:-300}"',
        ):
            self.assertIn(expected, text)
        self.assertNotIn("dispatch)", text)
        self.assertNotIn("experiments/exp1_relevance_bridge/run_exp1.sh", text)

        prepare_path = EXPERIMENT_ROOT / "prepare_exp1_representation.py"
        prepare_text = prepare_path.read_text(encoding="utf-8")
        self.assertIn('run_experiment(["--prepare_only", *argv])', prepare_text)
        self.assertNotIn("sbatch", prepare_text)
        self.assertNotIn("vllm", prepare_text)

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
        turn_budget = pd.read_csv(baseline.final_paths(output)["turn_budget_sanity"])
        self.assertEqual(
            turn_budget["representation_budget"].tolist(),
            list(baseline.DEFAULT_REPRESENTATION_BUDGETS),
        )
        self.assertTrue((turn_budget["mean_included_turns"].diff().dropna() >= 0).all())
        long_df = pd.read_csv(baseline.final_paths(output)["metrics_long"])
        self.assertTrue((long_df.loc[long_df["full_retained"] == True, "accuracy"] == 1).all())
        pairwise = pd.read_csv(baseline.final_paths(output)["pairwise"])
        self.assertTrue((pairwise.loc[pairwise["paired_sample_size"] > 0, "mean_improvement"].fillna(0) == 0).all())
        self.assertIn("output_hashes", summary)
        self.assertNotIn("summary", summary["output_hashes"])
        self.assertEqual(
            len(summary["complete_case_diagnostic_table"]),
            len(baseline.DEFAULT_CONDITIONS) * len(baseline.DEFAULT_FUTURE_HORIZONS),
        )

        self.assertFalse(summary["diagnostic_gate"]["ready_for_full_corpus"])
        decomposition = pd.read_csv(baseline.final_paths(output)["decomposition"])
        self.assertIn(
            "true_assumptions_vs_different_episode_control",
            set(decomposition["diagnostic_question"]),
        )
        audit = pd.read_csv(baseline.final_paths(output)["audit_sample"])
        self.assertTrue(set(audit["audit_outcome"]).issubset({"win", "loss", "tie"}))
        self.assertTrue((long_df["complete_case"] == True).any())

        resume_args = make_args(self.input_dir, output, "--dry_run", "--score_only")
        score_file = baseline.final_paths(output)["scores"]
        score_rows = baseline.read_jsonl(score_file)
        repairable_root = dict(
            score_rows[0],
            choice=None,
            positive_preference=None,
            parse_success=False,
            parse_error="invalid_json",
            raw_output=json.dumps(
                {
                    "answer": score_rows[0]["choice"],
                    "evidence": "The selected candidate is the better immediate continuation.",
                }
            ),
        )
        baseline.write_jsonl(score_file, [repairable_root, *score_rows[1:]])
        repaired_summary = baseline.analyze_dataset(args)
        self.assertEqual(repaired_summary["score_repair"]["repaired_score_count"], 1)
        repaired_root_rows = baseline.read_jsonl(score_file)
        self.assertTrue(repaired_root_rows[0]["parse_success"])
        self.assertEqual(repaired_root_rows[0]["parse_method"], "strict_json_answer_evidence")

        score_rows = repaired_root_rows
        old_pointwise_row = {
            "pair_id": score_rows[0]["pair_id"],
            "condition": score_rows[0]["condition"],
            "model_name": score_rows[0]["model_name"],
            "prompt_version": "representation-json-v3-diagnostic",
            "candidate_id": "legacy-candidate",
            "score": 7,
            "parse_success": True,
        }
        baseline.write_jsonl(score_file, [old_pointwise_row, *score_rows])
        resumed = baseline.score_dataset(resume_args)
        self.assertEqual(resumed["attempted_this_run"], 0)
        self.assertNotIn(old_pointwise_row, baseline.read_jsonl(score_file))

        score_rows = baseline.read_jsonl(score_file)
        invalid = dict(
            score_rows[0],
            choice=None,
            positive_preference=None,
            parse_success=False,
            parse_error="invalid_json",
            raw_output="invalid",
        )
        baseline.write_jsonl(score_file, [invalid, *score_rows[1:]])
        retried = baseline.score_dataset(resume_args)
        self.assertEqual(retried["attempted_this_run"], 1)
        self.assertEqual(retried["valid_task_count"], retried["expected_task_count"])

    def test_future_horizons_must_be_positive_and_odd(self) -> None:
        for value in (0, 2, -1):
            args = baseline.parse_args(["--future_horizons", str(value)])
            with self.assertRaisesRegex(ValueError, "positive odd integers"):
                baseline.validate_args(args)

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
        patch_zero_scores = baseline.score_path(baseline.patch_dir(output, 0, 2))
        patch_zero_rows = baseline.read_jsonl(patch_zero_scores)
        repairable = dict(
            patch_zero_rows[0],
            choice=None,
            positive_preference=None,
            parse_success=False,
            parse_error="invalid_json",
            raw_output=json.dumps(
                {
                    "answer": patch_zero_rows[0]["choice"],
                    "evidence": "The selected candidate is the better immediate continuation.",
                }
            ),
        )
        baseline.write_jsonl(patch_zero_scores, [repairable, *patch_zero_rows[1:]])
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
        self.assertEqual(merged["repaired_score_count"], 1)
        merged_rows = baseline.read_jsonl(baseline.final_paths(output)["scores"])
        repaired_rows = [row for row in merged_rows if baseline.task_key(row) == baseline.task_key(repairable)]
        self.assertEqual(len(repaired_rows), 1)
        self.assertTrue(repaired_rows[0]["parse_success"])
        self.assertEqual(repaired_rows[0]["parse_method"], "strict_json_answer_evidence")
        merged_order = [
            (
                row["pair_id"],
                row["condition"],
                int(row["negative_candidate_order"]),
                row["presentation_order"],
            )
            for row in merged_rows
        ]
        order_rank = {"positive_first": 0, "positive_second": 1}
        expected_order = sorted(
            merged_order,
            key=lambda row: (row[0], row[1], row[2], order_rank[row[3]]),
        )
        self.assertEqual(merged_order, expected_order)

        duplicate = baseline.read_jsonl(patch_zero_scores)[0]
        baseline.append_jsonl(patch_zero_scores, [duplicate])
        merged_again = baseline.merge_patch_scores(merge_args)
        self.assertEqual(merged_again["merged_score_count"], merged["merged_score_count"])

        conflicting = dict(duplicate, raw_output="conflicting duplicate")
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

    def test_patch_resume_drops_rows_from_an_old_partition(self) -> None:
        output = self.root / "repartitioned"
        prepare_args = make_args(self.input_dir, output, "--prepare_only")
        baseline.prepare_dataset(prepare_args)

        # First run: two episodes per patch. Patch 1 owns source paths 2 and 3.
        for patch_index in (0, 1):
            old_args = make_args(
                self.input_dir,
                output,
                "--dry_run",
                "--score_only",
                "--num_patches", "2",
                "--episodes_per_patch", "2",
                "--patch_index", str(patch_index),
            )
            baseline.score_dataset(old_args)

        # Second run: three episodes per patch. Source path 2 moves from patch 1 to patch 0.
        for patch_index in (0, 1):
            new_args = make_args(
                self.input_dir,
                output,
                "--dry_run",
                "--score_only",
                "--num_patches", "2",
                "--episodes_per_patch", "3",
                "--patch_index", str(patch_index),
            )
            baseline.score_dataset(new_args)

        prepared_pairs = baseline.read_jsonl(baseline.prepared_path(prepare_args))
        pair_to_path = {row["pair_id"]: row["source_path"] for row in prepared_pairs}
        for patch_index in (0, 1):
            patch = baseline.patch_dir(output, patch_index, 2)
            manifest = json.loads((patch / "patch_manifest.json").read_text(encoding="utf-8"))
            allowed_paths = set(manifest["selected_source_paths"])
            rows = baseline.read_jsonl(baseline.score_path(patch))
            self.assertTrue(rows)
            self.assertTrue(all(pair_to_path[row["pair_id"]] in allowed_paths for row in rows))

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

    def test_changed_prepared_dataset_invalidates_patch_resume(self) -> None:
        output = self.root / "changed-prepare"
        prepare_args = make_args(self.input_dir, output, "--prepare_only")
        baseline.prepare_dataset(prepare_args)
        score_args = make_args(self.input_dir, output, "--dry_run", "--score_only")
        first = baseline.score_dataset(score_args)
        self.assertGreater(first["attempted_this_run"], 0)

        episode_path = self.input_dir / "alpha" / "alpha-episode-0.json"
        turns = json.loads(episode_path.read_text(encoding="utf-8"))
        turns[1]["assumptions"] = ["changed assumption after first scoring run"]
        episode_path.write_text(json.dumps(turns), encoding="utf-8")

        baseline.prepare_dataset(prepare_args)
        rescored = baseline.score_dataset(score_args)
        self.assertEqual(rescored["attempted_this_run"], rescored["expected_task_count"])


if __name__ == "__main__":
    unittest.main()
