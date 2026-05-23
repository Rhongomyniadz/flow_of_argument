import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_INPUT_DIR = Path("data/conversation_moves_labeled")
DEFAULT_OUTPUT_DIR = Path("experiments/exp1_relevance_bridge/results")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_DOWNLOAD_DIR = Path("/shared/4/models")
DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS = 20
HARD_NEGATIVE_TARGET_COUNT = 24
HARD_NEGATIVE_LAYER_TARGET = 8
EXPECTED_CANDIDATE_COUNT = HARD_NEGATIVE_TARGET_COUNT + 1
PROMPT_CONDITIONS = ["without_assumptions", "with_assumptions"]
HEADLINE_CONSTRUCTIVE_MOVES = {
    "Assert / Elaborate",
    "Answer",
    "Agree / Align",
}


PromptCondition = Literal["without_assumptions", "with_assumptions"]


class TurnRecord(TypedDict):
    turn_id: str
    category: str
    episode_id: str
    turn_idx: int
    substantive_position: int
    move_label: str
    turn_text: str
    assumption_texts: list[str]


class PairRecord(TypedDict):
    category: str
    episode_id: str
    previous_turn_idx: int
    true_next_turn_idx: int
    pair_id: str
    previous_turn_id: str
    true_next_turn_id: str
    previous_turn_text: str
    previous_turn_assumptions: list[str]
    true_next_turn_text: str
    true_next_turn_move_label: str
    assumption_count: int
    negative_count: int
    candidate_count: int
    candidate_pool_complete: bool
    coverage_drop_reason: str | None


class CandidateRecord(TypedDict):
    pair_id: str
    candidate_id: str
    candidate_order: int
    candidate_turn_id: str
    candidate_category: str
    candidate_episode_id: str
    candidate_turn_idx: int
    candidate_move_label: str
    candidate_text: str
    candidate_assumptions: list[str]
    is_true_next_turn: bool
    negative_source: str


class PromptTask(TypedDict):
    pair_id: str
    candidate_id: str
    condition: PromptCondition
    prompt: str


class ParsedScore(TypedDict):
    score: int | None
    rationale: str | None
    confidence: float | None
    parse_success: bool
    parse_error: str | None


class CandidateScoreRecord(TypedDict):
    pair_id: str
    candidate_id: str
    candidate_order: int
    candidate_turn_id: str
    candidate_category: str
    candidate_episode_id: str
    candidate_turn_idx: int
    candidate_move_label: str
    candidate_text: str
    candidate_assumptions_json: str
    prompt_assumptions_json: str
    is_true_next_turn: bool
    negative_source: str
    condition: PromptCondition
    score: int | None
    rationale: str | None
    confidence: float | None
    parse_success: bool
    parse_error: str | None
    raw_output: str


class PairMetricRecord(TypedDict):
    category: str
    episode_id: str
    previous_turn_idx: int
    true_next_turn_idx: int
    pair_id: str
    previous_turn_text: str
    true_next_turn_text: str
    true_next_turn_move_label: str
    candidate_count: int
    negative_count: int
    assumption_count: int
    candidate_pool_complete: bool
    canonical_retained: bool
    coverage_drop_reason: str | None
    parsed_score_count: int
    expected_score_count: int
    true_score_without_assumptions: int | None
    true_score_with_assumptions: int | None
    true_rank_without_assumptions: int | None
    true_rank_with_assumptions: int | None
    rank_lift: int | None
    score_lift: int | None
    top1_without_assumptions: bool | None
    top1_with_assumptions: bool | None
    reciprocal_rank_without_assumptions: float | None
    reciprocal_rank_with_assumptions: float | None


class RankedCandidate(TypedDict):
    score: int
    rank: int
    is_true_next_turn: bool


class SummaryMetric(TypedDict):
    mean: float | None
    ci95_low: float | None
    ci95_high: float | None
    ci_unstable: bool
    cluster_count: int


class PatchPaths(TypedDict):
    pair_csv: Path
    candidate_jsonl: Path
    category_csv: Path
    move_csv: Path
    summary_json: Path


class LLMInterface:
    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        repetition_penalty: float,
        download_dir: Path,
        max_tokens: int,
    ) -> None:
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=str(download_dir),
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        self.params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.params)
        return [output.outputs[0].text.strip() for output in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max_episodes_per_category", type=int, default=None)
    parser.add_argument("--num_patches", type=int, default=1)
    parser.add_argument("--patch_index", type=int, default=0)
    parser.add_argument("--episodes_per_patch", type=int, default=None)
    parser.add_argument("--merge_patches_only", action="store_true")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--download_dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_patches < 1:
        raise ValueError(f"num_patches must be >= 1, got {args.num_patches}")
    if args.patch_index < 0 or args.patch_index >= args.num_patches:
        raise ValueError(f"patch_index must be in [0, {args.num_patches - 1}], got {args.patch_index}")
    if args.episodes_per_patch is not None and args.episodes_per_patch < 1:
        raise ValueError(f"episodes_per_patch must be >= 1, got {args.episodes_per_patch}")
    if args.prompt_batch_size < 1:
        raise ValueError(f"prompt_batch_size must be >= 1, got {args.prompt_batch_size}")
    if args.max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {args.max_tokens}")
    if args.bootstrap_draws < 1:
        raise ValueError(f"bootstrap_draws must be >= 1, got {args.bootstrap_draws}")


def resolve_patch_output_dir(base_output_dir: Path, num_patches: int, patch_index: int) -> Path:
    if num_patches == 1:
        return base_output_dir
    return base_output_dir / "patches" / f"patch_{patch_index:04d}_of_{num_patches:04d}"


def build_output_paths(output_dir: Path) -> PatchPaths:
    return {
        "pair_csv": output_dir / "exp1_llm_next_turn_pairs.csv",
        "candidate_jsonl": output_dir / "exp1_llm_next_turn_candidates.jsonl",
        "category_csv": output_dir / "exp1_llm_next_turn_by_category.csv",
        "move_csv": output_dir / "exp1_llm_next_turn_by_move.csv",
        "summary_json": output_dir / "exp1_summary.json",
    }


def normalize_categories(input_dir: Path, requested: list[str] | None) -> list[str]:
    available = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    if not requested or any(str(item).lower() == "all" for item in requested):
        return available
    lookup = {name.lower(): name for name in available}
    chosen: list[str] = []
    for raw_name in requested:
        match = lookup.get(str(raw_name).lower())
        if match is None:
            raise ValueError(f"Unknown category: {raw_name}. Available: {', '.join(available)}")
        if match not in chosen:
            chosen.append(match)
    return chosen


def collect_category_files(
    input_dir: Path,
    categories: list[str],
    max_episodes_per_category: int | None,
) -> list[tuple[str, Path]]:
    category_files: list[tuple[str, Path]] = []
    for category in categories:
        files = sorted((input_dir / category).glob("*.json"))
        if max_episodes_per_category is not None:
            files = files[:max_episodes_per_category]
        category_files.extend((category, path) for path in files)
    return category_files


def select_patch_files(
    category_files: list[tuple[str, Path]],
    num_patches: int,
    patch_index: int,
    episodes_per_patch: int | None,
) -> list[tuple[str, Path]]:
    if episodes_per_patch is not None:
        start = patch_index * episodes_per_patch
        end = min(start + episodes_per_patch, len(category_files))
        return category_files[start:end]
    return [item for index, item in enumerate(category_files) if index % num_patches == patch_index]


def load_turns(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("turns"), list):
        return [item for item in data["turns"] if isinstance(item, dict)]
    raise ValueError(f"Unrecognized episode JSON format: {path}")


def turn_time(turn: dict[str, object]) -> float:
    raw_time = turn.get("start_time", turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))))
    if isinstance(raw_time, (int, float)):
        return float(raw_time)
    return 0.0


def normalize_text_list(raw_items: object) -> list[str]:
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raise TypeError(f"Expected text list or list of text objects, got {type(raw_items).__name__}")
    texts: list[str] = []
    for index, item in enumerate(raw_items):
        raw_text: object
        if isinstance(item, dict):
            raw_text = item.get("text")
        else:
            raw_text = item
        if raw_text is None:
            continue
        if not isinstance(raw_text, str):
            raise TypeError(f"Text list item {index} must be a string or object with text, got {type(item).__name__}")
        text = raw_text.strip()
        if text:
            texts.append(text)
    return texts


def extract_turn_text(turn: dict[str, object]) -> str:
    for key in ["turn_text", "transcript", "text"]:
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    explicit_text = " ".join(normalize_text_list(turn.get("explicit_propositions"))).strip()
    if explicit_text:
        return explicit_text
    return ""


def build_seed_int(seed_text: str) -> int:
    return int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big", signed=False)


def build_rng(seed_text: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(build_seed_int(f"{seed_text}:{seed}"))


def sample_unique_ids(
    candidate_ids: list[str],
    sample_size: int,
    pair_id: str,
    sample_label: str,
    seed: int,
) -> list[str]:
    if sample_size <= 0 or not candidate_ids:
        return []
    unique_ids = list(dict.fromkeys(candidate_ids))
    if len(unique_ids) <= sample_size:
        return unique_ids
    rng = build_rng(f"{pair_id}:{sample_label}", seed)
    sampled_indices = rng.choice(len(unique_ids), size=sample_size, replace=False)
    return [unique_ids[int(index)] for index in np.asarray(sampled_indices).tolist()]


def build_episode_records(category: str, path: Path) -> tuple[list[TurnRecord], list[PairRecord]]:
    ordered_turns = sorted(load_turns(path), key=turn_time)
    turn_records: list[TurnRecord] = []
    source_by_list_index: dict[int, TurnRecord] = {}
    substantive_position = 0
    for list_index, turn in enumerate(ordered_turns):
        if str(turn.get("turn_type_label") or "").strip() != "Substantive":
            continue
        turn_text = extract_turn_text(turn)
        if not turn_text:
            continue
        episode_id = str(turn.get("episode_id") or path.stem)
        turn_idx = int(turn.get("turn_idx", list_index))
        substantive_position += 1
        record: TurnRecord = {
            "turn_id": f"{category}:{episode_id}:{turn_idx}",
            "category": category,
            "episode_id": episode_id,
            "turn_idx": turn_idx,
            "substantive_position": substantive_position,
            "move_label": str(turn.get("conversation_move_label") or "").strip(),
            "turn_text": turn_text,
            "assumption_texts": normalize_text_list(turn.get("assumptions")),
        }
        turn_records.append(record)
        source_by_list_index[list_index] = record

    pair_records: list[PairRecord] = []
    for list_index, next_list_index in zip(range(len(ordered_turns) - 1), range(1, len(ordered_turns))):
        previous_record = source_by_list_index.get(list_index)
        next_record = source_by_list_index.get(next_list_index)
        if previous_record is None or next_record is None:
            continue
        pair_id = f"{category}:{next_record['episode_id']}:{previous_record['turn_idx']}:{next_record['turn_idx']}"
        pair_records.append(
            {
                "category": category,
                "episode_id": next_record["episode_id"],
                "previous_turn_idx": previous_record["turn_idx"],
                "true_next_turn_idx": next_record["turn_idx"],
                "pair_id": pair_id,
                "previous_turn_id": previous_record["turn_id"],
                "true_next_turn_id": next_record["turn_id"],
                "previous_turn_text": previous_record["turn_text"],
                "previous_turn_assumptions": previous_record["assumption_texts"],
                "true_next_turn_text": next_record["turn_text"],
                "true_next_turn_move_label": next_record["move_label"],
                "assumption_count": len(previous_record["assumption_texts"]),
                "negative_count": 0,
                "candidate_count": 0,
                "candidate_pool_complete": False,
                "coverage_drop_reason": None,
            }
        )
    return turn_records, pair_records


def pair_sort_key(pair_record: PairRecord) -> tuple[str, str, int, int]:
    return (
        pair_record["category"],
        pair_record["episode_id"],
        pair_record["previous_turn_idx"],
        pair_record["true_next_turn_idx"],
    )


def collect_records(
    category_files: list[tuple[str, Path]],
    selected_files: list[tuple[str, Path]],
    use_tqdm: bool,
) -> tuple[list[TurnRecord], list[PairRecord]]:
    selected_paths = {path for _, path in selected_files}
    turn_records: list[TurnRecord] = []
    selected_pairs: list[PairRecord] = []
    iterator = tqdm(category_files, desc="Loading Exp 1 episodes", disable=not use_tqdm)
    for category, path in iterator:
        episode_turns, episode_pairs = build_episode_records(category, path)
        turn_records.extend(episode_turns)
        if path in selected_paths:
            selected_pairs.extend(episode_pairs)
    return turn_records, sorted(selected_pairs, key=pair_sort_key)


def build_turn_indexes(turn_records: list[TurnRecord]) -> dict[str, dict[str, object]]:
    turn_lookup: dict[str, TurnRecord] = {record["turn_id"]: record for record in turn_records}
    by_category_headline: dict[str, list[str]] = {}
    by_episode_substantive: dict[tuple[str, str], list[str]] = {}
    by_category_move: dict[tuple[str, str], list[str]] = {}
    by_category: dict[str, list[str]] = {}
    global_substantive: list[str] = []
    for record in sorted(turn_records, key=lambda item: (item["category"], item["episode_id"], item["turn_idx"])):
        turn_id = record["turn_id"]
        by_category.setdefault(record["category"], []).append(turn_id)
        by_category_move.setdefault((record["category"], record["move_label"]), []).append(turn_id)
        by_episode_substantive.setdefault((record["category"], record["episode_id"]), []).append(turn_id)
        global_substantive.append(turn_id)
        if record["move_label"] in HEADLINE_CONSTRUCTIVE_MOVES:
            by_category_headline.setdefault(record["category"], []).append(turn_id)
    return {
        "turn_lookup": turn_lookup,
        "by_category_headline": by_category_headline,
        "by_episode_substantive": by_episode_substantive,
        "by_category_move": by_category_move,
        "by_category": by_category,
        "global_substantive": {"all": global_substantive},
    }


def available_ids(candidate_ids: list[str], used_ids: set[str], true_next_turn_id: str) -> list[str]:
    return [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in used_ids and candidate_id != true_next_turn_id
    ]


def select_negative_ids(
    pair_record: PairRecord,
    turn_indexes: dict[str, dict[str, object]],
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, int], bool]:
    turn_lookup = turn_indexes["turn_lookup"]
    by_category_headline = turn_indexes["by_category_headline"]
    by_episode_substantive = turn_indexes["by_episode_substantive"]
    by_category_move = turn_indexes["by_category_move"]
    by_category = turn_indexes["by_category"]
    global_substantive = turn_indexes["global_substantive"]["all"]
    category = pair_record["category"]
    episode_id = pair_record["episode_id"]
    move_label = pair_record["true_next_turn_move_label"]
    pair_id = pair_record["pair_id"]
    true_next_turn_id = pair_record["true_next_turn_id"]
    true_record = turn_lookup.get(true_next_turn_id)
    true_position = int(true_record["substantive_position"]) if true_record is not None else -1
    selected_ids: list[tuple[str, str]] = []
    used_ids: set[str] = set()
    counts = {
        "negative_count_same_category_headline": 0,
        "negative_count_same_episode_gap3": 0,
        "negative_count_same_category_same_move": 0,
        "negative_count_same_category_backfill": 0,
        "negative_count_global_backfill": 0,
    }

    layers = [
        (
            "same_category_headline",
            [
                candidate_id
                for candidate_id in by_category_headline.get(category, [])
                if turn_lookup[candidate_id]["episode_id"] != episode_id
            ],
            HARD_NEGATIVE_LAYER_TARGET,
        ),
        (
            "same_episode_gap3",
            [
                candidate_id
                for candidate_id in by_episode_substantive.get((category, episode_id), [])
                if abs(int(turn_lookup[candidate_id]["substantive_position"]) - true_position) >= 3
            ],
            HARD_NEGATIVE_LAYER_TARGET,
        ),
        (
            "same_category_same_move",
            [
                candidate_id
                for candidate_id in by_category_move.get((category, move_label), [])
                if turn_lookup[candidate_id]["episode_id"] != episode_id
            ],
            HARD_NEGATIVE_LAYER_TARGET,
        ),
    ]
    for source_name, ids, target_count in layers:
        chosen = sample_unique_ids(
            available_ids(ids, used_ids, true_next_turn_id),
            target_count,
            pair_id,
            source_name,
            seed,
        )
        selected_ids.extend((candidate_id, source_name) for candidate_id in chosen)
        used_ids.update(chosen)
        counts[f"negative_count_{source_name}"] = len(chosen)

    if len(selected_ids) < HARD_NEGATIVE_TARGET_COUNT:
        same_category_backfill = [
            candidate_id
            for candidate_id in by_category.get(category, [])
            if turn_lookup[candidate_id]["episode_id"] != episode_id
        ]
        needed = HARD_NEGATIVE_TARGET_COUNT - len(selected_ids)
        chosen = sample_unique_ids(
            available_ids(same_category_backfill, used_ids, true_next_turn_id),
            needed,
            pair_id,
            "same_category_backfill",
            seed,
        )
        selected_ids.extend((candidate_id, "same_category_backfill") for candidate_id in chosen)
        used_ids.update(chosen)
        counts["negative_count_same_category_backfill"] = len(chosen)

    if len(selected_ids) < HARD_NEGATIVE_TARGET_COUNT:
        global_backfill = [
            candidate_id
            for candidate_id in global_substantive
            if turn_lookup[candidate_id]["episode_id"] != episode_id
        ]
        needed = HARD_NEGATIVE_TARGET_COUNT - len(selected_ids)
        chosen = sample_unique_ids(
            available_ids(global_backfill, used_ids, true_next_turn_id),
            needed,
            pair_id,
            "global_backfill",
            seed,
        )
        selected_ids.extend((candidate_id, "global_backfill") for candidate_id in chosen)
        counts["negative_count_global_backfill"] = len(chosen)

    return selected_ids, counts, len(selected_ids) == HARD_NEGATIVE_TARGET_COUNT


def build_candidate_id(pair_id: str, turn_id: str) -> str:
    digest = hashlib.sha256(f"{pair_id}:{turn_id}".encode("utf-8")).hexdigest()[:16]
    return f"{pair_id}:candidate:{digest}"


def build_pair_candidates(
    pair_record: PairRecord,
    turn_indexes: dict[str, dict[str, object]],
    seed: int,
) -> tuple[PairRecord, list[CandidateRecord], dict[str, int]]:
    turn_lookup = turn_indexes["turn_lookup"]
    negative_ids, counts, complete = select_negative_ids(pair_record, turn_indexes, seed)
    updated_pair = dict(pair_record)
    updated_pair["negative_count"] = len(negative_ids)
    updated_pair["candidate_count"] = 1 + len(negative_ids)
    updated_pair["candidate_pool_complete"] = complete
    if not complete:
        updated_pair["coverage_drop_reason"] = "insufficient_unique_negatives"
        return updated_pair, [], counts

    true_turn = turn_lookup[pair_record["true_next_turn_id"]]
    candidates: list[tuple[TurnRecord, bool, str]] = [(true_turn, True, "true_next_turn")]
    for negative_id, negative_source in negative_ids:
        candidates.append((turn_lookup[negative_id], False, negative_source))

    rng = build_rng(f"{pair_record['pair_id']}:candidate_order", seed)
    order = np.asarray(rng.permutation(len(candidates))).tolist()
    candidate_records: list[CandidateRecord] = []
    for candidate_order, raw_index in enumerate(order):
        turn_record, is_true, negative_source = candidates[int(raw_index)]
        candidate_records.append(
            {
                "pair_id": pair_record["pair_id"],
                "candidate_id": build_candidate_id(pair_record["pair_id"], turn_record["turn_id"]),
                "candidate_order": candidate_order,
                "candidate_turn_id": turn_record["turn_id"],
                "candidate_category": turn_record["category"],
                "candidate_episode_id": turn_record["episode_id"],
                "candidate_turn_idx": turn_record["turn_idx"],
                "candidate_move_label": turn_record["move_label"],
                "candidate_text": turn_record["turn_text"],
                "candidate_assumptions": turn_record["assumption_texts"],
                "is_true_next_turn": is_true,
                "negative_source": negative_source,
            }
        )
    return updated_pair, candidate_records, counts


def format_assumption_block(assumption_texts: list[str]) -> str:
    if not assumption_texts:
        return "None provided."
    return "\n".join(f"- {text}" for text in assumption_texts)


def build_scoring_prompt(pair_record: PairRecord, candidate_record: CandidateRecord, condition: PromptCondition) -> str:
    assumption_block = ""
    if condition == "with_assumptions":
        assumption_block = f"""

Previous-turn assumptions:
These are implicit premises likely taken for granted in the previous turn. Use them only as context for judging whether the candidate follows naturally.
{format_assumption_block(pair_record["previous_turn_assumptions"])}
"""
    return f"""You are a strict conversation-continuation judge.

Task:
Given a previous conversation turn and one candidate next turn, rate how likely the candidate is the true immediate next turn.

Use this 1-10 scale:
1 = impossible or totally unrelated
3 = weak fit
5 = plausible but uncertain
7 = strong local continuation
10 = almost certainly the immediate next turn

Previous turn:
{pair_record["previous_turn_text"]}

Candidate next turn:
{candidate_record["candidate_text"]}{assumption_block}

Return ONLY a raw JSON object with exactly these keys:
{{"score": <integer 1-10>, "rationale": "<brief reason>", "confidence": <number 0-1>}}
"""


def build_prompt_tasks(pair_records: list[PairRecord], candidate_records: list[CandidateRecord]) -> list[PromptTask]:
    pair_lookup = {pair_record["pair_id"]: pair_record for pair_record in pair_records}
    tasks: list[PromptTask] = []
    for candidate_record in candidate_records:
        pair_record = pair_lookup[candidate_record["pair_id"]]
        for condition in PROMPT_CONDITIONS:
            typed_condition: PromptCondition = "with_assumptions" if condition == "with_assumptions" else "without_assumptions"
            tasks.append(
                {
                    "pair_id": candidate_record["pair_id"],
                    "candidate_id": candidate_record["candidate_id"],
                    "condition": typed_condition,
                    "prompt": build_scoring_prompt(pair_record, candidate_record, typed_condition),
                }
            )
    return tasks


def safe_json_extract(text: str) -> dict[str, object] | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        return None
                    if isinstance(parsed, dict):
                        return parsed
                    return None
    return None


def stringify_optional(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def parse_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


def parse_llm_score(raw_output: str) -> ParsedScore:
    parsed = safe_json_extract(raw_output)
    if parsed is None:
        return {
            "score": None,
            "rationale": None,
            "confidence": None,
            "parse_success": False,
            "parse_error": "missing_json_object",
        }
    raw_score = parsed.get("score")
    rationale = stringify_optional(parsed.get("rationale"))
    confidence = parse_confidence(parsed.get("confidence"))
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return {
            "score": None,
            "rationale": rationale,
            "confidence": confidence,
            "parse_success": False,
            "parse_error": "score_not_numeric",
        }
    score = int(raw_score)
    if float(raw_score) != float(score) or score < 1 or score > 10:
        return {
            "score": None,
            "rationale": rationale,
            "confidence": confidence,
            "parse_success": False,
            "parse_error": "score_out_of_range",
        }
    return {
        "score": score,
        "rationale": rationale,
        "confidence": confidence,
        "parse_success": True,
        "parse_error": None,
    }


def score_candidates(
    args: argparse.Namespace,
    pair_records: list[PairRecord],
    candidate_records: list[CandidateRecord],
    use_tqdm: bool,
) -> list[CandidateScoreRecord]:
    if not candidate_records:
        return []
    tasks = build_prompt_tasks(pair_records, candidate_records)
    pair_lookup = {record["pair_id"]: record for record in pair_records}
    candidate_lookup = {record["candidate_id"]: record for record in candidate_records}
    llm = LLMInterface(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        download_dir=args.download_dir,
        max_tokens=args.max_tokens,
    )
    score_records: list[CandidateScoreRecord] = []
    iterator = tqdm(
        range(0, len(tasks), args.prompt_batch_size),
        desc="Scoring Exp 1 LLM prompts",
        disable=not use_tqdm,
    )
    for start in iterator:
        batch_tasks = tasks[start:start + args.prompt_batch_size]
        raw_outputs = llm.generate_batch([task["prompt"] for task in batch_tasks])
        for task, raw_output in zip(batch_tasks, raw_outputs):
            parsed = parse_llm_score(raw_output)
            candidate = candidate_lookup[task["candidate_id"]]
            pair_record = pair_lookup[task["pair_id"]]
            score_records.append(
                {
                    "pair_id": candidate["pair_id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_order": candidate["candidate_order"],
                    "candidate_turn_id": candidate["candidate_turn_id"],
                    "candidate_category": candidate["candidate_category"],
                    "candidate_episode_id": candidate["candidate_episode_id"],
                    "candidate_turn_idx": candidate["candidate_turn_idx"],
                    "candidate_move_label": candidate["candidate_move_label"],
                    "candidate_text": candidate["candidate_text"],
                    "candidate_assumptions_json": json.dumps(candidate["candidate_assumptions"], ensure_ascii=False),
                    "prompt_assumptions_json": json.dumps(pair_record["previous_turn_assumptions"], ensure_ascii=False),
                    "is_true_next_turn": candidate["is_true_next_turn"],
                    "negative_source": candidate["negative_source"],
                    "condition": task["condition"],
                    "score": parsed["score"],
                    "rationale": parsed["rationale"],
                    "confidence": parsed["confidence"],
                    "parse_success": parsed["parse_success"],
                    "parse_error": parsed["parse_error"],
                    "raw_output": raw_output,
                }
            )
    return score_records


def rank_condition_scores(condition_scores: list[CandidateScoreRecord]) -> list[RankedCandidate]:
    sortable: list[tuple[int, int, CandidateScoreRecord]] = []
    for record in condition_scores:
        score = record["score"]
        if score is None:
            continue
        sortable.append((int(score), int(record["candidate_order"]), record))
    ordered = sorted(sortable, key=lambda item: (-item[0], item[1]))
    ranked: list[RankedCandidate] = []
    for rank_index, (score, _, record) in enumerate(ordered, start=1):
        ranked.append(
            {
                "score": score,
                "rank": rank_index,
                "is_true_next_turn": record["is_true_next_turn"],
            }
        )
    return ranked


def require_int_metric(metrics: dict[str, object], key: str) -> int:
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Expected integer metric for {key}, got {type(value).__name__}")
    return int(value)


def build_pair_metrics(pair_records: list[PairRecord], score_records: list[CandidateScoreRecord]) -> pd.DataFrame:
    scores_by_pair_condition: dict[tuple[str, PromptCondition], list[CandidateScoreRecord]] = {}
    for record in score_records:
        scores_by_pair_condition.setdefault((record["pair_id"], record["condition"]), []).append(record)
    rows: list[PairMetricRecord] = []
    for pair_record in pair_records:
        base: PairMetricRecord = {
            "category": pair_record["category"],
            "episode_id": pair_record["episode_id"],
            "previous_turn_idx": pair_record["previous_turn_idx"],
            "true_next_turn_idx": pair_record["true_next_turn_idx"],
            "pair_id": pair_record["pair_id"],
            "previous_turn_text": pair_record["previous_turn_text"],
            "true_next_turn_text": pair_record["true_next_turn_text"],
            "true_next_turn_move_label": pair_record["true_next_turn_move_label"],
            "candidate_count": pair_record["candidate_count"],
            "negative_count": pair_record["negative_count"],
            "assumption_count": pair_record["assumption_count"],
            "candidate_pool_complete": pair_record["candidate_pool_complete"],
            "canonical_retained": False,
            "coverage_drop_reason": pair_record["coverage_drop_reason"],
            "parsed_score_count": 0,
            "expected_score_count": EXPECTED_CANDIDATE_COUNT * len(PROMPT_CONDITIONS),
            "true_score_without_assumptions": None,
            "true_score_with_assumptions": None,
            "true_rank_without_assumptions": None,
            "true_rank_with_assumptions": None,
            "rank_lift": None,
            "score_lift": None,
            "top1_without_assumptions": None,
            "top1_with_assumptions": None,
            "reciprocal_rank_without_assumptions": None,
            "reciprocal_rank_with_assumptions": None,
        }
        if not pair_record["candidate_pool_complete"]:
            rows.append(base)
            continue
        metrics: dict[str, object] = {}
        parsed_score_count = 0
        condition_failed = False
        for condition in PROMPT_CONDITIONS:
            typed_condition: PromptCondition = "with_assumptions" if condition == "with_assumptions" else "without_assumptions"
            condition_scores = scores_by_pair_condition.get((pair_record["pair_id"], typed_condition), [])
            parsed_scores = [record for record in condition_scores if record["parse_success"] and record["score"] is not None]
            parsed_score_count += len(parsed_scores)
            if len(parsed_scores) != EXPECTED_CANDIDATE_COUNT:
                condition_failed = True
                continue
            true_ranked = [record for record in rank_condition_scores(parsed_scores) if record["is_true_next_turn"]]
            if len(true_ranked) != 1:
                condition_failed = True
                continue
            true_record = true_ranked[0]
            metrics[f"true_score_{condition}"] = true_record["score"]
            metrics[f"true_rank_{condition}"] = true_record["rank"]
            metrics[f"top1_{condition}"] = true_record["rank"] == 1
            metrics[f"reciprocal_rank_{condition}"] = 1.0 / float(true_record["rank"])
        base["parsed_score_count"] = parsed_score_count
        if condition_failed:
            base["coverage_drop_reason"] = "score_parse_failed"
            rows.append(base)
            continue
        base["canonical_retained"] = True
        base["coverage_drop_reason"] = None
        base["true_score_without_assumptions"] = require_int_metric(metrics, "true_score_without_assumptions")
        base["true_score_with_assumptions"] = require_int_metric(metrics, "true_score_with_assumptions")
        base["true_rank_without_assumptions"] = require_int_metric(metrics, "true_rank_without_assumptions")
        base["true_rank_with_assumptions"] = require_int_metric(metrics, "true_rank_with_assumptions")
        base["rank_lift"] = base["true_rank_without_assumptions"] - base["true_rank_with_assumptions"]
        base["score_lift"] = base["true_score_with_assumptions"] - base["true_score_without_assumptions"]
        base["top1_without_assumptions"] = bool(metrics["top1_without_assumptions"])
        base["top1_with_assumptions"] = bool(metrics["top1_with_assumptions"])
        base["reciprocal_rank_without_assumptions"] = float(metrics["reciprocal_rank_without_assumptions"])
        base["reciprocal_rank_with_assumptions"] = float(metrics["reciprocal_rank_with_assumptions"])
        rows.append(base)
    return pd.DataFrame(rows).sort_values(
        by=["category", "episode_id", "previous_turn_idx", "true_next_turn_idx"],
        kind="stable",
    ).reset_index(drop=True)


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    value_column: str,
    seed_label: str,
    seed: int,
    draws: int,
) -> SummaryMetric:
    valid = df[df[value_column].notna()].copy()
    if valid.empty:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci_unstable": True,
            "cluster_count": 0,
        }
    valid["_cluster_key"] = valid["category"].astype(str) + "||" + valid["episode_id"].astype(str)
    clusters = [group[value_column].to_numpy(dtype=np.float64) for _, group in valid.groupby("_cluster_key", sort=False)]
    cluster_count = len(clusters)
    point_estimate = float(valid[value_column].mean())
    if cluster_count < DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS:
        return {
            "mean": point_estimate,
            "ci95_low": None,
            "ci95_high": None,
            "ci_unstable": True,
            "cluster_count": cluster_count,
        }
    rng = build_rng(seed_label, seed)
    draw_values: list[float] = []
    for _ in range(draws):
        sampled_indices = rng.integers(0, cluster_count, size=cluster_count)
        sampled_values = np.concatenate([clusters[int(index)] for index in sampled_indices])
        draw_values.append(float(sampled_values.mean()))
    alpha = 1.0 - DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL
    return {
        "mean": point_estimate,
        "ci95_low": float(np.quantile(draw_values, alpha / 2.0)),
        "ci95_high": float(np.quantile(draw_values, 1.0 - alpha / 2.0)),
        "ci_unstable": False,
        "cluster_count": cluster_count,
    }


def build_group_summary(df: pd.DataFrame, group_column: str, seed: int, draws: int) -> pd.DataFrame:
    retained = df[df["canonical_retained"] == True].copy()
    if retained.empty:
        return pd.DataFrame(columns=[group_column])
    rows: list[dict[str, object]] = []
    metric_columns = [
        ("mean_true_rank_without_assumptions", "true_rank_without_assumptions"),
        ("mean_true_rank_with_assumptions", "true_rank_with_assumptions"),
        ("top1_rate_without_assumptions", "top1_without_assumptions"),
        ("top1_rate_with_assumptions", "top1_with_assumptions"),
        ("mrr_without_assumptions", "reciprocal_rank_without_assumptions"),
        ("mrr_with_assumptions", "reciprocal_rank_with_assumptions"),
        ("mean_rank_lift", "rank_lift"),
        ("mean_score_lift", "score_lift"),
    ]
    for group_value, group_df in retained.groupby(group_column, sort=False, observed=False):
        row: dict[str, object] = {
            group_column: group_value,
            "pair_count": int(len(group_df)),
            "cluster_count": int(group_df[["category", "episode_id"]].drop_duplicates().shape[0]),
        }
        for output_column, metric_column in metric_columns:
            source_df = group_df.copy()
            if metric_column.startswith("top1_"):
                source_df[metric_column] = source_df[metric_column].astype(float)
            boot = cluster_bootstrap_mean(
                df=source_df,
                value_column=metric_column,
                seed_label=f"{group_column}:{group_value}:{metric_column}",
                seed=seed,
                draws=draws,
            )
            row[output_column] = boot["mean"]
            row[f"{output_column}_ci95_low"] = boot["ci95_low"]
            row[f"{output_column}_ci95_high"] = boot["ci95_high"]
            row[f"{output_column}_ci_unstable"] = boot["ci_unstable"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_candidate_jsonl(score_records: list[CandidateScoreRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in score_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_candidate_jsonl(path: Path) -> list[CandidateScoreRecord]:
    records: list[CandidateScoreRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"Candidate JSONL line {line_number} must be an object: {path}")
            records.append(item)
    return records


def coerce_nullable_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def coerce_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for column_name in ["candidate_pool_complete", "canonical_retained", "top1_without_assumptions", "top1_with_assumptions"]:
        if column_name in coerced.columns:
            coerced[column_name] = coerced[column_name].map(coerce_nullable_bool)
    for column_name in [
        "previous_turn_idx",
        "true_next_turn_idx",
        "candidate_count",
        "negative_count",
        "assumption_count",
        "parsed_score_count",
        "expected_score_count",
        "true_score_without_assumptions",
        "true_score_with_assumptions",
        "true_rank_without_assumptions",
        "true_rank_with_assumptions",
        "rank_lift",
        "score_lift",
        "reciprocal_rank_without_assumptions",
        "reciprocal_rank_with_assumptions",
    ]:
        if column_name in coerced.columns:
            coerced[column_name] = pd.to_numeric(coerced[column_name], errors="coerce")
    return coerced


def build_coverage_summary(pair_df: pd.DataFrame) -> dict[str, object]:
    return {
        "candidate_pool_incomplete_count": int((pair_df["coverage_drop_reason"] == "insufficient_unique_negatives").sum()),
        "score_parse_failed_count": int((pair_df["coverage_drop_reason"] == "score_parse_failed").sum()),
        "retained_pair_count": int((pair_df["canonical_retained"] == True).sum()),
        "dropped_pair_count": int((pair_df["canonical_retained"] != True).sum()),
    }


def build_headline_metrics(retained: pd.DataFrame) -> dict[str, object]:
    if retained.empty:
        return {
            "mean_true_rank_without_assumptions": None,
            "mean_true_rank_with_assumptions": None,
            "top1_rate_without_assumptions": None,
            "top1_rate_with_assumptions": None,
            "mrr_without_assumptions": None,
            "mrr_with_assumptions": None,
            "mean_rank_lift": None,
            "mean_score_lift": None,
        }
    return {
        "mean_true_rank_without_assumptions": float(retained["true_rank_without_assumptions"].mean()),
        "mean_true_rank_with_assumptions": float(retained["true_rank_with_assumptions"].mean()),
        "top1_rate_without_assumptions": float(retained["top1_without_assumptions"].astype(float).mean()),
        "top1_rate_with_assumptions": float(retained["top1_with_assumptions"].astype(float).mean()),
        "mrr_without_assumptions": float(retained["reciprocal_rank_without_assumptions"].mean()),
        "mrr_with_assumptions": float(retained["reciprocal_rank_with_assumptions"].mean()),
        "mean_rank_lift": float(retained["rank_lift"].mean()),
        "mean_score_lift": float(retained["score_lift"].mean()),
    }


def build_summary_payload(
    args: argparse.Namespace,
    output_dir: Path,
    pair_df: pd.DataFrame,
    category_summary: pd.DataFrame,
    move_summary: pd.DataFrame,
    output_paths: PatchPaths,
    categories: list[str],
    selected_episode_file_count: int,
    candidate_episode_file_count: int,
    analysis_stage: str,
    extra_sections: dict[str, object] | None,
) -> dict[str, object]:
    retained = pair_df[pair_df["canonical_retained"] == True].copy()
    payload: dict[str, object] = {
        "experiment": "Experiment 1: LLM Pointwise Next-Turn Ranking",
        "analysis_stage": analysis_stage,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "model_name": str(args.model_name),
        "download_dir": str(args.download_dir),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "prompt_batch_size": int(args.prompt_batch_size),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "categories": categories,
        "candidate_count_target": EXPECTED_CANDIDATE_COUNT,
        "negative_count_target": HARD_NEGATIVE_TARGET_COUNT,
        "selected_episode_file_count": int(selected_episode_file_count),
        "candidate_episode_file_count": int(candidate_episode_file_count),
        "pair_count": int(len(pair_df)),
        "retained_pair_count": int(len(retained)),
        "retained_pair_rate": float(len(retained) / len(pair_df)) if len(pair_df) > 0 else None,
        "coverage": build_coverage_summary(pair_df),
        "headline_metrics": build_headline_metrics(retained),
        "outputs": {
            "pair_csv": str(output_paths["pair_csv"]),
            "candidate_jsonl": str(output_paths["candidate_jsonl"]),
            "category_csv": str(output_paths["category_csv"]),
            "move_csv": str(output_paths["move_csv"]),
            "summary_json": str(output_paths["summary_json"]),
        },
        "category_summary": json.loads(category_summary.to_json(orient="records")),
        "move_summary": json.loads(move_summary.to_json(orient="records")),
    }
    if extra_sections is not None:
        payload.update(extra_sections)
    return payload


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    pair_df: pd.DataFrame,
    score_records: list[CandidateScoreRecord],
    categories: list[str],
    selected_episode_file_count: int,
    candidate_episode_file_count: int,
    analysis_stage: str,
    extra_sections: dict[str, object] | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(output_dir)
    pair_df.to_csv(output_paths["pair_csv"], index=False)
    write_candidate_jsonl(score_records, output_paths["candidate_jsonl"])
    category_summary = build_group_summary(pair_df, "category", args.seed, args.bootstrap_draws)
    move_summary = build_group_summary(pair_df, "true_next_turn_move_label", args.seed, args.bootstrap_draws)
    category_summary.to_csv(output_paths["category_csv"], index=False)
    move_summary.to_csv(output_paths["move_csv"], index=False)
    summary = build_summary_payload(
        args=args,
        output_dir=output_dir,
        pair_df=pair_df,
        category_summary=category_summary,
        move_summary=move_summary,
        output_paths=output_paths,
        categories=categories,
        selected_episode_file_count=selected_episode_file_count,
        candidate_episode_file_count=candidate_episode_file_count,
        analysis_stage=analysis_stage,
        extra_sections=extra_sections,
    )
    output_paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_candidate_pool(
    pair_records: list[PairRecord],
    turn_indexes: dict[str, dict[str, object]],
    seed: int,
) -> tuple[list[PairRecord], list[CandidateRecord], dict[str, int]]:
    updated_pairs: list[PairRecord] = []
    candidates: list[CandidateRecord] = []
    aggregate_counts = {
        "negative_count_same_category_headline": 0,
        "negative_count_same_episode_gap3": 0,
        "negative_count_same_category_same_move": 0,
        "negative_count_same_category_backfill": 0,
        "negative_count_global_backfill": 0,
    }
    for pair_record in pair_records:
        updated_pair, pair_candidates, counts = build_pair_candidates(pair_record, turn_indexes, seed)
        updated_pairs.append(updated_pair)
        candidates.extend(pair_candidates)
        for key, value in counts.items():
            aggregate_counts[key] += value
    return updated_pairs, candidates, aggregate_counts


def collect_patch_dirs(output_dir: Path) -> list[Path]:
    patches_dir = output_dir / "patches"
    if not patches_dir.exists():
        raise RuntimeError(f"Patches directory does not exist: {patches_dir}")
    patch_dirs = sorted(
        path
        for path in patches_dir.glob("patch_*_of_*")
        if path.is_dir()
        and (path / "exp1_llm_next_turn_pairs.csv").exists()
        and (path / "exp1_llm_next_turn_candidates.jsonl").exists()
    )
    if not patch_dirs:
        raise RuntimeError(f"No Exp 1 LLM patch outputs were found in {patches_dir}")
    return patch_dirs


def merge_patch_outputs(args: argparse.Namespace, categories: list[str], category_files: list[tuple[str, Path]]) -> None:
    patch_dirs = collect_patch_dirs(args.output_dir)
    pair_frames = [coerce_pair_frame(pd.read_csv(path / "exp1_llm_next_turn_pairs.csv")) for path in patch_dirs]
    pair_df = pd.concat(pair_frames, ignore_index=True)
    duplicate_mask = pair_df.duplicated(subset=["pair_id"], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        logger.warning("Dropping duplicate merged Exp 1 pair rows: duplicate_count=%d", duplicate_count)
        pair_df = pair_df.drop_duplicates(subset=["pair_id"], keep="first")
    pair_df = pair_df.sort_values(
        by=["category", "episode_id", "previous_turn_idx", "true_next_turn_idx"],
        kind="stable",
    ).reset_index(drop=True)
    score_records: list[CandidateScoreRecord] = []
    for patch_dir in patch_dirs:
        score_records.extend(read_candidate_jsonl(patch_dir / "exp1_llm_next_turn_candidates.jsonl"))
    summary = write_outputs(
        args=args,
        output_dir=args.output_dir,
        pair_df=pair_df,
        score_records=score_records,
        categories=categories,
        selected_episode_file_count=int(pair_df[["category", "episode_id"]].drop_duplicates().shape[0]),
        candidate_episode_file_count=len(category_files),
        analysis_stage="merged_full_analysis",
        extra_sections={
            "patch_merge": {
                "merged_patch_count": int(len(patch_dirs)),
                "patch_dirs": [str(path) for path in patch_dirs],
                "num_patches": int(args.num_patches),
                "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
            }
        },
    )
    print(json.dumps(summary, indent=2))


def write_dry_run_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    pair_records: list[PairRecord],
    candidate_records: list[CandidateRecord],
    categories: list[str],
    selected_episode_file_count: int,
    candidate_episode_file_count: int,
    aggregate_counts: dict[str, int],
) -> None:
    score_records: list[CandidateScoreRecord] = []
    pair_lookup = {pair_record["pair_id"]: pair_record for pair_record in pair_records}
    for candidate in candidate_records:
        pair_record = pair_lookup[candidate["pair_id"]]
        for condition in PROMPT_CONDITIONS:
            typed_condition: PromptCondition = "with_assumptions" if condition == "with_assumptions" else "without_assumptions"
            score_records.append(
                {
                    "pair_id": candidate["pair_id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_order": candidate["candidate_order"],
                    "candidate_turn_id": candidate["candidate_turn_id"],
                    "candidate_category": candidate["candidate_category"],
                    "candidate_episode_id": candidate["candidate_episode_id"],
                    "candidate_turn_idx": candidate["candidate_turn_idx"],
                    "candidate_move_label": candidate["candidate_move_label"],
                    "candidate_text": candidate["candidate_text"],
                    "candidate_assumptions_json": json.dumps(candidate["candidate_assumptions"], ensure_ascii=False),
                    "prompt_assumptions_json": json.dumps(pair_record["previous_turn_assumptions"], ensure_ascii=False),
                    "is_true_next_turn": candidate["is_true_next_turn"],
                    "negative_source": candidate["negative_source"],
                    "condition": typed_condition,
                    "score": 10 if candidate["is_true_next_turn"] else 1,
                    "rationale": "dry_run",
                    "confidence": 1.0,
                    "parse_success": True,
                    "parse_error": None,
                    "raw_output": '{"score": 10, "rationale": "dry_run", "confidence": 1.0}',
                }
            )
    pair_df = build_pair_metrics(pair_records, score_records)
    write_outputs(
        args=args,
        output_dir=output_dir,
        pair_df=pair_df,
        score_records=score_records,
        categories=categories,
        selected_episode_file_count=selected_episode_file_count,
        candidate_episode_file_count=candidate_episode_file_count,
        analysis_stage="dry_run_patch_validation",
        extra_sections={"negative_sampling_counts": aggregate_counts},
    )


def run_patch(args: argparse.Namespace, categories: list[str], category_files: list[tuple[str, Path]]) -> None:
    use_tqdm = not args.no_tqdm
    selected_files = select_patch_files(
        category_files=category_files,
        num_patches=args.num_patches,
        patch_index=args.patch_index,
        episodes_per_patch=args.episodes_per_patch,
    )
    if not selected_files:
        raise RuntimeError(
            f"No episode files selected for patch {args.patch_index} out of {args.num_patches}. "
            f"candidate_file_count={len(category_files)}"
        )
    output_dir = resolve_patch_output_dir(args.output_dir, args.num_patches, args.patch_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    turn_records, selected_pairs = collect_records(
        category_files=category_files,
        selected_files=selected_files,
        use_tqdm=use_tqdm,
    )
    turn_indexes = build_turn_indexes(turn_records)
    pair_records, candidate_records, aggregate_counts = build_candidate_pool(
        pair_records=selected_pairs,
        turn_indexes=turn_indexes,
        seed=args.seed,
    )
    if args.dry_run:
        write_dry_run_outputs(
            args=args,
            output_dir=output_dir,
            pair_records=pair_records,
            candidate_records=candidate_records,
            categories=categories,
            selected_episode_file_count=len(selected_files),
            candidate_episode_file_count=len(category_files),
            aggregate_counts=aggregate_counts,
        )
        logger.info("Done. Wrote dry-run Exp 1 LLM validation outputs to %s", output_dir)
        return
    complete_pairs = [pair_record for pair_record in pair_records if pair_record["candidate_pool_complete"]]
    score_records = score_candidates(
        args=args,
        pair_records=complete_pairs,
        candidate_records=candidate_records,
        use_tqdm=use_tqdm,
    )
    pair_df = build_pair_metrics(pair_records, score_records)
    summary = write_outputs(
        args=args,
        output_dir=output_dir,
        pair_df=pair_df,
        score_records=score_records,
        categories=categories,
        selected_episode_file_count=len(selected_files),
        candidate_episode_file_count=len(category_files),
        analysis_stage="patch_pair_scoring_only" if args.num_patches > 1 else "full_analysis",
        extra_sections={
            "num_patches": int(args.num_patches),
            "patch_index": int(args.patch_index),
            "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
            "negative_sampling_counts": aggregate_counts,
        },
    )
    logger.info("Done. Wrote Exp 1 LLM results to %s", output_dir)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    validate_args(args)
    categories = normalize_categories(args.input_dir, args.categories)
    category_files = collect_category_files(args.input_dir, categories, args.max_episodes_per_category)
    if not category_files:
        raise RuntimeError(
            "No episode files matched Exp 1 inputs. "
            f"input_dir={args.input_dir}, categories={categories}"
        )
    if args.merge_patches_only:
        merge_patch_outputs(args, categories, category_files)
        return
    run_patch(args, categories, category_files)


if __name__ == "__main__":
    main()
