from __future__ import annotations

"""Diagnostic explicit/implicit representation decomposition for next-turn ranking."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TypedDict

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SCRIPT_VERSION = "2.0.0"
PROMPT_VERSION = "representation-diagnostic-v2"
DEFAULT_INPUT_DIR = Path("data/conversation_moves_labeled")
DEFAULT_OUTPUT_ROOT = Path("iclr/exp1_representation_baselines/results")
DEFAULT_PREPARED_NAME = "exp1_representation_prepared_pairs.jsonl"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_DOWNLOAD_DIR = Path("/shared/4/models")
DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS = 20
HARD_NEGATIVE_TARGET_COUNT = 24
HARD_NEGATIVE_LAYER_TARGET = 8
EXPECTED_CANDIDATE_COUNT = 25
EMPTY_EXPLICIT = "None extracted."
EMPTY_ASSUMPTIONS = "None extracted."
EMPTY_HISTORY = "No earlier substantive turn available."

DEFAULT_CONDITIONS = (
    "raw_turn",
    "raw_turn_with_history",
    "raw_turn_plus_assumptions",
    "explicit_only",
    "explicit_plus_top1_assumption",
    "explicit_plus_top3_assumptions",
    "explicit_plus_assumptions",
)
OPTIONAL_CONDITIONS = (
    "assumptions_only",
    "explicit_plus_shuffled_assumptions",
    "explicit_plus_wrong_episode_assumptions",
)
ALL_CONDITIONS = DEFAULT_CONDITIONS + OPTIONAL_CONDITIONS
CONTROL_CONDITIONS = (
    "explicit_plus_shuffled_assumptions",
    "explicit_plus_wrong_episode_assumptions",
)
HEADLINE_CONSTRUCTIVE_MOVES = {
    "Assert / Elaborate",
    "Answer",
    "Agree / Align",
}
DIAGNOSTIC_CONTRASTS = (
    ("explicit_plus_assumptions", "explicit_only"),
    ("raw_turn_plus_assumptions", "raw_turn"),
    ("raw_turn", "explicit_only"),
    ("raw_turn_with_history", "raw_turn"),
    ("explicit_plus_top1_assumption", "explicit_only"),
    ("explicit_plus_top3_assumptions", "explicit_only"),
    ("explicit_plus_assumptions", "explicit_plus_top1_assumption"),
    ("explicit_plus_assumptions", "explicit_plus_top3_assumptions"),
)
OPTIONAL_CONTROL_CONTRASTS = (
    ("explicit_plus_assumptions", "assumptions_only"),
    ("explicit_plus_assumptions", "explicit_plus_shuffled_assumptions"),
    ("explicit_plus_assumptions", "explicit_plus_wrong_episode_assumptions"),
)
REQUIRED_CONTRASTS = DIAGNOSTIC_CONTRASTS + OPTIONAL_CONTROL_CONTRASTS


class TurnRecord(TypedDict):
    turn_id: str
    category: str
    episode_id: str
    turn_idx: int
    list_position: int
    substantive_position: int
    timestamp: float
    move_label: str
    turn_text: str
    explicit_texts: list[str]
    assumption_texts: list[str]
    history_turn_ids: list[str]
    history_turn_texts: list[str]


class CandidateRecord(TypedDict):
    candidate_id: str
    candidate_order: int
    candidate_turn_id: str
    candidate_category: str
    candidate_episode_id: str
    candidate_turn_idx: int
    candidate_move_label: str
    candidate_text: str
    is_true_next_turn: bool
    negative_source: str


class ParsedScore(TypedDict):
    score: int | None
    rationale: str | None
    confidence: float | None
    parse_success: bool
    parse_error: str | None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seed_int(seed_text: str) -> int:
    return int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")


def build_rng(seed_text: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(seed_int(f"{seed_text}:{seed}"))


def donor_rng(pair_id: str, condition_id: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(seed_int(pair_id + condition_id + str(seed)))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized)
    atomic_write_text(path, text)
    return len(materialized)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "matplotlib", "torch", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def normalize_conditions(values: list[str] | None) -> list[str]:
    if not values or any(value.casefold() == "all" for value in values):
        return list(DEFAULT_CONDITIONS)
    chosen: list[str] = []
    for value in values:
        if value not in ALL_CONDITIONS:
            raise ValueError(f"Unknown condition {value!r}; choose from {', '.join(ALL_CONDITIONS)}")
        if value not in chosen:
            chosen.append(value)
    if not chosen:
        raise ValueError("At least one condition is required")
    return chosen


def model_output_name(model_name: str) -> str:
    """Return a filesystem-safe, reversible-enough model identifier."""
    normalized = model_name.strip().replace("\\", "/")
    slug = normalized.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", slug).strip("._-")
    if not slug:
        raise ValueError("model_name must contain at least one filesystem-safe character")
    return slug


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Exact output directory. By default, results/<model-name-with-slashes-as-__> is used.",
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max_episodes_per_category", type=int, default=None)
    parser.add_argument("--num_patches", type=int, default=1)
    parser.add_argument("--patch_index", type=int, default=0)
    parser.add_argument("--episodes_per_patch", type=int, default=None)
    parser.add_argument("--merge_patches_only", action="store_true")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--download_dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Use deterministic fake scores; do not load a model.")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--prepared_pairs_jsonl", type=Path, default=None)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--strict_all_conditions", action="store_true")
    parser.add_argument(
        "--save_source_representation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite_scores", action="store_true")
    parser.add_argument(
        "--audit_sample_size_per_outcome",
        type=int,
        default=25,
        help="Number of strongest wins, losses, and deterministic ties to emit for manual audit.",
    )
    parser.add_argument("--plot_dpi", type=int, default=300)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    stage_count = sum(
        bool(value)
        for value in (args.prepare_only, args.score_only, args.analysis_only, args.merge_patches_only)
    )
    if stage_count > 1:
        raise ValueError("Preparation, scoring, merge, and analysis stage flags are mutually exclusive")
    if args.num_patches < 1:
        raise ValueError("num_patches must be at least one")
    if not 0 <= args.patch_index < args.num_patches:
        raise ValueError(f"patch_index must be in [0, {args.num_patches - 1}]")
    if args.episodes_per_patch is not None and args.episodes_per_patch < 1:
        raise ValueError("episodes_per_patch must be positive")
    if args.max_episodes_per_category is not None and args.max_episodes_per_category < 1:
        raise ValueError("max_episodes_per_category must be positive")
    if args.history_turns < 0:
        raise ValueError("history_turns cannot be negative")
    if args.prompt_batch_size < 1 or args.max_tokens < 1 or args.bootstrap_draws < 1:
        raise ValueError("Batch size, max tokens, and bootstrap draws must be positive")
    if args.audit_sample_size_per_outcome < 1:
        raise ValueError("audit_sample_size_per_outcome must be positive")
    args.conditions = normalize_conditions(args.conditions)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / model_output_name(args.model_name)
    if args.strict_all_conditions and not set(DEFAULT_CONDITIONS).issubset(args.conditions):
        raise ValueError("strict_all_conditions requires all seven diagnostic conditions")


def prepared_path(args: argparse.Namespace) -> Path:
    return args.prepared_pairs_jsonl or args.output_dir / DEFAULT_PREPARED_NAME


def prepare_manifest_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "exp1_representation_prepare_manifest.json"


def load_prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = prepare_manifest_path(args)
    pairs_path = prepared_path(args)
    if not manifest_path.exists() or not pairs_path.exists():
        raise RuntimeError("Prepared pairs and their preparation manifest are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"Incomplete preparation manifest: {manifest_path}")
    observed_hash = file_hash(pairs_path)
    if observed_hash != manifest.get("prepared_pairs_sha256"):
        raise RuntimeError("Prepared-pair hash does not match the preparation manifest")
    missing = set(args.conditions) - set(manifest.get("conditions", []))
    if missing:
        raise RuntimeError(f"Requested conditions were not prepared: {', '.join(sorted(missing))}")
    if int(args.seed) != int(manifest.get("seed", args.seed)):
        raise RuntimeError("Scoring/analysis seed must match the preparation seed")
    if int(args.history_turns) != int(manifest.get("history_turns", args.history_turns)):
        raise RuntimeError("Scoring/analysis history length must match the prepared representations")
    return manifest


def score_path(output_dir: Path) -> Path:
    return output_dir / "exp1_representation_scores.jsonl"


def patch_dir(output_dir: Path, index: int, total: int) -> Path:
    if total == 1:
        return output_dir
    return output_dir / "patches" / f"patch_{index:04d}_of_{total:04d}"


def final_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "pairs": output_dir / "exp1_representation_pairs.csv",
        "scores": output_dir / "exp1_representation_scores.jsonl",
        "metrics_long": output_dir / "exp1_representation_metrics_long.csv",
        "metrics_wide": output_dir / "exp1_representation_metrics_wide.csv",
        "by_category": output_dir / "exp1_representation_by_category.csv",
        "by_move": output_dir / "exp1_representation_by_move.csv",
        "pairwise": output_dir / "exp1_representation_pairwise_deltas.csv",
        "decomposition": output_dir / "exp1_representation_decomposition.csv",
        "audit_sample": output_dir / "exp1_representation_audit_sample.csv",
        "diagnostic_gate": output_dir / "exp1_representation_diagnostic_gate.json",
        "coverage": output_dir / "exp1_representation_coverage.csv",
        "donors": output_dir / "exp1_representation_donors.jsonl",
        "summary": output_dir / "exp1_representation_summary.json",
        "diagnostic_pdf": output_dir / "exp1_representation_diagnostic_comparison.pdf",
        "diagnostic_png": output_dir / "exp1_representation_diagnostic_comparison.png",
        "decomposition_pdf": output_dir / "exp1_representation_decomposition_lifts.pdf",
        "decomposition_png": output_dir / "exp1_representation_decomposition_lifts.png",
    }


def normalize_categories(input_dir: Path, requested: list[str] | None) -> list[str]:
    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    available = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    if not requested or any(str(item).casefold() == "all" for item in requested):
        return available
    lookup = {name.casefold(): name for name in available}
    chosen: list[str] = []
    for raw in requested:
        match = lookup.get(str(raw).casefold())
        if match is None:
            raise ValueError(f"Unknown category {raw!r}; available: {', '.join(available)}")
        if match not in chosen:
            chosen.append(match)
    return chosen


def collect_category_files(
    input_dir: Path,
    categories: list[str],
    max_episodes_per_category: int | None,
) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for category in categories:
        paths = sorted((input_dir / category).glob("*.json"))
        if max_episodes_per_category is not None:
            paths = paths[:max_episodes_per_category]
        rows.extend((category, path) for path in paths)
    return rows


def load_turns(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        turns = value
    elif isinstance(value, dict) and isinstance(value.get("turns"), list):
        turns = value["turns"]
    else:
        raise ValueError(f"Unrecognized episode JSON format: {path}")
    return [turn for turn in turns if isinstance(turn, dict)]


def turn_time(turn: dict[str, Any]) -> float:
    value = turn.get("start_time", turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected a list of strings or text objects, got {type(value).__name__}")
    rows: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        raw = item.get("text") if isinstance(item, dict) else item
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise TypeError(f"Text item {index} must be a string or object with a string text field")
        text = raw.strip()
        if text and text not in seen:
            rows.append(text)
            seen.add(text)
    return rows


def extract_turn_text(turn: dict[str, Any]) -> str:
    for key in ("turn_text", "transcript", "text"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return " ".join(normalize_text_list(turn.get("explicit_propositions"))).strip()


def build_episode_records(
    category: str,
    path: Path,
    history_turns: int,
) -> tuple[list[TurnRecord], list[dict[str, Any]]]:
    raw_turns = load_turns(path)
    indexed = list(enumerate(raw_turns))
    indexed.sort(key=lambda item: (turn_time(item[1]), item[0]))
    ordered_turns = [turn for _, turn in indexed]
    turns: list[TurnRecord] = []
    substantive_by_list_position: dict[int, TurnRecord] = {}
    history: list[TurnRecord] = []
    substantive_position = 0
    for list_position, turn in enumerate(ordered_turns):
        if str(turn.get("turn_type_label") or "").strip() != "Substantive":
            continue
        text = extract_turn_text(turn)
        if not text:
            continue
        episode_id = str(turn.get("episode_id") or path.stem)
        raw_index = turn.get("turn_idx", list_position)
        try:
            turn_idx = int(raw_index)
        except (TypeError, ValueError):
            turn_idx = list_position
        substantive_position += 1
        previous = history[-history_turns:] if history_turns else []
        record: TurnRecord = {
            "turn_id": f"{category}:{episode_id}:{turn_idx}",
            "category": category,
            "episode_id": episode_id,
            "turn_idx": turn_idx,
            "list_position": list_position,
            "substantive_position": substantive_position,
            "timestamp": turn_time(turn),
            "move_label": str(turn.get("conversation_move_label") or "").strip(),
            "turn_text": text,
            "explicit_texts": normalize_text_list(turn.get("explicit_propositions")),
            "assumption_texts": normalize_text_list(turn.get("assumptions")),
            "history_turn_ids": [item["turn_id"] for item in previous],
            "history_turn_texts": [item["turn_text"] for item in previous],
        }
        turns.append(record)
        history.append(record)
        substantive_by_list_position[list_position] = record

    pairs: list[dict[str, Any]] = []
    for list_position in range(len(ordered_turns) - 1):
        source = substantive_by_list_position.get(list_position)
        target = substantive_by_list_position.get(list_position + 1)
        if source is None or target is None:
            continue
        pair_id = f"{category}:{target['episode_id']}:{source['turn_idx']}:{target['turn_idx']}"
        pairs.append(
            {
                "pair_id": pair_id,
                "category": category,
                "episode_id": target["episode_id"],
                "source_path": str(path),
                "source_turn_id": source["turn_id"],
                "source_turn_idx": source["turn_idx"],
                "source_substantive_position": source["substantive_position"],
                "source_turn_text": source["turn_text"],
                "source_explicit_texts": source["explicit_texts"],
                "source_assumption_texts": source["assumption_texts"],
                "history_turn_ids": source["history_turn_ids"],
                "history_turn_texts": source["history_turn_texts"],
                "history_turn_count": len(source["history_turn_ids"]),
                "true_next_turn_id": target["turn_id"],
                "true_next_turn_idx": target["turn_idx"],
                "true_next_turn_text": target["turn_text"],
                "true_next_turn_move_label": target["move_label"],
                "candidate_pool_complete": False,
                "coverage_drop_reason": None,
                "candidate_pool_sha256": None,
                "candidates": [],
                "conditions": {},
                "donors": {},
            }
        )
    return turns, pairs


def build_turn_indexes(turns: list[TurnRecord]) -> dict[str, Any]:
    lookup = {turn["turn_id"]: turn for turn in turns}
    by_category_headline: dict[str, list[str]] = defaultdict(list)
    by_episode: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_category_move: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    global_ids: list[str] = []
    for turn in sorted(turns, key=lambda row: (row["category"], row["episode_id"], row["turn_idx"])):
        turn_id = turn["turn_id"]
        by_episode[(turn["category"], turn["episode_id"])].append(turn_id)
        by_category[turn["category"]].append(turn_id)
        by_category_move[(turn["category"], turn["move_label"])].append(turn_id)
        global_ids.append(turn_id)
        if turn["move_label"] in HEADLINE_CONSTRUCTIVE_MOVES:
            by_category_headline[turn["category"]].append(turn_id)
    return {
        "lookup": lookup,
        "by_category_headline": by_category_headline,
        "by_episode": by_episode,
        "by_category_move": by_category_move,
        "by_category": by_category,
        "global": global_ids,
    }


def sample_unique_ids(
    ids: list[str],
    count: int,
    pair_id: str,
    label: str,
    seed: int,
) -> list[str]:
    unique = list(dict.fromkeys(ids))
    if count <= 0 or not unique:
        return []
    if len(unique) <= count:
        return unique
    rng = build_rng(f"{pair_id}:{label}", seed)
    indices = np.asarray(rng.choice(len(unique), size=count, replace=False)).tolist()
    return [unique[int(index)] for index in indices]


def build_candidates(pair: dict[str, Any], indexes: dict[str, Any], seed: int) -> dict[str, int]:
    lookup: dict[str, TurnRecord] = indexes["lookup"]
    target = lookup[pair["true_next_turn_id"]]
    history_ids = set(pair["history_turn_ids"])
    history_texts = set(pair["history_turn_texts"])
    excluded_ids = history_ids | {pair["source_turn_id"], pair["true_next_turn_id"]}
    if pair["true_next_turn_text"] in history_texts:
        pair["coverage_drop_reason"] = "true_candidate_text_in_history"
        return {}
    used: set[str] = set(excluded_ids)
    selected: list[tuple[str, str]] = []
    counts = {
        "negative_count_same_category_headline": 0,
        "negative_count_same_episode_gap3": 0,
        "negative_count_same_category_same_move": 0,
        "negative_count_same_category_backfill": 0,
        "negative_count_global_backfill": 0,
    }

    def eligible(ids: list[str], *, outside_episode: bool | None = None) -> list[str]:
        rows: list[str] = []
        for turn_id in ids:
            turn = lookup[turn_id]
            if turn_id in used or turn["turn_text"] in history_texts:
                continue
            if outside_episode is True and turn["episode_id"] == pair["episode_id"]:
                continue
            rows.append(turn_id)
        return rows

    layers = (
        (
            "same_category_headline",
            indexes["by_category_headline"].get(pair["category"], []),
            True,
            False,
        ),
        (
            "same_episode_gap3",
            indexes["by_episode"].get((pair["category"], pair["episode_id"]), []),
            None,
            True,
        ),
        (
            "same_category_same_move",
            indexes["by_category_move"].get((pair["category"], pair["true_next_turn_move_label"]), []),
            True,
            False,
        ),
    )
    for label, raw_pool, outside_episode, require_gap in layers:
        pool = eligible(raw_pool, outside_episode=outside_episode)
        if require_gap:
            pool = [
                turn_id
                for turn_id in pool
                if abs(int(lookup[turn_id]["substantive_position"]) - int(target["substantive_position"])) >= 3
            ]
        chosen = sample_unique_ids(pool, HARD_NEGATIVE_LAYER_TARGET, pair["pair_id"], label, seed)
        selected.extend((turn_id, label) for turn_id in chosen)
        used.update(chosen)
        counts[f"negative_count_{label}"] = len(chosen)

    if len(selected) < HARD_NEGATIVE_TARGET_COUNT:
        pool = eligible(indexes["by_category"].get(pair["category"], []), outside_episode=True)
        chosen = sample_unique_ids(
            pool,
            HARD_NEGATIVE_TARGET_COUNT - len(selected),
            pair["pair_id"],
            "same_category_backfill",
            seed,
        )
        selected.extend((turn_id, "same_category_backfill") for turn_id in chosen)
        used.update(chosen)
        counts["negative_count_same_category_backfill"] = len(chosen)
    if len(selected) < HARD_NEGATIVE_TARGET_COUNT:
        pool = eligible(indexes["global"], outside_episode=True)
        chosen = sample_unique_ids(
            pool,
            HARD_NEGATIVE_TARGET_COUNT - len(selected),
            pair["pair_id"],
            "global_backfill",
            seed,
        )
        selected.extend((turn_id, "global_backfill") for turn_id in chosen)
        used.update(chosen)
        counts["negative_count_global_backfill"] = len(chosen)
    if len(selected) != HARD_NEGATIVE_TARGET_COUNT:
        pair["coverage_drop_reason"] = "insufficient_unique_negatives"
        return counts

    raw_candidates: list[tuple[TurnRecord, bool, str]] = [(target, True, "true_next_turn")]
    raw_candidates.extend((lookup[turn_id], False, label) for turn_id, label in selected)
    order = np.asarray(build_rng(f"{pair['pair_id']}:candidate_order", seed).permutation(len(raw_candidates))).tolist()
    candidates: list[CandidateRecord] = []
    for candidate_order, raw_index in enumerate(order):
        turn, is_true, source = raw_candidates[int(raw_index)]
        digest = sha256_text(f"{pair['pair_id']}:{turn['turn_id']}")[:16]
        candidates.append(
            {
                "candidate_id": f"{pair['pair_id']}:candidate:{digest}",
                "candidate_order": candidate_order,
                "candidate_turn_id": turn["turn_id"],
                "candidate_category": turn["category"],
                "candidate_episode_id": turn["episode_id"],
                "candidate_turn_idx": turn["turn_idx"],
                "candidate_move_label": turn["move_label"],
                "candidate_text": turn["turn_text"],
                "is_true_next_turn": is_true,
                "negative_source": source,
            }
        )
    pair["candidates"] = candidates
    pair["candidate_pool_sha256"] = stable_hash([row["candidate_turn_id"] for row in candidates])
    pair["candidate_pool_complete"] = True
    pair["coverage_drop_reason"] = None
    return counts


def choose_donors(
    pair: dict[str, Any],
    turns: list[TurnRecord],
    indexes: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    candidate_ids = {row["candidate_turn_id"] for row in pair["candidates"]}
    excluded_ids = candidate_ids | {pair["source_turn_id"], pair["true_next_turn_id"]}
    source_assumptions = pair["source_assumption_texts"]
    source_position = int(pair["source_substantive_position"])

    def base_eligible(turn: TurnRecord) -> bool:
        return bool(
            turn["turn_id"] not in excluded_ids
            and turn["assumption_texts"]
            and turn["assumption_texts"] != source_assumptions
        )

    shuffled_same_category = sorted(
        (
            turn
            for turn in turns
            if base_eligible(turn)
            and turn["episode_id"] != pair["episode_id"]
            and turn["category"] == pair["category"]
        ),
        key=lambda row: row["turn_id"],
    )
    shuffled_any = sorted(
        (
            turn
            for turn in turns
            if base_eligible(turn) and turn["episode_id"] != pair["episode_id"]
        ),
        key=lambda row: row["turn_id"],
    )
    shuffled_pool = shuffled_same_category or shuffled_any
    shuffled_level = "same_category" if shuffled_same_category else "any_category"
    shuffled = None
    if shuffled_pool:
        rng = donor_rng(pair["pair_id"], "explicit_plus_shuffled_assumptions", seed)
        shuffled = shuffled_pool[int(rng.integers(0, len(shuffled_pool)))]

    same_episode = [
        turn
        for turn in turns
        if base_eligible(turn)
        and turn["category"] == pair["category"]
        and turn["episode_id"] == pair["episode_id"]
        and abs(int(turn["substantive_position"]) - source_position) >= 3
    ]
    wrong = None
    if same_episode:
        largest_distance = max(abs(int(turn["substantive_position"]) - source_position) for turn in same_episode)
        ties = sorted(
            (
                turn
                for turn in same_episode
                if abs(int(turn["substantive_position"]) - source_position) == largest_distance
            ),
            key=lambda row: row["turn_id"],
        )
        rng = donor_rng(pair["pair_id"], "explicit_plus_wrong_episode_assumptions", seed)
        wrong = ties[int(rng.integers(0, len(ties)))]

    rows: list[dict[str, Any]] = []
    for condition, donor, fallback, unavailable_reason in (
        (
            "explicit_plus_shuffled_assumptions",
            shuffled,
            shuffled_level if shuffled is not None else None,
            None if shuffled is not None else "no_different_episode_assumption_donor",
        ),
        (
            "explicit_plus_wrong_episode_assumptions",
            wrong,
            "same_episode_farthest" if wrong is not None else None,
            None if wrong is not None else "no_valid_same_episode_nonadjacent_donor",
        ),
    ):
        audit = {
            "pair_id": pair["pair_id"],
            "condition": condition,
            "source_turn_id": pair["source_turn_id"],
            "donor_turn_id": donor["turn_id"] if donor else None,
            "donor_episode_id": donor["episode_id"] if donor else None,
            "donor_category": donor["category"] if donor else None,
            "donor_fallback_level": fallback,
            "donor_assumption_count": len(donor["assumption_texts"]) if donor else 0,
            "donor_assumptions": donor["assumption_texts"] if donor else [],
            "control_unavailable_reason": unavailable_reason,
        }
        pair["donors"][condition] = audit
        rows.append(audit)
    return rows


def format_bullets(values: list[str], empty_value: str) -> str:
    return "\n".join(f"- {value}" for value in values) if values else empty_value


def format_representation(pair: dict[str, Any], condition: str) -> str:
    explicit = format_bullets(pair["source_explicit_texts"], EMPTY_EXPLICIT)
    assumptions = format_bullets(pair["source_assumption_texts"], EMPTY_ASSUMPTIONS)
    top1_assumption = format_bullets(pair["source_assumption_texts"][:1], EMPTY_ASSUMPTIONS)
    top3_assumptions = format_bullets(pair["source_assumption_texts"][:3], EMPTY_ASSUMPTIONS)
    if condition == "raw_turn":
        return f"[Raw current turn]\n{pair['source_turn_text']}"
    if condition == "raw_turn_with_history":
        if pair["history_turn_texts"]:
            history = "\n".join(
                f"Earlier turn {index}: {text}"
                for index, text in enumerate(pair["history_turn_texts"], start=1)
            )
        else:
            history = EMPTY_HISTORY
        return f"[Earlier substantive turns]\n{history}\n\n[Raw current turn]\n{pair['source_turn_text']}"
    if condition == "explicit_only":
        return f"[Explicit propositions]\n{explicit}"
    if condition == "assumptions_only":
        return f"[Implicit assumptions]\n{assumptions}"
    if condition == "explicit_plus_assumptions":
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions]\n{assumptions}"
    if condition == "explicit_plus_top1_assumption":
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions: first 1]\n{top1_assumption}"
    if condition == "explicit_plus_top3_assumptions":
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions: first 3]\n{top3_assumptions}"
    if condition == "raw_turn_plus_assumptions":
        return f"[Raw current turn]\n{pair['source_turn_text']}\n\n[Implicit assumptions]\n{assumptions}"
    if condition in CONTROL_CONDITIONS:
        donor = pair["donors"].get(condition, {})
        if donor.get("control_unavailable_reason"):
            raise ValueError(f"Condition {condition} is unavailable for {pair['pair_id']}")
        donor_text = format_bullets(list(donor["donor_assumptions"]), EMPTY_ASSUMPTIONS)
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions]\n{donor_text}"
    raise ValueError(f"Unknown condition: {condition}")


def build_conditions(pair: dict[str, Any], conditions: list[str]) -> None:
    for condition in conditions:
        donor = pair["donors"].get(condition)
        unavailable = donor.get("control_unavailable_reason") if donor else None
        if unavailable:
            pair["conditions"][condition] = {
                "available": False,
                "control_unavailable_reason": unavailable,
                "source_representation": None,
            }
        else:
            pair["conditions"][condition] = {
                "available": True,
                "control_unavailable_reason": None,
                "source_representation": format_representation(pair, condition),
            }


def validate_prepared_pair(pair: dict[str, Any], indexes: dict[str, Any], conditions: list[str]) -> None:
    if not pair["candidate_pool_complete"]:
        if pair["candidates"]:
            raise ValueError(f"Incomplete pair {pair['pair_id']} unexpectedly has candidates")
        return
    candidates = pair["candidates"]
    ids = [row["candidate_turn_id"] for row in candidates]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT or len(ids) != len(set(ids)):
        raise ValueError(f"Pair {pair['pair_id']} does not have 25 unique candidates")
    if sum(bool(row["is_true_next_turn"]) for row in candidates) != 1:
        raise ValueError(f"Pair {pair['pair_id']} does not have exactly one positive")
    if pair["true_next_turn_id"] not in ids:
        raise ValueError(f"Pair {pair['pair_id']} is missing its true next turn")
    if pair["candidate_pool_sha256"] != stable_hash(ids):
        raise ValueError(f"Candidate pool hash mismatch for {pair['pair_id']}")
    history_ids = set(pair["history_turn_ids"])
    history_texts = set(pair["history_turn_texts"])
    if history_ids.intersection(ids):
        raise ValueError(f"Candidate/history ID leakage for {pair['pair_id']}")
    if any(row["candidate_text"] in history_texts for row in candidates):
        raise ValueError(f"Candidate/history text leakage for {pair['pair_id']}")
    source = indexes["lookup"][pair["source_turn_id"]]
    for history_id in history_ids:
        history = indexes["lookup"][history_id]
        if history["episode_id"] != source["episode_id"] or history["substantive_position"] >= source["substantive_position"]:
            raise ValueError(f"Future or cross-episode history leakage for {pair['pair_id']}")
    for condition in CONTROL_CONDITIONS:
        donor = pair["donors"].get(condition)
        if not donor or donor.get("donor_turn_id") is None:
            continue
        if donor["donor_turn_id"] in set(ids) | {pair["source_turn_id"], pair["true_next_turn_id"]}:
            raise ValueError(f"Donor/candidate leakage for {pair['pair_id']} / {condition}")
    for condition in conditions:
        metadata = pair["conditions"].get(condition)
        if metadata is None:
            raise ValueError(f"Pair {pair['pair_id']} is missing condition {condition}")
        if metadata["available"] and not metadata["source_representation"]:
            raise ValueError(f"Pair {pair['pair_id']} has an empty representation for {condition}")


def pair_csv_row(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "category": pair["category"],
        "episode_id": pair["episode_id"],
        "source_path": pair["source_path"],
        "source_turn_id": pair["source_turn_id"],
        "source_turn_idx": pair["source_turn_idx"],
        "true_next_turn_id": pair["true_next_turn_id"],
        "true_next_turn_idx": pair["true_next_turn_idx"],
        "true_next_turn_move_label": pair["true_next_turn_move_label"],
        "explicit_count": len(pair["source_explicit_texts"]),
        "assumption_count": len(pair["source_assumption_texts"]),
        "history_turn_count": pair["history_turn_count"],
        "candidate_count": len(pair["candidates"]),
        "candidate_pool_complete": pair["candidate_pool_complete"],
        "candidate_pool_sha256": pair["candidate_pool_sha256"],
        "coverage_drop_reason": pair["coverage_drop_reason"],
        "available_conditions_json": json.dumps(
            [name for name, value in pair["conditions"].items() if value["available"]],
            ensure_ascii=False,
        ),
    }


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    categories = normalize_categories(args.input_dir, args.categories)
    category_files = collect_category_files(args.input_dir, categories, args.max_episodes_per_category)
    if not category_files:
        raise RuntimeError(f"No episode JSON files found under {args.input_dir}")
    logger.info(
        "Preparing representation baseline: version=%s prompt=%s input=%s categories=%s files=%d seed=%d conditions=%s",
        SCRIPT_VERSION,
        PROMPT_VERSION,
        args.input_dir,
        categories,
        len(category_files),
        args.seed,
        args.conditions,
    )
    turns: list[TurnRecord] = []
    pairs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for category, path in category_files:
        try:
            episode_turns, episode_pairs = build_episode_records(category, path, args.history_turns)
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
            continue
        turns.extend(episode_turns)
        pairs.extend(episode_pairs)
    if not turns:
        detail = errors[0]["error"] if errors else "no substantive turns"
        raise RuntimeError(f"No usable turns were loaded; first error: {detail}")
    indexes = build_turn_indexes(turns)
    aggregate_negative_counts: dict[str, int] = defaultdict(int)
    donors: list[dict[str, Any]] = []
    for pair in pairs:
        counts = build_candidates(pair, indexes, args.seed)
        for key, value in counts.items():
            aggregate_negative_counts[key] += value
        if pair["candidate_pool_complete"]:
            donors.extend(choose_donors(pair, turns, indexes, args.seed))
        else:
            for condition in CONTROL_CONDITIONS:
                audit = {
                    "pair_id": pair["pair_id"],
                    "condition": condition,
                    "source_turn_id": pair["source_turn_id"],
                    "donor_turn_id": None,
                    "donor_episode_id": None,
                    "donor_category": None,
                    "donor_fallback_level": None,
                    "donor_assumption_count": 0,
                    "donor_assumptions": [],
                    "control_unavailable_reason": "candidate_pool_incomplete",
                }
                pair["donors"][condition] = audit
                donors.append(audit)
        build_conditions(pair, args.conditions)
        validate_prepared_pair(pair, indexes, args.conditions)
    pairs.sort(key=lambda row: (row["category"], row["episode_id"], row["source_turn_idx"], row["true_next_turn_idx"]))
    donors.sort(key=lambda row: (row["pair_id"], row["condition"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepared_path(args)
    write_jsonl(prepared, pairs)
    write_jsonl(final_paths(args.output_dir)["donors"], donors)
    pd.DataFrame([pair_csv_row(pair) for pair in pairs]).to_csv(final_paths(args.output_dir)["pairs"], index=False)
    input_files = [
        {
            "category": category,
            "path": str(path),
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for category, path in category_files
    ]
    manifest = {
        "experiment": "Experiment 1: Explicit-Implicit Representation Baselines",
        "stage": "prepare",
        "complete": True,
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "git_commit": git_commit(),
        "runtime": runtime_versions(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "categories": categories,
        "selected_episode_files": input_files,
        "input_manifest_sha256": stable_hash(input_files),
        "prepared_pairs_jsonl": str(prepared),
        "prepared_pairs_sha256": file_hash(prepared),
        "conditions": args.conditions,
        "history_turns": args.history_turns,
        "seed": args.seed,
        "candidate_count_target": EXPECTED_CANDIDATE_COUNT,
        "negative_count_target": HARD_NEGATIVE_TARGET_COUNT,
        "episode_file_count": len(category_files),
        "source_episode_count": len({str(pair["source_path"]) for pair in pairs}),
        "turn_count": len(turns),
        "pair_count_before_candidate_filter": len(pairs),
        "candidate_complete_pair_count": sum(bool(pair["candidate_pool_complete"]) for pair in pairs),
        "candidate_incomplete_pair_count": sum(not pair["candidate_pool_complete"] for pair in pairs),
        "assumption_eligible_pair_count": sum(bool(pair["source_assumption_texts"]) for pair in pairs),
        "unavailable_controls": {
            condition: sum(
                pair["conditions"].get(condition, {}).get("available") is False for pair in pairs
            )
            for condition in CONTROL_CONDITIONS
        },
        "unavailable_control_reasons": {
            condition: {
                str(reason): int(count)
                for reason, count in pd.Series(
                    [
                        pair["conditions"].get(condition, {}).get("control_unavailable_reason")
                        for pair in pairs
                        if pair["conditions"].get(condition, {}).get("control_unavailable_reason")
                    ],
                    dtype="object",
                ).value_counts().items()
            }
            for condition in CONTROL_CONDITIONS
        },
        "normalization_error_count": len(errors),
        "normalization_errors": errors,
        "negative_sampling_counts": dict(aggregate_negative_counts),
    }
    write_json(prepare_manifest_path(args), manifest)
    logger.info("Prepared %d pairs (%d complete) at %s", len(pairs), manifest["candidate_complete_pair_count"], prepared)
    return manifest


def build_scoring_prompt(source_representation: str, candidate_text: str) -> str:
    return f"""You are a strict conversation-continuation judge.

Task:
Given a representation of the current dialogue state and one candidate next turn,
rate how likely the candidate is the true immediate next turn.

Use this 1-10 scale:
1 = impossible or totally unrelated
3 = weak fit
5 = plausible but uncertain
7 = strong local continuation
10 = almost certainly the immediate next turn

Source representation:
{source_representation}

Candidate next turn:
{candidate_text}

Return ONLY a raw JSON object with exactly these keys:
{{"score": <integer 1-10>, "rationale": "<brief reason>", "confidence": <number 0-1>}}
"""


def safe_json_extract(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
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
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def parse_llm_score(raw_output: str) -> ParsedScore:
    parsed = safe_json_extract(raw_output)
    if parsed is None:
        return {"score": None, "rationale": None, "confidence": None, "parse_success": False, "parse_error": "missing_json_object"}
    rationale_value = parsed.get("rationale")
    rationale = None if rationale_value is None else str(rationale_value).strip()
    confidence_value = parsed.get("confidence")
    confidence = None
    if isinstance(confidence_value, (int, float)) and not isinstance(confidence_value, bool):
        numeric = float(confidence_value)
        confidence = numeric if 0.0 <= numeric <= 1.0 else None
    raw_score = parsed.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return {"score": None, "rationale": rationale, "confidence": confidence, "parse_success": False, "parse_error": "score_not_numeric"}
    score = int(raw_score)
    if float(score) != float(raw_score) or not 1 <= score <= 10:
        return {"score": None, "rationale": rationale, "confidence": confidence, "parse_success": False, "parse_error": "score_out_of_range"}
    return {"score": score, "rationale": rationale, "confidence": confidence, "parse_success": True, "parse_error": None}


def task_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["pair_id"]),
        str(row["candidate_id"]),
        str(row["condition"]),
        str(row["model_name"]),
        str(row["prompt_version"]),
    )


def score_record_valid(row: dict[str, Any]) -> bool:
    return bool(row.get("parse_success") and isinstance(row.get("score"), int))


def compact_existing_scores(
    path: Path,
    *,
    model_name: str,
    prompt_version: str,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str, str]]]:
    if not path.exists():
        return [], set()
    observed = read_jsonl(path)
    kept: list[dict[str, Any]] = []
    canonical_by_key: dict[tuple[str, str, str, str, str], str] = {}
    completed: set[tuple[str, str, str, str, str]] = set()
    changed = False
    for row in observed:
        key = task_key(row)
        selected_config = key[3] == model_name and key[4] == prompt_version
        if selected_config and (overwrite or not score_record_valid(row)):
            changed = True
            continue
        canonical = canonical_json(row)
        previous = canonical_by_key.get(key)
        if previous is not None:
            if previous != canonical:
                raise RuntimeError(f"Conflicting duplicate score task key: {key}")
            changed = True
            continue
        canonical_by_key[key] = canonical
        kept.append(row)
        if score_record_valid(row):
            completed.add(key)
    if changed:
        write_jsonl(path, kept)
    return kept, completed


def select_patch_pairs(pairs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = sorted({str(pair["source_path"]) for pair in pairs})
    if args.episodes_per_patch is not None:
        start = args.patch_index * args.episodes_per_patch
        selected_paths = set(paths[start:start + args.episodes_per_patch])
    else:
        selected_paths = {path for index, path in enumerate(paths) if index % args.num_patches == args.patch_index}
    return [pair for pair in pairs if str(pair["source_path"]) in selected_paths]


def build_tasks(pairs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pair in pairs:
        if not pair["candidate_pool_complete"]:
            continue
        if args.strict_all_conditions and not all(
            pair["conditions"].get(condition, {}).get("available") for condition in args.conditions
        ):
            continue
        for condition in args.conditions:
            metadata = pair["conditions"].get(condition)
            if not metadata or not metadata["available"]:
                continue
            for candidate in pair["candidates"]:
                tasks.append(
                    {
                        "pair": pair,
                        "candidate": candidate,
                        "condition": condition,
                        "source_representation": metadata["source_representation"],
                    }
                )
    tasks.sort(
        key=lambda task: (
            task["pair"]["pair_id"],
            task["candidate"]["candidate_order"],
            task["condition"],
        )
    )
    if tasks:
        order = np.asarray(build_rng("representation_task_order", args.seed).permutation(len(tasks))).tolist()
        tasks = [tasks[int(index)] for index in order]
    return tasks


class LLMInterface:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError("Scoring requires the optional vllm dependency; use --dry_run for CPU validation") from error
        self.llm = LLM(
            model=args.model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            download_dir=str(args.download_dir),
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend="mp",
            trust_remote_code=True,
        )
        self.params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.params)
        return [output.outputs[0].text.strip() for output in outputs]


def score_row(task: dict[str, Any], raw_output: str, parsed: ParsedScore, args: argparse.Namespace) -> dict[str, Any]:
    pair = task["pair"]
    candidate = task["candidate"]
    donor = pair["donors"].get(task["condition"], {})
    return {
        "pair_id": pair["pair_id"],
        "candidate_id": candidate["candidate_id"],
        "condition": task["condition"],
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "source_turn_id": pair["source_turn_id"],
        "candidate_turn_id": candidate["candidate_turn_id"],
        "candidate_order": candidate["candidate_order"],
        "candidate_text": candidate["candidate_text"],
        "is_true_next_turn": candidate["is_true_next_turn"],
        "negative_source": candidate["negative_source"],
        "source_representation": task["source_representation"] if args.save_source_representation else None,
        "source_explicit_json": json.dumps(pair["source_explicit_texts"], ensure_ascii=False),
        "source_assumptions_json": json.dumps(pair["source_assumption_texts"], ensure_ascii=False),
        "donor_turn_id": donor.get("donor_turn_id"),
        "donor_episode_id": donor.get("donor_episode_id"),
        "donor_category": donor.get("donor_category"),
        "donor_fallback_level": donor.get("donor_fallback_level"),
        "candidate_pool_sha256": pair["candidate_pool_sha256"],
        "score": parsed["score"],
        "rationale": parsed["rationale"],
        "confidence": parsed["confidence"],
        "parse_success": parsed["parse_success"],
        "parse_error": parsed["parse_error"],
        "raw_output": raw_output,
    }


def score_dataset(args: argparse.Namespace) -> dict[str, Any]:
    path = prepared_path(args)
    prepare_manifest = load_prepare_manifest(args)
    pairs = read_jsonl(path)
    selected_pairs = select_patch_pairs(pairs, args)
    if not selected_pairs:
        raise RuntimeError(f"Patch {args.patch_index} selected no source episodes")
    logger.info(
        "Starting score stage: model=%s prompt=%s seed=%d conditions=%s prepared_hash=%s",
        args.model_name,
        PROMPT_VERSION,
        args.seed,
        args.conditions,
        prepare_manifest["prepared_pairs_sha256"],
    )
    output = patch_dir(args.output_dir, args.patch_index, args.num_patches)
    output.mkdir(parents=True, exist_ok=True)
    scores_file = score_path(output)
    existing, completed = compact_existing_scores(
        scores_file,
        model_name=args.model_name,
        prompt_version=PROMPT_VERSION,
        overwrite=args.overwrite_scores,
    )
    tasks = build_tasks(selected_pairs, args)
    pending = []
    for task in tasks:
        candidate = task["candidate"]
        key = (
            task["pair"]["pair_id"],
            candidate["candidate_id"],
            task["condition"],
            args.model_name,
            PROMPT_VERSION,
        )
        if key not in completed:
            pending.append(task)
    logger.info(
        "Scoring patch %d/%d: pairs=%d tasks=%d completed=%d pending=%d dry_run=%s",
        args.patch_index,
        args.num_patches,
        len(selected_pairs),
        len(tasks),
        len(tasks) - len(pending),
        len(pending),
        args.dry_run,
    )
    llm = None if args.dry_run or not pending else LLMInterface(args)
    attempted = 0
    parse_failures = 0
    for start in range(0, len(pending), args.prompt_batch_size):
        batch = pending[start:start + args.prompt_batch_size]
        if args.dry_run:
            raw_outputs = []
            for task in batch:
                candidate = task["candidate"]
                if candidate["is_true_next_turn"]:
                    score = 10
                else:
                    score = 1 + seed_int(task["pair"]["pair_id"] + candidate["candidate_id"] + task["condition"]) % 5
                raw_outputs.append(json.dumps({"score": score, "rationale": "dry_run", "confidence": 1.0}))
        else:
            assert llm is not None
            prompts = [build_scoring_prompt(task["source_representation"], task["candidate"]["candidate_text"]) for task in batch]
            raw_outputs = llm.generate_batch(prompts)
        if len(raw_outputs) != len(batch):
            raise RuntimeError("Model returned a different number of outputs than prompts")
        rows: list[dict[str, Any]] = []
        for task, raw_output in zip(batch, raw_outputs):
            parsed = parse_llm_score(raw_output)
            parse_failures += int(not parsed["parse_success"])
            rows.append(score_row(task, raw_output, parsed, args))
        append_jsonl(scores_file, rows)
        attempted += len(rows)
    if not scores_file.exists():
        write_jsonl(scores_file, [])
    all_rows = read_jsonl(scores_file) if scores_file.exists() else existing
    selected_keys = {
        (
            task["pair"]["pair_id"],
            task["candidate"]["candidate_id"],
            task["condition"],
            args.model_name,
            PROMPT_VERSION,
        )
        for task in tasks
    }
    valid_count = sum(score_record_valid(row) and task_key(row) in selected_keys for row in all_rows)
    config = {
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "conditions": args.conditions,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "strict_all_conditions": args.strict_all_conditions,
        "dry_run": args.dry_run,
    }
    manifest = {
        "stage": "score_patch",
        "complete": True,
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "patch_index": args.patch_index,
        "num_patches": args.num_patches,
        "episodes_per_patch": args.episodes_per_patch,
        "prepared_pairs_sha256": prepare_manifest["prepared_pairs_sha256"],
        "config": config,
        "config_sha256": stable_hash(config),
        "selected_source_paths": sorted({str(pair["source_path"]) for pair in selected_pairs}),
        "selected_pair_count": len(selected_pairs),
        "expected_task_count": len(tasks),
        "valid_task_count": valid_count,
        "attempted_this_run": attempted,
        "parse_failures_this_run": parse_failures,
        "scores_sha256": file_hash(scores_file) if scores_file.exists() else None,
    }
    write_json(output / "patch_manifest.json", manifest)
    return manifest


def rank_condition_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sortable = [row for row in rows if score_record_valid(row)]
    sortable.sort(key=lambda row: (-int(row["score"]), int(row["candidate_order"])))
    return [dict(row, rank=index) for index, row in enumerate(sortable, start=1)]


def cluster_bootstrap(
    rows: pd.DataFrame,
    value_column: str,
    *,
    seed_label: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    valid = rows[rows[value_column].notna()].copy()
    if valid.empty:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "ci_unstable": True, "cluster_count": 0}
    valid["_cluster"] = valid["category"].astype(str) + "||" + valid["episode_id"].astype(str)
    clusters = [group[value_column].to_numpy(dtype=np.float64) for _, group in valid.groupby("_cluster", sort=False)]
    cluster_count = len(clusters)
    point = float(valid[value_column].astype(float).mean())
    if cluster_count < DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS:
        logger.warning("Only %d clusters for %s; omitting confidence interval", cluster_count, seed_label)
        return {"mean": point, "ci95_low": None, "ci95_high": None, "ci_unstable": True, "cluster_count": cluster_count}
    rng = build_rng(seed_label, seed)
    samples = []
    for _ in range(draws):
        indices = rng.integers(0, cluster_count, size=cluster_count)
        values = np.concatenate([clusters[int(index)] for index in indices])
        samples.append(float(values.mean()))
    return {
        "mean": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "ci_unstable": False,
        "cluster_count": cluster_count,
    }


def build_metrics(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_scores = [
        row
        for row in scores
        if row.get("model_name") == args.model_name and row.get("prompt_version") == PROMPT_VERSION
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_scores:
        grouped[(str(row["pair_id"]), str(row["condition"]))].append(row)
    long_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for condition in args.conditions:
            metadata = pair["conditions"].get(condition, {"available": False, "control_unavailable_reason": "not_prepared"})
            condition_rows = grouped.get((pair["pair_id"], condition), [])
            parsed = [row for row in condition_rows if score_record_valid(row)]
            metric: dict[str, Any] = {
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "episode_id": pair["episode_id"],
                "source_turn_idx": pair["source_turn_idx"],
                "true_next_turn_idx": pair["true_next_turn_idx"],
                "true_next_turn_move_label": pair["true_next_turn_move_label"],
                "condition": condition,
                "assumption_count": len(pair["source_assumption_texts"]),
                "analysis_subset_flags": "[]",
                "true_rank": None,
                "true_score": None,
                "top1": None,
                "reciprocal_rank": None,
                "candidate_count": len(pair["candidates"]),
                "parsed_score_count": len(parsed),
                "condition_available": bool(metadata.get("available")),
                "control_unavailable_reason": metadata.get("control_unavailable_reason"),
                "candidate_pool_complete": bool(pair["candidate_pool_complete"]),
                "full_retained": False,
                "assumption_eligible": False,
                "complete_case": False,
            }
            if pair["candidate_pool_complete"] and metric["condition_available"] and len(parsed) == EXPECTED_CANDIDATE_COUNT:
                ranked = rank_condition_scores(parsed)
                positives = [row for row in ranked if row["is_true_next_turn"]]
                if len(positives) == 1:
                    positive = positives[0]
                    metric["true_rank"] = int(positive["rank"])
                    metric["true_score"] = int(positive["score"])
                    metric["top1"] = int(positive["rank"] == 1)
                    metric["reciprocal_rank"] = 1.0 / float(positive["rank"])
                    metric["full_retained"] = True
                    metric["assumption_eligible"] = bool(pair["source_assumption_texts"])
            long_rows.append(metric)
    long_df = pd.DataFrame(long_rows)
    complete_by_pair: dict[str, bool] = {}
    selected_rows = long_df[long_df["condition"].isin(args.conditions)]
    for pair_id, group in selected_rows.groupby("pair_id", sort=False):
        complete_by_pair[str(pair_id)] = bool(
            len(group) == len(args.conditions)
            and set(group["condition"]) == set(args.conditions)
            and group["full_retained"].all()
        )
    long_df["complete_case"] = long_df["pair_id"].map(complete_by_pair).fillna(False).astype(bool)
    for index, row in long_df.iterrows():
        flags = []
        if bool(row["full_retained"]):
            flags.append("full")
        if bool(row["assumption_eligible"]):
            flags.append("assumption_eligible")
        if bool(row["complete_case"]):
            flags.append("complete_case")
        long_df.at[index, "analysis_subset_flags"] = json.dumps(flags)
    metadata_columns = [
        "pair_id", "category", "episode_id", "source_turn_idx", "true_next_turn_idx",
        "true_next_turn_move_label", "assumption_count",
    ]
    wide = long_df[metadata_columns].drop_duplicates("pair_id").set_index("pair_id")
    for condition in args.conditions:
        part = long_df[long_df["condition"] == condition].set_index("pair_id")
        for column in (
            "true_rank", "true_score", "top1", "reciprocal_rank", "parsed_score_count",
            "condition_available", "control_unavailable_reason", "full_retained",
            "assumption_eligible", "complete_case",
        ):
            wide[f"{condition}__{column}"] = part[column]
    return long_df, wide.reset_index()


def condition_summary(
    long_df: pd.DataFrame,
    group_column: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "full": "full_retained",
        "assumption_eligible": "assumption_eligible",
        "complete_case": "complete_case",
    }
    metrics = ("true_rank", "top1", "reciprocal_rank", "true_score")
    for subset, flag in subsets.items():
        eligible = long_df[long_df[flag] == True].copy()
        if subset == "complete_case":
            eligible = eligible[eligible["full_retained"] == True]
        if eligible.empty:
            continue
        for (condition, group_value), group in eligible.groupby(["condition", group_column], sort=False, dropna=False):
            row: dict[str, Any] = {
                "analysis_subset": subset,
                "condition": condition,
                group_column: group_value,
                "pair_count": len(group),
            }
            for metric in metrics:
                result = cluster_bootstrap(
                    group,
                    metric,
                    seed_label=f"{subset}:{condition}:{group_column}:{group_value}:{metric}",
                    seed=args.seed,
                    draws=args.bootstrap_draws,
                )
                prefix = {"true_rank": "mean_rank", "top1": "top1_rate", "reciprocal_rank": "mrr", "true_score": "mean_true_score"}[metric]
                row[prefix] = result["mean"]
                row[f"{prefix}_ci95_low"] = result["ci95_low"]
                row[f"{prefix}_ci95_high"] = result["ci95_high"]
                row[f"{prefix}_ci_unstable"] = result["ci_unstable"]
                row["cluster_count"] = result["cluster_count"]
            rows.append(row)
    return pd.DataFrame(rows)


def overall_condition_summary(long_df: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subsets = {
        "full": "full_retained",
        "assumption_eligible": "assumption_eligible",
        "complete_case": "complete_case",
    }
    metric_names = {
        "true_rank": "mean_rank",
        "top1": "top1_rate",
        "reciprocal_rank": "mrr",
        "true_score": "mean_true_score",
    }
    for subset, flag in subsets.items():
        for condition in args.conditions:
            group = long_df[(long_df["condition"] == condition) & (long_df[flag] == True)].copy()
            if subset == "complete_case":
                group = group[group["full_retained"] == True]
            row: dict[str, Any] = {
                "analysis_subset": subset,
                "condition": condition,
                "pair_count": int(len(group)),
            }
            for metric, output_name in metric_names.items():
                result = cluster_bootstrap(
                    group,
                    metric,
                    seed_label=f"overall:{subset}:{condition}:{metric}",
                    seed=args.seed,
                    draws=args.bootstrap_draws,
                )
                row[output_name] = result["mean"]
                row[f"{output_name}_ci95_low"] = result["ci95_low"]
                row[f"{output_name}_ci95_high"] = result["ci95_high"]
                row[f"{output_name}_ci_unstable"] = result["ci_unstable"]
                row["cluster_count"] = result["cluster_count"]
            rows.append(row)
    return rows


def contrast_list(conditions: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for contrast in REQUIRED_CONTRASTS:
        if contrast[0] in conditions and contrast[1] in conditions and contrast not in rows:
            rows.append(contrast)
    for target in conditions:
        for baseline in ("raw_turn", "explicit_only"):
            contrast = (target, baseline)
            if target != baseline and baseline in conditions and contrast not in rows:
                rows.append(contrast)
    return rows


def build_pairwise(long_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = ("true_rank", "top1", "reciprocal_rank", "true_score")
    for target, baseline in contrast_list(args.conditions):
        target_df = long_df[long_df["condition"] == target].set_index("pair_id")
        baseline_df = long_df[long_df["condition"] == baseline].set_index("pair_id")
        common_ids = target_df.index.intersection(baseline_df.index)
        for subset in ("full", "assumption_eligible", "complete_case"):
            subset_ids = []
            for pair_id in common_ids:
                target_row = target_df.loc[pair_id]
                baseline_row = baseline_df.loc[pair_id]
                if subset == "full":
                    keep = bool(target_row["full_retained"] and baseline_row["full_retained"])
                elif subset == "assumption_eligible":
                    keep = bool(target_row["assumption_eligible"] and baseline_row["assumption_eligible"])
                else:
                    keep = bool(
                        target_row["complete_case"]
                        and baseline_row["complete_case"]
                        and target_row["full_retained"]
                        and baseline_row["full_retained"]
                    )
                if keep:
                    subset_ids.append(pair_id)
            for metric in metric_columns:
                delta_rows = []
                for pair_id in subset_ids:
                    target_row = target_df.loc[pair_id]
                    baseline_row = baseline_df.loc[pair_id]
                    target_value = float(target_row[metric])
                    baseline_value = float(baseline_row[metric])
                    delta = baseline_value - target_value if metric == "true_rank" else target_value - baseline_value
                    delta_rows.append(
                        {
                            "pair_id": pair_id,
                            "category": target_row["category"],
                            "episode_id": target_row["episode_id"],
                            "delta": delta,
                        }
                    )
                delta_df = pd.DataFrame(delta_rows, columns=["pair_id", "category", "episode_id", "delta"])
                result = cluster_bootstrap(
                    delta_df,
                    "delta",
                    seed_label=f"paired:{subset}:{target}:{baseline}:{metric}",
                    seed=args.seed,
                    draws=args.bootstrap_draws,
                )
                values = delta_df["delta"] if not delta_df.empty else pd.Series(dtype=float)
                rows.append(
                    {
                        "analysis_subset": subset,
                        "target_condition": target,
                        "baseline_condition": baseline,
                        "metric": metric,
                        "paired_sample_size": len(delta_df),
                        "mean_improvement": result["mean"],
                        "ci95_low": result["ci95_low"],
                        "ci95_high": result["ci95_high"],
                        "ci_unstable": result["ci_unstable"],
                        "cluster_count": result["cluster_count"],
                        "wins": int((values > 0).sum()),
                        "ties": int((values == 0).sum()),
                        "losses": int((values < 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def coverage_summary(long_df: pd.DataFrame, pairs: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    total = len(pairs)
    for condition in args.conditions:
        group = long_df[long_df["condition"] == condition]
        parse_eligible = (group["condition_available"] == True) & (group["candidate_pool_complete"] == True)
        parse_failures = parse_eligible & (group["full_retained"] == False)
        parse_eligible_count = int(parse_eligible.sum())
        rows.append(
            {
                "condition": condition,
                "pair_count": total,
                "candidate_complete_pair_count": int(group["candidate_pool_complete"].sum()),
                "condition_available_pair_count": int(group["condition_available"].sum()),
                "fully_parsed_pair_count": int(group["full_retained"].sum()),
                "assumption_eligible_pair_count": int(group["assumption_eligible"].sum()),
                "complete_case_pair_count": int(group["complete_case"].sum()),
                "retained_pair_rate": float(group["full_retained"].mean()) if total else None,
                "score_parse_eligible_pair_count": parse_eligible_count,
                "score_parse_failure_pair_count": int(parse_failures.sum()),
                "score_parse_failure_rate": (
                    float(parse_failures.sum() / parse_eligible_count) if parse_eligible_count else None
                ),
            }
        )
    return pd.DataFrame(rows)


def build_decomposition_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    questions = {
        ("explicit_plus_assumptions", "explicit_only"): "incremental_implicit_value_after_abstraction",
        ("raw_turn_plus_assumptions", "raw_turn"): "incremental_implicit_value_with_lexical_context",
        ("raw_turn", "explicit_only"): "information_retained_by_raw_turn",
        ("raw_turn_with_history", "raw_turn"): "value_of_discourse_history",
        ("explicit_plus_top1_assumption", "explicit_only"): "first_assumption_budget",
        ("explicit_plus_top3_assumptions", "explicit_only"): "first_three_assumption_budget",
        ("explicit_plus_assumptions", "explicit_plus_top1_assumption"): "all_assumptions_vs_first_one",
        ("explicit_plus_assumptions", "explicit_plus_top3_assumptions"): "all_assumptions_vs_first_three",
    }
    if pairwise.empty:
        return pd.DataFrame(columns=[*pairwise.columns, "diagnostic_question", "contrast"])
    selected = pairwise[
        pairwise.apply(
            lambda row: (str(row["target_condition"]), str(row["baseline_condition"])) in questions,
            axis=1,
        )
    ].copy()
    selected["diagnostic_question"] = selected.apply(
        lambda row: questions[(str(row["target_condition"]), str(row["baseline_condition"]))],
        axis=1,
    )
    selected["contrast"] = selected["target_condition"].astype(str) + " - " + selected["baseline_condition"].astype(str)
    order = {contrast: index for index, contrast in enumerate(questions)}
    selected["_contrast_order"] = selected.apply(
        lambda row: order[(str(row["target_condition"]), str(row["baseline_condition"]))],
        axis=1,
    )
    subset_order = {"assumption_eligible": 0, "complete_case": 1, "full": 2}
    metric_order = {"reciprocal_rank": 0, "top1": 1, "true_rank": 2, "true_score": 3}
    selected["_subset_order"] = selected["analysis_subset"].map(subset_order).fillna(99)
    selected["_metric_order"] = selected["metric"].map(metric_order).fillna(99)
    return selected.sort_values(
        ["_contrast_order", "_subset_order", "_metric_order"], kind="stable"
    ).drop(columns=["_contrast_order", "_subset_order", "_metric_order"])


def build_audit_sample(
    pairs: list[dict[str, Any]],
    long_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    columns = [
        "audit_outcome", "audit_priority", "pair_id", "category", "episode_id",
        "true_next_turn_move_label", "assumption_count", "mrr_delta", "rank_improvement",
        "top1_delta", "true_score_delta", "explicit_rank", "combined_rank", "explicit_score",
        "combined_score", "source_turn_text", "source_explicit_json", "source_assumptions_json",
        "history_turns_json", "true_next_turn_text",
    ]
    required = {"explicit_only", "explicit_plus_assumptions"}
    if not required.issubset(set(args.conditions)):
        return pd.DataFrame(columns=columns)
    lookup = {pair["pair_id"]: pair for pair in pairs}
    explicit = long_df[long_df["condition"] == "explicit_only"].set_index("pair_id")
    combined = long_df[long_df["condition"] == "explicit_plus_assumptions"].set_index("pair_id")
    rows: list[dict[str, Any]] = []
    for pair_id in explicit.index.intersection(combined.index):
        explicit_row = explicit.loc[pair_id]
        combined_row = combined.loc[pair_id]
        if not bool(explicit_row["assumption_eligible"] and combined_row["assumption_eligible"]):
            continue
        pair = lookup[str(pair_id)]
        mrr_delta = float(combined_row["reciprocal_rank"]) - float(explicit_row["reciprocal_rank"])
        outcome = "win" if mrr_delta > 0 else "loss" if mrr_delta < 0 else "tie"
        rows.append(
            {
                "audit_outcome": outcome,
                "pair_id": pair_id,
                "category": pair["category"],
                "episode_id": pair["episode_id"],
                "true_next_turn_move_label": pair["true_next_turn_move_label"],
                "assumption_count": len(pair["source_assumption_texts"]),
                "mrr_delta": mrr_delta,
                "rank_improvement": float(explicit_row["true_rank"]) - float(combined_row["true_rank"]),
                "top1_delta": float(combined_row["top1"]) - float(explicit_row["top1"]),
                "true_score_delta": float(combined_row["true_score"]) - float(explicit_row["true_score"]),
                "explicit_rank": int(explicit_row["true_rank"]),
                "combined_rank": int(combined_row["true_rank"]),
                "explicit_score": int(explicit_row["true_score"]),
                "combined_score": int(combined_row["true_score"]),
                "source_turn_text": pair["source_turn_text"],
                "source_explicit_json": json.dumps(pair["source_explicit_texts"], ensure_ascii=False),
                "source_assumptions_json": json.dumps(pair["source_assumption_texts"], ensure_ascii=False),
                "history_turns_json": json.dumps(pair["history_turn_texts"], ensure_ascii=False),
                "true_next_turn_text": pair["true_next_turn_text"],
                "_tie_order": seed_int(f"audit:{pair_id}:{args.seed}"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    audit = pd.DataFrame(rows)
    samples = []
    for outcome in ("win", "loss", "tie"):
        group = audit[audit["audit_outcome"] == outcome].copy()
        if outcome == "win":
            group = group.sort_values(["mrr_delta", "rank_improvement", "pair_id"], ascending=[False, False, True])
        elif outcome == "loss":
            group = group.sort_values(["mrr_delta", "rank_improvement", "pair_id"], ascending=[True, True, True])
        else:
            group["_absolute_score_delta"] = group["true_score_delta"].abs()
            group = group.sort_values(["_absolute_score_delta", "_tie_order"], ascending=[False, True])
        group = group.head(args.audit_sample_size_per_outcome).copy()
        group["audit_priority"] = np.arange(1, len(group) + 1)
        samples.append(group)
    result = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=columns)
    return result.reindex(columns=columns)


def diagnostic_gate(
    pairwise: pd.DataFrame,
    long_df: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    def contrast_row(target: str, baseline: str) -> dict[str, Any] | None:
        match = pairwise[
            (pairwise["analysis_subset"] == "assumption_eligible")
            & (pairwise["target_condition"] == target)
            & (pairwise["baseline_condition"] == baseline)
            & (pairwise["metric"] == "reciprocal_rank")
        ]
        return None if match.empty else match.iloc[0].to_dict()

    def category_deltas(target: str, baseline: str) -> dict[str, float]:
        target_rows = long_df[long_df["condition"] == target].set_index("pair_id")
        baseline_rows = long_df[long_df["condition"] == baseline].set_index("pair_id")
        values: list[dict[str, Any]] = []
        for pair_id in target_rows.index.intersection(baseline_rows.index):
            target_row = target_rows.loc[pair_id]
            baseline_row = baseline_rows.loc[pair_id]
            if not bool(target_row["assumption_eligible"] and baseline_row["assumption_eligible"]):
                continue
            values.append(
                {
                    "category": str(target_row["category"]),
                    "delta": float(target_row["reciprocal_rank"]) - float(baseline_row["reciprocal_rank"]),
                }
            )
        if not values:
            return {}
        frame = pd.DataFrame(values)
        return {str(key): float(value) for key, value in frame.groupby("category")["delta"].mean().items()}

    primary = contrast_row("explicit_plus_assumptions", "explicit_only")
    raw_increment = contrast_row("raw_turn_plus_assumptions", "raw_turn")
    primary_categories = category_deltas("explicit_plus_assumptions", "explicit_only")
    raw_categories = category_deltas("raw_turn_plus_assumptions", "raw_turn")

    def supported(row: dict[str, Any] | None) -> bool:
        return bool(row and row.get("ci95_low") is not None and float(row["ci95_low"]) > 0.0)

    minimum_retained = float(coverage["retained_pair_rate"].min()) if not coverage.empty else 0.0
    primary_supported = supported(primary)
    raw_supported = supported(raw_increment)
    positive_primary_categories = sum(value > 0 for value in primary_categories.values())
    positive_raw_categories = sum(value > 0 for value in raw_categories.values())
    category_breadth = max(positive_primary_categories, positive_raw_categories) >= 2
    coverage_acceptable = minimum_retained >= 0.98
    if raw_supported:
        interpretation = "assumptions_add_signal_beyond_raw_lexical_context"
    elif primary_supported:
        interpretation = "assumptions_help_after_abstraction_but_not_beyond_raw_context"
    else:
        interpretation = "no_robust_incremental_assumption_signal"
    ready_for_cross_model = bool((primary_supported or raw_supported) and category_breadth and coverage_acceptable)
    return {
        "gate_version": "diagnostic-gate-v1",
        "primary_contrast": primary,
        "raw_context_contrast": raw_increment,
        "primary_category_mrr_deltas": primary_categories,
        "raw_context_category_mrr_deltas": raw_categories,
        "criteria": {
            "primary_mrr_ci_excludes_zero": primary_supported,
            "raw_context_mrr_ci_excludes_zero": raw_supported,
            "positive_category_count_primary": positive_primary_categories,
            "positive_category_count_raw_context": positive_raw_categories,
            "category_breadth_at_least_two": category_breadth,
            "minimum_condition_retained_rate": minimum_retained,
            "coverage_at_least_98_percent": coverage_acceptable,
        },
        "interpretation": interpretation,
        "ready_for_cross_model_smoke": ready_for_cross_model,
        "ready_for_full_corpus": False,
        "full_corpus_blocker": "A second judge-model smoke run and manual audit are required before the full-corpus gate can pass.",
        "recommended_next_stage": "cross_model_smoke" if ready_for_cross_model else "manual_audit_or_representation_revision",
    }


def plot_results(long_df: pd.DataFrame, pairwise: pd.DataFrame, paths: dict[str, Path], args: argparse.Namespace) -> None:
    try:
        import matplotlib
    except ImportError:
        logger.warning(
            "matplotlib is unavailable; writing conspicuous placeholder plot files. "
            "Install the repository's base dependencies before producing paper artifacts."
        )
        transparent_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        placeholder_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 93>>stream\nBT /F1 14 Tf 72 720 Td (matplotlib unavailable - regenerate paper plot after installing dependencies.) Tj ET\nendstream endobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000238 00000 n \n0000000380 00000 n \n"
            b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n450\n%%EOF\n"
        )
        for key in ("diagnostic_png", "decomposition_png"):
            atomic_write_bytes(paths[key], transparent_png)
        for key in ("diagnostic_pdf", "decomposition_pdf"):
            atomic_write_bytes(paths[key], placeholder_pdf)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "raw_turn": "Raw",
        "raw_turn_with_history": "Raw + history",
        "raw_turn_plus_assumptions": "Raw + assumptions",
        "explicit_only": "Explicit",
        "explicit_plus_top1_assumption": "Explicit + first 1",
        "explicit_plus_top3_assumptions": "Explicit + first 3",
        "assumptions_only": "Assumptions",
        "explicit_plus_assumptions": "Explicit + all",
        "explicit_plus_shuffled_assumptions": "Explicit + shuffled",
        "explicit_plus_wrong_episode_assumptions": "Explicit + same-episode wrong",
    }
    complete = long_df[(long_df["complete_case"] == True) & (long_df["full_retained"] == True)]
    condition_rows = []
    for condition in args.conditions:
        group = complete[complete["condition"] == condition]
        for metric in ("reciprocal_rank", "top1"):
            result = cluster_bootstrap(
                group,
                metric,
                seed_label=f"plot:complete:{condition}:{metric}",
                seed=args.seed,
                draws=args.bootstrap_draws,
            )
            condition_rows.append({"condition": condition, "metric": metric, **result})
    plot_df = pd.DataFrame(condition_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    if complete.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No complete-case data available", ha="center", va="center")
            axis.set_axis_off()
    else:
        for axis, metric, title in zip(axes, ("reciprocal_rank", "top1"), ("MRR", "Top-1 accuracy")):
            metric_df = plot_df[plot_df["metric"] == metric].set_index("condition").reindex(args.conditions)
            means = metric_df["mean"].astype(float).to_numpy()
            x = np.arange(len(args.conditions))
            axis.bar(x, means, color="#4C78A8")
            lows = metric_df["ci95_low"].to_numpy(dtype=float)
            highs = metric_df["ci95_high"].to_numpy(dtype=float)
            if np.isfinite(lows).all() and np.isfinite(highs).all():
                axis.errorbar(x, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", color="black", capsize=3)
            axis.set_xticks(x, [labels[value] for value in args.conditions], rotation=35, ha="right")
            axis.set_title(title)
            axis.set_ylabel(title)
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Diagnostic representations — complete cases (n={complete['pair_id'].nunique()})")
    fig.savefig(paths["diagnostic_pdf"], bbox_inches="tight")
    fig.savefig(paths["diagnostic_png"], dpi=args.plot_dpi, bbox_inches="tight")
    plt.close(fig)

    decomposition_contrasts = (
        ("explicit_plus_assumptions", "explicit_only"),
        ("raw_turn_plus_assumptions", "raw_turn"),
        ("raw_turn", "explicit_only"),
        ("raw_turn_with_history", "raw_turn"),
        ("explicit_plus_top1_assumption", "explicit_only"),
        ("explicit_plus_top3_assumptions", "explicit_only"),
    )
    lift_rows = []
    for target, baseline in decomposition_contrasts:
        match = pairwise[
            (pairwise["analysis_subset"] == "assumption_eligible")
            & (pairwise["target_condition"] == target)
            & (pairwise["baseline_condition"] == baseline)
            & (pairwise["metric"] == "reciprocal_rank")
        ]
        if not match.empty:
            lift_rows.append(dict(match.iloc[0], contrast=f"{labels[target]} - {labels[baseline]}"))
    lift_df = pd.DataFrame(lift_rows)
    fig, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    if lift_df.empty or lift_df["mean_improvement"].isna().all():
        axis.text(0.5, 0.5, "No diagnostic lift data available", ha="center", va="center")
        axis.set_axis_off()
    else:
        means = lift_df["mean_improvement"].astype(float).to_numpy()
        x = np.arange(len(lift_df))
        axis.bar(x, means, color="#F58518")
        lows = lift_df["ci95_low"].to_numpy(dtype=float)
        highs = lift_df["ci95_high"].to_numpy(dtype=float)
        if np.isfinite(lows).all() and np.isfinite(highs).all():
            axis.errorbar(x, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", color="black", capsize=3)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, lift_df["contrast"].tolist(), rotation=25, ha="right")
        axis.set_ylabel("Paired MRR improvement")
        axis.set_title("Diagnostic decomposition — assumption-eligible pairs")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(paths["decomposition_pdf"], bbox_inches="tight")
    fig.savefig(paths["decomposition_png"], dpi=args.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepared_path(args)
    scores_file = final_paths(args.output_dir)["scores"]
    prepare_manifest = load_prepare_manifest(args)
    if not scores_file.exists():
        raise RuntimeError("Analysis requires a merged/root score JSONL file")
    logger.info(
        "Starting analysis stage: model=%s prompt=%s seed=%d conditions=%s",
        args.model_name,
        PROMPT_VERSION,
        args.seed,
        args.conditions,
    )
    pairs = read_jsonl(prepared)
    scores = read_jsonl(scores_file)
    seen: dict[tuple[str, str, str, str, str], str] = {}
    for row in scores:
        key = task_key(row)
        canonical = canonical_json(row)
        previous = seen.setdefault(key, canonical)
        if previous != canonical:
            raise RuntimeError(f"Conflicting duplicate score task key during analysis: {key}")
    long_df, wide_df = build_metrics(pairs, scores, args)
    category_df = condition_summary(long_df, "category", args)
    move_df = condition_summary(long_df, "true_next_turn_move_label", args)
    pairwise_df = build_pairwise(long_df, args)
    coverage_df = coverage_summary(long_df, pairs, args)
    decomposition_df = build_decomposition_table(pairwise_df)
    audit_df = build_audit_sample(pairs, long_df, args)
    gate = diagnostic_gate(pairwise_df, long_df, coverage_df)
    overall_metrics = overall_condition_summary(long_df, args)
    paths = final_paths(args.output_dir)
    long_df.to_csv(paths["metrics_long"], index=False)
    wide_df.to_csv(paths["metrics_wide"], index=False)
    category_df.to_csv(paths["by_category"], index=False)
    move_df.to_csv(paths["by_move"], index=False)
    pairwise_df.to_csv(paths["pairwise"], index=False)
    decomposition_df.to_csv(paths["decomposition"], index=False)
    audit_df.to_csv(paths["audit_sample"], index=False)
    write_json(paths["diagnostic_gate"], gate)
    coverage_df.to_csv(paths["coverage"], index=False)
    plot_results(long_df, pairwise_df, paths, args)
    hash_candidates = dict(paths)
    hash_candidates["prepared_pairs"] = prepared
    hash_candidates["prepare_manifest"] = prepare_manifest_path(args)
    merge_manifest_path = args.output_dir / "exp1_representation_merge_manifest.json"
    if merge_manifest_path.exists():
        hash_candidates["merge_manifest"] = merge_manifest_path
    root_patch_manifest = args.output_dir / "patch_manifest.json"
    if root_patch_manifest.exists():
        hash_candidates["score_manifest"] = root_patch_manifest
    output_hashes = {
        name: file_hash(path)
        for name, path in hash_candidates.items()
        if name != "summary" and path.exists()
    }
    complete_lookup = {
        row["condition"]: row
        for row in overall_metrics
        if row["analysis_subset"] == "complete_case"
    }
    complete_mrr_lifts = {
        str(row["target_condition"]): row["mean_improvement"]
        for _, row in pairwise_df.iterrows()
        if row["analysis_subset"] == "complete_case"
        and row["baseline_condition"] == "explicit_only"
        and row["metric"] == "reciprocal_rank"
    }
    diagnostic_table = [
        {
            "condition": condition,
            "mean_rank": complete_lookup.get(condition, {}).get("mean_rank"),
            "top1_rate": complete_lookup.get(condition, {}).get("top1_rate"),
            "mrr": complete_lookup.get(condition, {}).get("mrr"),
            "mrr_lift_vs_explicit_only": 0.0 if condition == "explicit_only" else complete_mrr_lifts.get(condition),
            "pairs": complete_lookup.get(condition, {}).get("pair_count", 0),
        }
        for condition in args.conditions
    ]
    summary = {
        "experiment": "Experiment 1: Explicit-Implicit Diagnostic Decomposition",
        "analysis_stage": "final_analysis",
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "git_commit": git_commit(),
        "runtime": runtime_versions(),
        "input_dir": prepare_manifest["input_dir"],
        "output_dir": str(args.output_dir),
        "categories": prepare_manifest["categories"],
        "selected_episode_files": prepare_manifest["selected_episode_files"],
        "input_manifest_sha256": prepare_manifest["input_manifest_sha256"],
        "prepared_pairs_sha256": prepare_manifest["prepared_pairs_sha256"],
        "model_name": args.model_name,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
        },
        "seed": args.seed,
        "candidate_count_target": EXPECTED_CANDIDATE_COUNT,
        "conditions": args.conditions,
        "history_turns": prepare_manifest["history_turns"],
        "pair_count": len(pairs),
        "candidate_complete_pair_count": prepare_manifest["candidate_complete_pair_count"],
        "filter_counts": {
            "episode_file_count": prepare_manifest["episode_file_count"],
            "source_episode_count": prepare_manifest["source_episode_count"],
            "turn_count": prepare_manifest["turn_count"],
            "pair_count_before_candidate_filter": prepare_manifest["pair_count_before_candidate_filter"],
            "candidate_complete_pair_count": prepare_manifest["candidate_complete_pair_count"],
            "candidate_incomplete_pair_count": prepare_manifest["candidate_incomplete_pair_count"],
            "assumption_eligible_pair_count": prepare_manifest["assumption_eligible_pair_count"],
        },
        "full_retained_by_condition": dict(zip(coverage_df["condition"], coverage_df["fully_parsed_pair_count"])),
        "complete_case_pair_count": int(long_df.loc[long_df["complete_case"] == True, "pair_id"].nunique()),
        "complete_case_removed_pair_count": len(pairs) - int(long_df.loc[long_df["complete_case"] == True, "pair_id"].nunique()),
        "unavailable_controls": prepare_manifest["unavailable_controls"],
        "unavailable_control_reasons": prepare_manifest["unavailable_control_reasons"],
        "parse_failures_by_condition": dict(zip(coverage_df["condition"], coverage_df["score_parse_failure_pair_count"])),
        "condition_metrics": overall_metrics,
        "complete_case_diagnostic_table": diagnostic_table,
        "diagnostic_gate": gate,
        "audit_sample_size": len(audit_df),
        "output_hashes": output_hashes,
        "summary_self_hash_excluded": True,
    }
    write_json(paths["summary"], summary)
    return summary


def merge_patch_scores(args: argparse.Namespace) -> dict[str, Any]:
    expected_dirs = [patch_dir(args.output_dir, index, args.num_patches) for index in range(args.num_patches)]
    manifests = []
    for expected_index, directory in enumerate(expected_dirs):
        manifest_path = directory / "patch_manifest.json"
        scores_file = score_path(directory)
        if not manifest_path.exists() or not scores_file.exists():
            raise RuntimeError(f"Missing patch output: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise RuntimeError(f"Incomplete patch manifest: {manifest_path}")
        if int(manifest.get("patch_index", -1)) != expected_index:
            raise RuntimeError(f"Patch index mismatch in {manifest_path}")
        if int(manifest.get("num_patches", -1)) != args.num_patches:
            raise RuntimeError(f"Patch count mismatch in {manifest_path}")
        manifests.append(manifest)
    for key in ("prepared_pairs_sha256", "config_sha256", "num_patches"):
        values = {canonical_json(manifest.get(key)) for manifest in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Mixed {key} values across patches")
    merged: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str, str], str] = {}
    for directory in expected_dirs:
        for row in read_jsonl(score_path(directory)):
            key = task_key(row)
            canonical = canonical_json(row)
            previous = seen.get(key)
            if previous is not None:
                if previous != canonical:
                    raise RuntimeError(f"Conflicting duplicate score task key during merge: {key}")
                continue
            seen[key] = canonical
            merged.append(row)
    merged.sort(key=lambda row: (row["pair_id"], row["condition"], int(row["candidate_order"])))
    destination = final_paths(args.output_dir)["scores"]
    write_jsonl(destination, merged)
    merge_manifest = {
        "stage": "merge",
        "complete": True,
        "patch_count": len(expected_dirs),
        "patch_dirs": [str(path) for path in expected_dirs],
        "prepared_pairs_sha256": manifests[0]["prepared_pairs_sha256"],
        "config_sha256": manifests[0]["config_sha256"],
        "merged_score_count": len(merged),
        "scores_sha256": file_hash(destination),
    }
    write_json(args.output_dir / "exp1_representation_merge_manifest.json", merge_manifest)
    logger.info("Merged %d score rows from %d patches", len(merged), len(expected_dirs))
    return merge_manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.merge_patches_only:
        print(json.dumps(merge_patch_scores(args), indent=2))
        return
    if args.prepare_only:
        print(json.dumps(prepare_dataset(args), indent=2))
        return
    if args.score_only:
        print(json.dumps(score_dataset(args), indent=2))
        return
    if args.analysis_only:
        print(json.dumps(analyze_dataset(args), indent=2))
        return
    prepare_dataset(args)
    score_dataset(args)
    print(json.dumps(analyze_dataset(args), indent=2))


if __name__ == "__main__":
    main()
