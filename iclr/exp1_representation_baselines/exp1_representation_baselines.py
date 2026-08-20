from __future__ import annotations

"""Explicit/implicit representation accuracy for future-turn prediction horizons."""

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
from collections import Counter, defaultdict
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


SCRIPT_VERSION = "6.0.4"
PROMPT_VERSION = "raw-augmentation-v1-json-evidence"
DEFAULT_INPUT_DIR = Path("data_cleaned/conversation_moves_labeled")
DEFAULT_OUTPUT_ROOT = Path("iclr/exp1_representation_baselines/results")
DEFAULT_PREPARED_NAME = "exp1_representation_prepared_pairs.jsonl"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_DOWNLOAD_DIR = Path("/shared/4/models")
DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_MAX_TOKENS = 64
DEFAULT_MAX_SCORE_RETRIES = 2
DEFAULT_MAX_RETRY_TOKENS = 128
DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS = 20
HARD_NEGATIVE_TARGET_COUNT = 24
SAME_EPISODE_NEGATIVE_TARGET = 12
SAME_MOVE_NEGATIVE_TARGET = 6
EXPECTED_CANDIDATE_COUNT = 25
EXPECTED_COMPARISONS_PER_CONDITION = HARD_NEGATIVE_TARGET_COUNT * 2
DEFAULT_SOURCE_TAIL_WORDS = 100
DEFAULT_CANDIDATE_HEAD_WORDS = 100
DEFAULT_ASSUMPTION_BUDGET = 3
DEFAULT_FUTURE_HORIZONS = (1, 3, 5)
DEFAULT_REPRESENTATION_BUDGETS = (256,)
REPRESENTATION_BUDGET_TOKENIZER = "whitespace_v1"
PLOT_REFERENCE_BASE_CONDITION = "raw_history"
EMPTY_EXPLICIT = "None extracted."
EMPTY_ASSUMPTIONS = "None extracted."
EMPTY_HISTORY = "No earlier substantive turn available."

MATCHED_BASE_CONDITIONS = (
    "raw_history",
    "raw_history_true_assumptions",
    "raw_history_different_episode_assumptions",
    "raw_history_same_episode_random_turn_assumptions",
    "raw_history_explicit",
)
DEFAULT_CONDITIONS = tuple(
    f"{condition}_b{budget}"
    for budget in DEFAULT_REPRESENTATION_BUDGETS
    for condition in MATCHED_BASE_CONDITIONS
)
LEGACY_CONDITIONS = (
    "raw_turn",
    "raw_turn_with_history",
    "raw_turn_plus_assumptions",
    "explicit_only",
    "explicit_plus_top3_assumptions",
    "explicit_plus_different_episode_assumptions",
    "explicit_plus_same_episode_random_turn_assumptions",
)
LEGACY_OPTIONAL_CONDITIONS = (
    "assumptions_only",
    "explicit_plus_top1_assumption",
    "explicit_plus_assumptions",
)
ALL_CONDITIONS = DEFAULT_CONDITIONS + LEGACY_CONDITIONS + LEGACY_OPTIONAL_CONDITIONS
CONTROL_CONDITIONS = (
    "raw_history_different_episode_assumptions",
    "raw_history_same_episode_random_turn_assumptions",
)
LEGACY_CONTROL_MAP = {
    "explicit_plus_different_episode_assumptions": "raw_history_different_episode_assumptions",
    "explicit_plus_same_episode_random_turn_assumptions": "raw_history_same_episode_random_turn_assumptions",
}
HEADLINE_CONSTRUCTIVE_MOVES = {
    "Assert / Elaborate",
    "Answer",
    "Agree / Align",
}
DIAGNOSTIC_CONTRASTS = (
    ("explicit_plus_top3_assumptions", "explicit_only"),
    ("explicit_plus_top3_assumptions", "explicit_plus_different_episode_assumptions"),
    ("explicit_plus_top3_assumptions", "explicit_plus_same_episode_random_turn_assumptions"),
    ("raw_turn_plus_assumptions", "raw_turn"),
    ("raw_turn", "explicit_only"),
    ("raw_turn_with_history", "raw_turn"),
    ("explicit_plus_top1_assumption", "explicit_only"),
    ("explicit_plus_assumptions", "explicit_plus_top1_assumption"),
    ("explicit_plus_assumptions", "explicit_plus_top3_assumptions"),
)
OPTIONAL_CONTROL_CONTRASTS = (
    ("explicit_plus_assumptions", "assumptions_only"),
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
    source_tail_text: str
    candidate_head_text: str
    word_count: int
    explicit_texts: list[str]
    assumption_texts: list[str]
    all_assumption_texts: list[str]
    history_turn_ids: list[str]
    history_turn_texts: list[str]
    history_turn_records: list[dict[str, Any]]
    original_turn_indices: list[int]
    merge_provenance_present: bool


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


class ParsedChoice(TypedDict):
    choice: str | None
    evidence: str | None
    parse_success: bool
    parse_error: str | None
    parse_method: str | None


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


def matched_condition_id(base_condition: str, budget: int) -> str:
    if base_condition not in MATCHED_BASE_CONDITIONS:
        raise ValueError(f"Unknown matched-history base condition: {base_condition}")
    return f"{base_condition}_b{int(budget)}"


def parse_matched_condition(condition: str) -> tuple[str | None, int | None]:
    match = re.fullmatch(r"(.+)_b([1-9][0-9]*)", condition)
    if not match or match.group(1) not in MATCHED_BASE_CONDITIONS:
        return None, None
    return match.group(1), int(match.group(2))


def condition_donor_key(condition: str) -> str | None:
    base_condition, _ = parse_matched_condition(condition)
    if base_condition in CONTROL_CONDITIONS:
        return base_condition
    if condition in CONTROL_CONDITIONS:
        return condition
    return LEGACY_CONTROL_MAP.get(condition)


def normalize_conditions(values: list[str] | None, budgets: list[int]) -> list[str]:
    if not values or any(value.casefold() == "all" for value in values):
        return [
            matched_condition_id(condition, budget)
            for budget in budgets
            for condition in MATCHED_BASE_CONDITIONS
        ]
    chosen: list[str] = []
    for value in values:
        expanded = (
            [matched_condition_id(value, budget) for budget in budgets]
            if value in MATCHED_BASE_CONDITIONS
            else [value]
        )
        for condition in expanded:
            base, condition_budget = parse_matched_condition(condition)
            valid_dynamic = base is not None and condition_budget in budgets
            if condition not in ALL_CONDITIONS and not valid_dynamic:
                choices = (*MATCHED_BASE_CONDITIONS, *LEGACY_CONDITIONS, *LEGACY_OPTIONAL_CONDITIONS)
                raise ValueError(f"Unknown condition {value!r}; choose from {', '.join(choices)}")
            if condition not in chosen:
                chosen.append(condition)
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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Initial JSON judgment generation token budget.",
    )
    parser.add_argument(
        "--max_score_retries",
        type=int,
        default=DEFAULT_MAX_SCORE_RETRIES,
        help="Retry malformed/incomplete judge outputs this many times.",
    )
    parser.add_argument(
        "--max_retry_tokens",
        type=int,
        default=DEFAULT_MAX_RETRY_TOKENS,
        help="Maximum generation token budget used by parse-failure retries.",
    )
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
    parser.add_argument(
        "--representation_budgets",
        nargs="+",
        type=int,
        default=list(DEFAULT_REPRESENTATION_BUDGETS),
        help="Whitespace-token caps for matched raw-history and structured-state conditions.",
    )
    parser.add_argument("--future_horizons", nargs="+", type=int, default=list(DEFAULT_FUTURE_HORIZONS))
    parser.add_argument("--source_tail_words", type=int, default=DEFAULT_SOURCE_TAIL_WORDS)
    parser.add_argument("--candidate_head_words", type=int, default=DEFAULT_CANDIDATE_HEAD_WORDS)
    parser.add_argument("--assumption_budget", type=int, default=DEFAULT_ASSUMPTION_BUDGET)
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
    args.representation_budgets = list(dict.fromkeys(args.representation_budgets))
    if not args.representation_budgets or any(value < 32 for value in args.representation_budgets):
        raise ValueError("representation_budgets must contain unique positive values of at least 32")
    args.future_horizons = list(dict.fromkeys(args.future_horizons))
    if not args.future_horizons:
        raise ValueError("At least one future horizon is required")
    if any(horizon < 1 or horizon % 2 == 0 for horizon in args.future_horizons):
        raise ValueError(
            "future_horizons must contain positive odd integers so the target is spoken "
            "by the other speaker in the cleaned ABAB dialogue"
        )
    if args.source_tail_words < 1 or args.candidate_head_words < 1:
        raise ValueError("source_tail_words and candidate_head_words must be positive")
    if args.assumption_budget < 1:
        raise ValueError("assumption_budget must be positive")
    if args.prompt_batch_size < 1 or args.max_tokens < 1 or args.bootstrap_draws < 1:
        raise ValueError("Batch size, max tokens, and bootstrap draws must be positive")
    if args.max_score_retries < 0:
        raise ValueError("max_score_retries cannot be negative")
    if args.max_score_retries > 0 and args.max_retry_tokens <= args.max_tokens:
        raise ValueError("max_retry_tokens must be greater than max_tokens when retries are enabled")
    if args.audit_sample_size_per_outcome < 1:
        raise ValueError("audit_sample_size_per_outcome must be positive")
    args.conditions = normalize_conditions(args.conditions, args.representation_budgets)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / model_output_name(args.model_name)
    required = {
        matched_condition_id(condition, budget)
        for budget in args.representation_budgets
        for condition in MATCHED_BASE_CONDITIONS
    }
    if args.strict_all_conditions and not required.issubset(args.conditions):
        raise ValueError("strict_all_conditions requires all five matched conditions at every budget")


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
    if list(args.representation_budgets) != list(
        manifest.get("representation_budgets", args.representation_budgets)
    ):
        raise RuntimeError("Scoring/analysis representation budgets must match preparation")
    if list(args.future_horizons) != list(manifest.get("future_horizons", args.future_horizons)):
        raise RuntimeError("Scoring/analysis future horizons must match preparation")
    for argument_name in ("source_tail_words", "candidate_head_words", "assumption_budget"):
        if int(getattr(args, argument_name)) != int(manifest.get(argument_name, getattr(args, argument_name))):
            raise RuntimeError(f"Scoring/analysis {argument_name} must match preparation")
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
        "flips": output_dir / "exp1_representation_flip_analysis.csv",
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


def text_words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def tail_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[-limit:])


def head_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


def lexical_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
        "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
        "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "them",
        "they", "this", "to", "was", "we", "were", "will", "with", "would", "you",
    }
    return {
        token.casefold()
        for token in text_words(text)
        if len(token) > 2 and token.casefold() not in stopwords
    }


def rank_assumptions_by_confidence(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected a list of assumption strings or objects, got {type(value).__name__}")

    ranked: list[tuple[int, float, int, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        raw_text = item.get("text") if isinstance(item, dict) else item
        if raw_text is None:
            continue
        if not isinstance(raw_text, str):
            raise TypeError(
                f"Assumption item {index} must be a string or object with a string text field"
            )
        text = raw_text.strip()
        if not text or text in seen:
            continue

        raw_confidence = None
        if isinstance(item, dict):
            raw_confidence = item.get("confidence", item.get("confidence_score"))
        if raw_confidence is None:
            missing_confidence = 1
            descending_confidence = 0.0
        else:
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
                raise TypeError(
                    f"Assumption item {index} confidence must be a number between 0 and 1"
                )
            confidence = float(raw_confidence)
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Assumption item {index} confidence must be finite and between 0 and 1, "
                    f"got {raw_confidence!r}"
                )
            missing_confidence = 0
            descending_confidence = -confidence

        seen.add(text)
        ranked.append((missing_confidence, descending_confidence, index, text))

    ranked.sort(key=lambda row: row[:3])
    return [text for _, _, _, text in ranked]


def original_turn_indices(turn: dict[str, Any], fallback: int) -> tuple[list[int], bool]:
    raw = turn.get("merged_from_turn_indices")
    if raw is None:
        raw = turn.get("merged_from_turn_ids")
    if isinstance(raw, list) and raw:
        values: list[int] = []
        for value in raw:
            try:
                values.append(int(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid merged-turn provenance value: {value!r}") from error
        return values, True
    raw_index = turn.get("turn_idx", fallback)
    try:
        return [int(raw_index)], False
    except (TypeError, ValueError):
        return [fallback], False


def provenance_is_contiguous(indices: list[int]) -> bool:
    return bool(indices) and all(right == left + 1 for left, right in zip(indices, indices[1:]))


def build_episode_records(
    category: str,
    path: Path,
    history_turns: int,
    source_tail_words: int,
    candidate_head_words: int,
    assumption_budget: int,
    future_horizons: list[int],
) -> tuple[list[TurnRecord], list[dict[str, Any]], dict[str, int]]:
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
        provenance_indices, provenance_present = original_turn_indices(turn, list_position)
        source_tail = tail_words(text, source_tail_words)
        all_assumptions = rank_assumptions_by_confidence(turn.get("assumptions"))
        substantive_position += 1
        previous = history[-history_turns:] if history_turns else []
        previous_records = [
            {
                "turn_id": item["turn_id"],
                "turn_idx": item["turn_idx"],
                "substantive_position": item["substantive_position"],
                "turn_text": item["turn_text"],
                "explicit_texts": list(item["explicit_texts"]),
                "assumption_texts": list(item["assumption_texts"]),
            }
            for item in previous
        ]
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
            "source_tail_text": source_tail,
            "candidate_head_text": head_words(text, candidate_head_words),
            "word_count": len(text_words(text)),
            "explicit_texts": normalize_text_list(turn.get("explicit_propositions")),
            "assumption_texts": all_assumptions[:assumption_budget],
            "all_assumption_texts": all_assumptions,
            "history_turn_ids": [item["turn_id"] for item in previous],
            "history_turn_texts": [item["source_tail_text"] for item in previous],
            "history_turn_records": previous_records,
            "original_turn_indices": provenance_indices,
            "merge_provenance_present": provenance_present,
        }
        turns.append(record)
        history.append(record)
        substantive_by_list_position[list_position] = record

    pairs: list[dict[str, Any]] = []
    boundary_counts = {
        "boundary_invalid_merged_group_count": 0,
        "boundary_verified_pair_count": 0,
        "boundary_unverified_pair_count": 0,
    }
    for list_position, source in sorted(substantive_by_list_position.items()):
        for future_horizon in future_horizons:
            target_position = list_position + future_horizon
            span = [
                substantive_by_list_position.get(position)
                for position in range(list_position, target_position + 1)
            ]
            if any(record is None for record in span):
                continue
            verified_span = [record for record in span if record is not None]
            target = verified_span[-1]
            if any(
                not provenance_is_contiguous(record["original_turn_indices"])
                for record in verified_span
            ):
                boundary_counts["boundary_invalid_merged_group_count"] += 1
                continue
            boundary_verified = all(record["merge_provenance_present"] for record in verified_span)
            if boundary_verified and any(
                left["original_turn_indices"][-1] + 1 != right["original_turn_indices"][0]
                for left, right in zip(verified_span, verified_span[1:])
            ):
                boundary_counts["boundary_invalid_merged_group_count"] += 1
                continue
            count_key = "boundary_verified_pair_count" if boundary_verified else "boundary_unverified_pair_count"
            boundary_counts[count_key] += 1
            pair_id = (
                f"{category}:{target['episode_id']}:{source['turn_idx']}:"
                f"{target['turn_idx']}:h{future_horizon}"
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "future_horizon": future_horizon,
                    "category": category,
                    "episode_id": target["episode_id"],
                    "source_path": str(path),
                    "source_turn_id": source["turn_id"],
                    "source_turn_idx": source["turn_idx"],
                    "source_substantive_position": source["substantive_position"],
                    "source_turn_text": source["turn_text"],
                    "source_tail_text": source["source_tail_text"],
                    "source_word_count": source["word_count"],
                    "source_explicit_texts": source["explicit_texts"],
                    "source_assumption_texts": source["assumption_texts"],
                    "source_all_assumption_texts": source["all_assumption_texts"],
                    "source_original_turn_indices": source["original_turn_indices"],
                    "true_next_original_turn_indices": target["original_turn_indices"],
                    "original_boundary_verified": boundary_verified,
                    "history_turn_ids": source["history_turn_ids"],
                    "history_turn_texts": source["history_turn_texts"],
                    "context_turns": [
                        *source["history_turn_records"],
                        {
                            "turn_id": source["turn_id"],
                            "turn_idx": source["turn_idx"],
                            "substantive_position": source["substantive_position"],
                            "turn_text": source["turn_text"],
                            "explicit_texts": list(source["explicit_texts"]),
                            "assumption_texts": list(source["assumption_texts"]),
                        },
                    ],
                    "history_turn_count": len(source["history_turn_ids"]),
                    "true_next_turn_id": target["turn_id"],
                    "true_next_turn_idx": target["turn_idx"],
                    "true_next_turn_text": target["turn_text"],
                    "true_next_turn_head_text": target["candidate_head_text"],
                    "true_next_turn_move_label": target["move_label"],
                    "candidate_pool_complete": False,
                    "coverage_drop_reason": None,
                    "candidate_pool_sha256": None,
                    "candidates": [],
                    "conditions": {},
                    "donors": {},
                }
            )
    return turns, pairs, boundary_counts


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


def reserve_same_episode_random_turn_assumption_donor(
    pair: dict[str, Any],
    turns: list[TurnRecord],
    seed: int,
) -> str | None:
    """Reserve an eligible prior-turn donor from the same episode so candidate sampling cannot select it."""
    excluded_ids = set(pair["history_turn_ids"]) | {
        pair["source_turn_id"],
        pair["true_next_turn_id"],
    }
    source_position = int(pair["source_substantive_position"])
    target_words = auxiliary_word_budget(pair)
    if target_words < 1:
        return None
    eligible = [
        turn
        for turn in turns
        if turn["turn_id"] not in excluded_ids
        and turn["category"] == pair["category"]
        and turn["episode_id"] == pair["episode_id"]
        and turn["assumption_texts"]
        and turn["assumption_texts"] != pair["source_assumption_texts"]
        and source_position - int(turn["substantive_position"]) >= 3
        and assumption_word_count(turn.get("all_assumption_texts") or turn["assumption_texts"]) >= target_words
    ]
    if not eligible:
        return None
    ranked = sorted(
        eligible,
        key=lambda turn: (
            -lexical_similarity(pair["source_tail_text"], turn["source_tail_text"]),
            source_position - int(turn["substantive_position"]),
            seed_int(f"{pair['pair_id']}:same_episode_random_turn:{turn['turn_id']}:{seed}"),
        ),
    )
    return ranked[0]["turn_id"]


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


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)
    union = left_tokens.union(right_tokens)
    if not union:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(union)


def length_similarity(left_count: int, right_count: int) -> float:
    if left_count < 1 or right_count < 1:
        return 0.0
    return min(left_count, right_count) / max(left_count, right_count)


def rank_hard_negative_ids(
    ids: list[str],
    pair: dict[str, Any],
    lookup: dict[str, TurnRecord],
    count: int,
    label: str,
    seed: int,
) -> list[str]:
    unique_ids = list(dict.fromkeys(ids))
    ranked: list[tuple[float, int, str]] = []
    for turn_id in unique_ids:
        turn = lookup[turn_id]
        topic_score = lexical_similarity(pair["source_tail_text"], turn["candidate_head_text"])
        size_score = length_similarity(
            len(text_words(pair["true_next_turn_head_text"])),
            len(text_words(turn["candidate_head_text"])),
        )
        hardness = 0.7 * topic_score + 0.3 * size_score
        tie_break = seed_int(f"{pair['pair_id']}:{label}:{turn_id}:{seed}")
        ranked.append((-hardness, tie_break, turn_id))
    ranked.sort()
    return [turn_id for _, _, turn_id in ranked[:count]]


def build_candidates(pair: dict[str, Any], indexes: dict[str, Any], seed: int) -> dict[str, int]:
    lookup: dict[str, TurnRecord] = indexes["lookup"]
    target = lookup[pair["true_next_turn_id"]]
    history_ids = set(pair["history_turn_ids"])
    history_texts = set(pair["history_turn_texts"])
    excluded_ids = history_ids | {pair["source_turn_id"], pair["true_next_turn_id"]}
    reserved_same_episode_random_turn_donor_id = pair.get("reserved_same_episode_random_turn_donor_id")
    if reserved_same_episode_random_turn_donor_id:
        excluded_ids.add(str(reserved_same_episode_random_turn_donor_id))
    if pair["true_next_turn_head_text"] in history_texts:
        pair["coverage_drop_reason"] = "true_candidate_text_in_history"
        return {}
    used: set[str] = set(excluded_ids)
    selected: list[tuple[str, str]] = []
    counts = {
        "negative_count_same_episode_hard": 0,
        "negative_count_same_category_same_move": 0,
        "negative_count_same_category_topic_length": 0,
        "negative_count_global_topic_length": 0,
    }

    def eligible(ids: list[str], outside_episode: bool | None) -> list[str]:
        rows: list[str] = []
        for turn_id in ids:
            turn = lookup[turn_id]
            if turn_id in used or turn["turn_text"] in history_texts:
                continue
            if outside_episode is True and turn["episode_id"] == pair["episode_id"]:
                continue
            rows.append(turn_id)
        return rows

    layers = [
        (
            "same_episode_hard",
            [
                turn_id
                for turn_id in eligible(
                    indexes["by_episode"].get((pair["category"], pair["episode_id"]), []),
                    None,
                )
                if abs(int(lookup[turn_id]["substantive_position"]) - int(target["substantive_position"])) >= 3
            ],
            SAME_EPISODE_NEGATIVE_TARGET,
        ),
        (
            "same_category_same_move",
            eligible(
                indexes["by_category_move"].get(
                    (pair["category"], pair["true_next_turn_move_label"]),
                    [],
                ),
                True,
            ),
            SAME_MOVE_NEGATIVE_TARGET,
        ),
        (
            "same_category_topic_length",
            eligible(indexes["by_category"].get(pair["category"], []), True),
            HARD_NEGATIVE_TARGET_COUNT,
        ),
        (
            "global_topic_length",
            eligible(indexes["global"], True),
            HARD_NEGATIVE_TARGET_COUNT,
        ),
    ]
    for label, raw_pool, layer_target in layers:
        remaining = HARD_NEGATIVE_TARGET_COUNT - len(selected)
        if remaining <= 0:
            break
        pool = [turn_id for turn_id in raw_pool if turn_id not in used]
        chosen = rank_hard_negative_ids(
            pool,
            pair,
            lookup,
            min(layer_target, remaining),
            label,
            seed,
        )
        selected.extend((turn_id, label) for turn_id in chosen)
        used.update(chosen)
        counts[f"negative_count_{label}"] = len(chosen)
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
                "candidate_text": turn["candidate_head_text"],
                "is_true_next_turn": is_true,
                "negative_source": source,
            }
        )
    pair["candidates"] = candidates
    pair["candidate_pool_sha256"] = stable_hash([row["candidate_turn_id"] for row in candidates])
    pair["candidate_pool_complete"] = True
    pair["coverage_drop_reason"] = None
    return counts


def assumption_word_count(values: list[str]) -> int:
    return sum(len(text_words(value)) for value in values)


def truncate_text_items_to_words(values: list[str], budget: int) -> tuple[list[str], int]:
    """Truncate ordered text items to exactly `budget` content words when possible."""
    if budget <= 0:
        return [], 0
    selected: list[str] = []
    remaining = int(budget)
    for value in values:
        words = text_words(str(value))
        if not words or remaining <= 0:
            continue
        take = min(len(words), remaining)
        selected.append(" ".join(words[:take]))
        remaining -= take
        if remaining == 0:
            break
    return selected, budget - remaining


def auxiliary_word_budget(pair: dict[str, Any]) -> int:
    return assumption_word_count(list(pair.get("source_assumption_texts", [])))


def context_explicit_values(pair: dict[str, Any]) -> list[str]:
    context = deduplicate_context_state(list(pair.get("context_turns", [])))
    values: list[str] = []
    for turn in reversed(context):
        values.extend(str(value) for value in turn.get("explicit_texts", []))
    return values


def choose_donors(
    pair: dict[str, Any],
    turns: list[TurnRecord],
    indexes: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    candidate_ids = {row["candidate_turn_id"] for row in pair["candidates"]}
    excluded_ids = candidate_ids | set(pair["history_turn_ids"]) | {
        pair["source_turn_id"],
        pair["true_next_turn_id"],
    }
    source_assumptions = pair["source_assumption_texts"]
    source_position = int(pair["source_substantive_position"])
    target_words = auxiliary_word_budget(pair)

    def donor_values(turn: TurnRecord) -> list[str]:
        return list(turn.get("all_assumption_texts") or turn.get("assumption_texts") or [])

    def base_eligible(turn: TurnRecord) -> bool:
        return bool(
            target_words > 0
            and turn["turn_id"] not in excluded_ids
            and turn["assumption_texts"]
            and turn["assumption_texts"] != source_assumptions
            and assumption_word_count(donor_values(turn)) >= target_words
        )

    different_episode_same_category = [
        turn
        for turn in turns
        if base_eligible(turn)
        and turn["episode_id"] != pair["episode_id"]
        and turn["category"] == pair["category"]
    ]
    different_episode_any = [
        turn
        for turn in turns
        if base_eligible(turn) and turn["episode_id"] != pair["episode_id"]
    ]
    different_episode_pool = different_episode_same_category or different_episode_any
    different_episode_level = "hard_same_category" if different_episode_same_category else "hard_any_category"
    different_episode = None
    if different_episode_pool:
        different_episode = sorted(
            different_episode_pool,
            key=lambda turn: (
                -lexical_similarity(pair["source_tail_text"], turn["source_tail_text"]),
                -length_similarity(pair["source_word_count"], turn["word_count"]),
                seed_int(f"{pair['pair_id']}:different_episode:{turn['turn_id']}:{seed}"),
            ),
        )[0]

    same_episode_random_turn = None
    reserved_same_episode_random_turn_donor_id = pair.get("reserved_same_episode_random_turn_donor_id")
    if reserved_same_episode_random_turn_donor_id:
        matches = [turn for turn in turns if turn["turn_id"] == reserved_same_episode_random_turn_donor_id]
        if len(matches) != 1:
            raise ValueError(
                f"Reserved same-episode random-turn donor became invalid for {pair['pair_id']}: {reserved_same_episode_random_turn_donor_id}"
            )
        same_episode_random_turn = matches[0]

    rows: list[dict[str, Any]] = []
    for condition, donor, fallback, unavailable_reason in (
        (
            "raw_history_different_episode_assumptions",
            different_episode,
            different_episode_level if different_episode is not None else None,
            None if different_episode is not None else (
                "source_has_no_assumptions" if target_words == 0 else "no_length_matched_different_episode_assumption_donor"
            ),
        ),
        (
            "raw_history_same_episode_random_turn_assumptions",
            same_episode_random_turn,
            "same_episode_prior_turn_gap3" if same_episode_random_turn is not None else None,
            None if same_episode_random_turn is not None else (
                "source_has_no_assumptions" if target_words == 0 else "no_length_matched_same_episode_random_turn_donor"
            ),
        ),
    ):
        donor_all = donor_values(donor) if donor else []
        matched_assumptions, matched_words = truncate_text_items_to_words(donor_all, target_words)
        if donor is not None and matched_words != target_words:
            raise AssertionError(f"Donor word matching failed for {pair['pair_id']} / {condition}")
        audit = {
            "pair_id": pair["pair_id"],
            "future_horizon": int(pair["future_horizon"]),
            "condition": condition,
            "source_turn_id": pair["source_turn_id"],
            "source_substantive_position": source_position,
            "target_auxiliary_word_count": target_words,
            "donor_turn_id": donor["turn_id"] if donor else None,
            "donor_episode_id": donor["episode_id"] if donor else None,
            "donor_category": donor["category"] if donor else None,
            "donor_substantive_position": donor["substantive_position"] if donor else None,
            "donor_fallback_level": fallback,
            "donor_assumption_count": len(donor["assumption_texts"]) if donor else 0,
            "donor_assumptions": donor["assumption_texts"] if donor else [],
            "donor_all_assumptions": donor_all,
            "matched_donor_assumptions": matched_assumptions,
            "matched_auxiliary_word_count": matched_words,
            "control_unavailable_reason": unavailable_reason,
        }
        pair["donors"][condition] = audit
        rows.append(audit)
    return rows


def format_bullets(values: list[str], empty_value: str) -> str:
    return "\n".join(f"- {value}" for value in values) if values else empty_value


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def deduplicate_context_state(context_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain the most recent occurrence of each state item without changing turn order."""
    retained: list[dict[str, Any]] = [dict(turn, explicit_texts=[], assumption_texts=[]) for turn in context_turns]
    for field in ("explicit_texts", "assumption_texts"):
        seen: set[str] = set()
        for index in range(len(context_turns) - 1, -1, -1):
            values = list(context_turns[index].get(field, []))
            chosen: list[str] = []
            for value in reversed(values):
                key = re.sub(r"\s+", " ", str(value)).strip().casefold()
                if key and key not in seen:
                    seen.add(key)
                    chosen.append(str(value).strip())
            retained[index][field] = list(reversed(chosen))
    return retained


def context_turn_header(index: int, count: int, kind: str | None = None) -> str:
    offset = index - (count - 1)
    turn_name = "t (current)" if offset == 0 else f"t{offset}"
    suffix = f" {kind}" if kind else ""
    return f"[Turn {turn_name}{suffix}]"


def format_raw_history(pair: dict[str, Any], budget: int) -> str:
    context = list(pair["context_turns"])
    headers = [context_turn_header(index, len(context), "raw") for index in range(len(context))]
    remaining = budget - sum(whitespace_token_count(header) for header in headers)
    if remaining < 0:
        raise ValueError(f"Budget {budget} is too small for {len(context)} turn headers")
    selected: list[list[str]] = [[] for _ in context]
    for index in range(len(context) - 1, -1, -1):
        if remaining == 0:
            break
        words = str(context[index]["turn_text"]).split()
        take = min(remaining, len(words))
        selected[index] = words[-take:] if take else []
        remaining -= take
    lines: list[str] = []
    for header, words in zip(headers, selected):
        lines.append(header)
        if words:
            lines.append(" ".join(words))
    representation = "\n".join(lines)
    if whitespace_token_count(representation) > budget:
        raise AssertionError("Raw history exceeded its representation budget")
    return representation


def replace_context_assumptions(
    context: list[dict[str, Any]],
    donor_assumptions: list[str],
) -> list[dict[str, Any]]:
    replacement = [dict(turn, assumption_texts=[]) for turn in context]
    cursor = 0
    for index in range(len(context) - 1, -1, -1):
        turn = context[index]
        count = len(turn.get("assumption_texts", []))
        replacement[index]["assumption_texts"] = donor_assumptions[cursor:cursor + count]
        cursor += count
    return replacement


def format_structured_history(
    pair: dict[str, Any],
    budget: int,
    *,
    include_assumptions: bool,
    donor_assumptions: list[str] | None = None,
) -> str:
    context = deduplicate_context_state(list(pair["context_turns"]))
    if donor_assumptions is not None:
        context = replace_context_assumptions(context, list(dict.fromkeys(donor_assumptions)))
    headers: list[tuple[str, str, str | None]] = []
    for index in range(len(context)):
        headers.append(
            (
                context_turn_header(index, len(context)),
                "Explicit:",
                "Implicit:" if include_assumptions else None,
            )
        )
    fixed_words = sum(
        whitespace_token_count(line)
        for header_group in headers
        for line in header_group
        if line is not None
    )
    remaining = budget - fixed_words
    if remaining < 0:
        raise ValueError(f"Budget {budget} is too small for {len(context)} structured turn headers")
    selected_explicit: list[list[str]] = [[] for _ in context]
    selected_implicit: list[list[str]] = [[] for _ in context]
    for index in range(len(context) - 1, -1, -1):
        explicit_values = list(context[index].get("explicit_texts", []))
        implicit_values = list(context[index].get("assumption_texts", [])) if include_assumptions else []
        for item_index in range(max(len(explicit_values), len(implicit_values))):
            for values, destination in (
                (explicit_values, selected_explicit[index]),
                (implicit_values, selected_implicit[index]),
            ):
                if item_index >= len(values) or remaining <= 1:
                    continue
                words = str(values[item_index]).split()
                take = min(len(words), remaining - 1)
                if take:
                    destination.append("- " + " ".join(words[:take]))
                    remaining -= take + 1
            if remaining <= 1:
                break
        if remaining <= 1:
            break
    lines: list[str] = []
    for index, (turn_header, explicit_header, implicit_header) in enumerate(headers):
        lines.extend((turn_header, explicit_header, *selected_explicit[index]))
        if implicit_header is not None:
            lines.extend((implicit_header, *selected_implicit[index]))
    representation = "\n".join(lines)
    if whitespace_token_count(representation) > budget:
        raise AssertionError("Structured history exceeded its representation budget")
    return representation


def format_representation(pair: dict[str, Any], condition: str) -> str:
    base_condition, budget = parse_matched_condition(condition)
    if base_condition is not None and budget is not None:
        raw = format_raw_history(pair, budget)
        target_words = auxiliary_word_budget(pair)
        if base_condition == "raw_history":
            return raw
        if base_condition == "raw_history_true_assumptions":
            selected, observed = truncate_text_items_to_words(
                list(pair.get("source_assumption_texts", [])), target_words
            )
            if target_words < 1 or observed != target_words:
                raise ValueError(f"Condition {condition} is unavailable for {pair['pair_id']}")
            return raw + "\n\n[Implicit assumptions]\n" + " ".join(selected)
        if base_condition in {"raw_history_different_episode_assumptions", "raw_history_same_episode_random_turn_assumptions"}:
            donor = pair["donors"].get(base_condition, {})
            if donor.get("control_unavailable_reason"):
                raise ValueError(f"Condition {condition} is unavailable for {pair['pair_id']}")
            selected = list(donor.get("matched_donor_assumptions") or [])
            if int(donor.get("matched_auxiliary_word_count") or 0) != target_words:
                raise ValueError(f"Length-matched donor missing for {pair['pair_id']} / {condition}")
            return raw + "\n\n[Implicit assumptions]\n" + " ".join(selected)
        if base_condition == "raw_history_explicit":
            selected, observed = truncate_text_items_to_words(context_explicit_values(pair), target_words)
            if target_words < 1 or observed != target_words:
                raise ValueError(f"Condition {condition} is unavailable for {pair['pair_id']}")
            return raw + "\n\n[Explicit propositions]\n" + " ".join(selected)
        raise ValueError(f"Unknown matched-history condition: {condition}")

    # Legacy representations are retained for backwards-compatible reanalysis.
    explicit = format_bullets(pair["source_explicit_texts"], EMPTY_EXPLICIT)
    all_assumptions = format_bullets(pair["source_all_assumption_texts"], EMPTY_ASSUMPTIONS)
    top1_assumption = format_bullets(pair["source_assumption_texts"][:1], EMPTY_ASSUMPTIONS)
    top3_assumptions = format_bullets(pair["source_assumption_texts"][:3], EMPTY_ASSUMPTIONS)
    if condition == "raw_turn":
        return f"[Final local window of the current turn]\n{pair['source_tail_text']}"
    if condition == "raw_turn_with_history":
        history = "\n\n".join(pair.get("history_turn_texts", [])) or EMPTY_HISTORY
        return f"[Earlier substantive history]\n{history}\n\n[Final local window of the current turn]\n{pair['source_tail_text']}"
    if condition == "explicit_only":
        return f"[Explicit propositions]\n{explicit}"
    if condition == "assumptions_only":
        return f"[Implicit assumptions]\n{all_assumptions}"
    if condition == "explicit_plus_assumptions":
        return f"[Explicit propositions]\n{explicit}\n\n[All extracted implicit assumptions]\n{all_assumptions}"
    if condition == "explicit_plus_top1_assumption":
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions: first 1]\n{top1_assumption}"
    if condition == "explicit_plus_top3_assumptions":
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions: first 3]\n{top3_assumptions}"
    if condition == "raw_turn_plus_assumptions":
        return f"[Final local window of the current turn]\n{pair['source_tail_text']}\n\n[Implicit assumptions]\n{top3_assumptions}"
    if condition in LEGACY_CONTROL_MAP:
        donor = pair["donors"].get(LEGACY_CONTROL_MAP[condition], {})
        if donor.get("control_unavailable_reason"):
            raise ValueError(f"Condition {condition} is unavailable for {pair['pair_id']}")
        donor_text = format_bullets(list(donor.get("matched_donor_assumptions") or donor.get("donor_assumptions") or []), EMPTY_ASSUMPTIONS)
        return f"[Explicit propositions]\n{explicit}\n\n[Implicit assumptions]\n{donor_text}"
    raise ValueError(f"Unknown condition: {condition}")

def build_conditions(pair: dict[str, Any], conditions: list[str]) -> None:
    target_words = auxiliary_word_budget(pair)
    for condition in conditions:
        base_condition, budget = parse_matched_condition(condition)
        donor_key = condition_donor_key(condition)
        donor = pair["donors"].get(donor_key) if donor_key else None
        unavailable = donor.get("control_unavailable_reason") if donor else None
        if base_condition in {"raw_history_true_assumptions", "raw_history_explicit"} and target_words < 1:
            unavailable = "source_has_no_assumptions"
        if base_condition == "raw_history_explicit" and target_words > 0:
            _, explicit_words = truncate_text_items_to_words(context_explicit_values(pair), target_words)
            if explicit_words != target_words:
                unavailable = "insufficient_explicit_words_for_length_match"
        if unavailable:
            pair["conditions"][condition] = {
                "available": False,
                "control_unavailable_reason": unavailable,
                "source_representation": None,
                "base_condition": base_condition,
                "representation_budget": budget,
                "representation_token_count": None,
                "raw_history_token_count": None,
                "auxiliary_token_count": None,
                "raw_history_sha256": None,
                "budget_tokenizer": REPRESENTATION_BUDGET_TOKENIZER if budget else None,
            }
            continue
        representation = format_representation(pair, condition)
        raw = format_raw_history(pair, budget) if base_condition is not None and budget is not None else None
        raw_count = whitespace_token_count(raw) if raw is not None else None
        total_count = whitespace_token_count(representation)
        auxiliary_count = total_count - raw_count if raw_count is not None else None
        pair["conditions"][condition] = {
            "available": True,
            "control_unavailable_reason": None,
            "source_representation": representation,
            "base_condition": base_condition,
            "representation_budget": budget,
            "representation_token_count": total_count,
            "raw_history_token_count": raw_count,
            "auxiliary_token_count": auxiliary_count,
            "raw_history_sha256": sha256_text(raw) if raw is not None else None,
            "target_auxiliary_content_word_count": target_words if base_condition != "raw_history" else 0,
            "budget_tokenizer": REPRESENTATION_BUDGET_TOKENIZER if budget else None,
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
        if donor["donor_turn_id"] in set(ids) | history_ids | {
            pair["source_turn_id"],
            pair["true_next_turn_id"],
        }:
            raise ValueError(f"Donor/candidate leakage for {pair['pair_id']} / {condition}")
        if condition == "raw_history_same_episode_random_turn_assumptions":
            donor_turn = indexes["lookup"][donor["donor_turn_id"]]
            if donor_turn["episode_id"] != source["episode_id"]:
                raise ValueError(f"Stale control crossed episodes for {pair['pair_id']}")
            donor_distance = int(source["substantive_position"]) - int(
                donor_turn["substantive_position"]
            )
            if donor_distance < 3:
                raise ValueError(
                    f"Stale control is not an earlier gap-three donor for {pair['pair_id']}"
                )
    for condition in conditions:
        metadata = pair["conditions"].get(condition)
        if metadata is None:
            raise ValueError(f"Pair {pair['pair_id']} is missing condition {condition}")
        if metadata["available"] and not metadata["source_representation"]:
            raise ValueError(f"Pair {pair['pair_id']} has an empty representation for {condition}")
        base_condition, budget = parse_matched_condition(condition)
        if metadata["available"] and base_condition is not None and budget is not None:
            observed = whitespace_token_count(str(metadata["source_representation"]))
            if observed != int(metadata["representation_token_count"]):
                raise ValueError(f"Representation token-count mismatch for {pair['pair_id']} / {condition}")
            raw_count = int(metadata.get("raw_history_token_count") or 0)
            if raw_count > budget:
                raise ValueError(f"Raw-history budget exceeded for {pair['pair_id']} / {condition}")
            if metadata.get("raw_history_sha256") != sha256_text(format_raw_history(pair, budget)):
                raise ValueError(f"Raw-history mismatch for {pair['pair_id']} / {condition}")
            if base_condition != "raw_history":
                raw = format_raw_history(pair, budget)
                if not str(metadata["source_representation"]).startswith(raw + "\n\n"):
                    raise ValueError(f"Augmented condition changed raw history for {pair['pair_id']} / {condition}")
            if metadata.get("budget_tokenizer") != REPRESENTATION_BUDGET_TOKENIZER:
                raise ValueError(f"Missing budget-tokenizer audit field for {pair['pair_id']} / {condition}")

    # For each raw-history budget, every available augmentation must share the same raw block
    # and the same realized auxiliary length as the true-assumption condition.
    budgets = sorted(
        {budget for condition in conditions for _, budget in [parse_matched_condition(condition)] if budget is not None}
    )
    for budget in budgets:
        raw_condition = matched_condition_id("raw_history", budget)
        true_condition = matched_condition_id("raw_history_true_assumptions", budget)
        raw_meta = pair["conditions"].get(raw_condition)
        true_meta = pair["conditions"].get(true_condition)
        if not raw_meta or not raw_meta.get("available") or not true_meta or not true_meta.get("available"):
            continue
        expected_hash = raw_meta.get("raw_history_sha256")
        expected_aux = true_meta.get("auxiliary_token_count")
        for base in (
            "raw_history_true_assumptions",
            "raw_history_different_episode_assumptions",
            "raw_history_same_episode_random_turn_assumptions",
            "raw_history_explicit",
        ):
            condition = matched_condition_id(base, budget)
            metadata = pair["conditions"].get(condition)
            if not metadata or not metadata.get("available"):
                continue
            if metadata.get("raw_history_sha256") != expected_hash:
                raise ValueError(f"Raw-history hash differs across conditions for {pair['pair_id']} / b{budget}")
            if metadata.get("auxiliary_token_count") != expected_aux:
                raise ValueError(f"Auxiliary length mismatch for {pair['pair_id']} / {condition}")


def pair_csv_row(pair: dict[str, Any]) -> dict[str, Any]:
    context_explicit = sum(len(turn.get("explicit_texts", [])) for turn in pair.get("context_turns", []))
    context_assumptions = sum(
        len(turn.get("assumption_texts", [])) for turn in pair.get("context_turns", [])
    )
    return {
        "pair_id": pair["pair_id"],
        "category": pair["category"],
        "episode_id": pair["episode_id"],
        "source_path": pair["source_path"],
        "source_turn_id": pair["source_turn_id"],
        "source_turn_idx": pair["source_turn_idx"],
        "future_horizon": pair["future_horizon"],
        "true_future_turn_id": pair["true_next_turn_id"],
        "true_future_turn_idx": pair["true_next_turn_idx"],
        "true_future_turn_move_label": pair["true_next_turn_move_label"],
        "source_word_count": pair["source_word_count"],
        "source_tail_word_count": len(text_words(pair["source_tail_text"])),
        "explicit_count": len(pair["source_explicit_texts"]),
        "assumption_count": len(pair["source_assumption_texts"]),
        "all_assumption_count": len(pair["source_all_assumption_texts"]),
        "history_turn_count": pair["history_turn_count"],
        "context_explicit_count": context_explicit,
        "context_assumption_count": context_assumptions,
        "original_boundary_verified": pair["original_boundary_verified"],
        "source_original_turn_indices_json": json.dumps(pair["source_original_turn_indices"]),
        "true_next_original_turn_indices_json": json.dumps(pair["true_next_original_turn_indices"]),
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
        "Preparing raw-history augmentation experiment: version=%s prompt=%s input=%s categories=%s files=%d seed=%d budgets=%s conditions=%s",
        SCRIPT_VERSION,
        PROMPT_VERSION,
        args.input_dir,
        categories,
        len(category_files),
        args.seed,
        args.representation_budgets,
        args.conditions,
    )
    turns: list[TurnRecord] = []
    pairs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    aggregate_boundary_counts: dict[str, int] = defaultdict(int)
    for category, path in category_files:
        try:
            episode_turns, episode_pairs, boundary_counts = build_episode_records(
                category,
                path,
                args.history_turns,
                args.source_tail_words,
                args.candidate_head_words,
                args.assumption_budget,
                args.future_horizons,
            )
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
            continue
        turns.extend(episode_turns)
        pairs.extend(episode_pairs)
        for key, value in boundary_counts.items():
            aggregate_boundary_counts[key] += value
    if not turns:
        detail = errors[0]["error"] if errors else "no substantive turns"
        raise RuntimeError(f"No usable turns were loaded; first error: {detail}")
    unverified_boundary_count = aggregate_boundary_counts["boundary_unverified_pair_count"]
    if unverified_boundary_count:
        raise RuntimeError(
            f"Preparation found {unverified_boundary_count} pair boundaries without original-turn "
            "provenance. Regenerate data_cleaned with the current deduplicate_data.py before "
            "running this confirmatory experiment."
        )
    indexes = build_turn_indexes(turns)
    aggregate_negative_counts: dict[str, int] = defaultdict(int)
    donors: list[dict[str, Any]] = []
    for pair in pairs:
        pair["reserved_same_episode_random_turn_donor_id"] = reserve_same_episode_random_turn_assumption_donor(pair, turns, args.seed)
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
    pairs.sort(
        key=lambda row: (
            row["category"],
            row["episode_id"],
            row["source_turn_idx"],
            row["future_horizon"],
        )
    )
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
        "experiment": "Experiment 1: Assumption-Augmented Future-Turn Accuracy",
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
        "representation_budgets": args.representation_budgets,
        "representation_budget_tokenizer": REPRESENTATION_BUDGET_TOKENIZER,
        "representation_budget_policy": "budget_caps_raw_history_only; identical_raw_history_reused_across_conditions; auxiliary_content_length_matched_to_true_assumptions",
        "state_deduplication": "raw_history_not_deduplicated; explicit_control_uses_exact_normalized_text_keep_most_recent",
        "future_horizons": args.future_horizons,
        "history_turns": args.history_turns,
        "source_tail_words": args.source_tail_words,
        "candidate_head_words": args.candidate_head_words,
        "assumption_budget": args.assumption_budget,
        "assumption_selection": "candidate_blind_descending_extraction_confidence_current_source_turn; original_order_tie_break; unscored_legacy_items_last",
        "seed": args.seed,
        "candidate_count_target": EXPECTED_CANDIDATE_COUNT,
        "negative_count_target": HARD_NEGATIVE_TARGET_COUNT,
        "comparison_rows_per_pair_condition": EXPECTED_COMPARISONS_PER_CONDITION,
        "episode_file_count": len(category_files),
        "source_episode_count": len({str(pair["source_path"]) for pair in pairs}),
        "turn_count": len(turns),
        "pair_count_before_candidate_filter": len(pairs),
        "pair_count_by_horizon": {
            str(horizon): sum(int(pair["future_horizon"]) == horizon for pair in pairs)
            for horizon in args.future_horizons
        },
        "candidate_complete_pair_count": sum(bool(pair["candidate_pool_complete"]) for pair in pairs),
        "candidate_incomplete_pair_count": sum(not pair["candidate_pool_complete"] for pair in pairs),
        "assumption_eligible_pair_count": sum(bool(pair["source_assumption_texts"]) for pair in pairs),
        "unavailable_controls": {
            condition: sum(
                pair["conditions"].get(condition, {}).get("available") is False for pair in pairs
            )
            for condition in args.conditions
            if condition_donor_key(condition) is not None
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
            for condition in args.conditions
            if condition_donor_key(condition) is not None
        },
        "normalization_error_count": len(errors),
        "normalization_errors": errors,
        "boundary_provenance_counts": dict(aggregate_boundary_counts),
        "negative_sampling_counts": dict(aggregate_negative_counts),
    }
    write_json(prepare_manifest_path(args), manifest)
    logger.info("Prepared %d pairs (%d complete) at %s", len(pairs), manifest["candidate_complete_pair_count"], prepared)
    return manifest


def build_scoring_prompt(
    source_representation: str,
    candidate_a_text: str,
    candidate_b_text: str,
    future_horizon: int,
) -> str:
    target_description = (
        "the immediate next turn" if future_horizon == 1 else f"the turn exactly {future_horizon} turns later"
    )
    return f"""You are a strict conversation-continuation judge.

Your task is to determine which candidate is more likely to be {target_description} in the conversation described by the source representation.

Evaluate Candidate A and Candidate B comparatively. Prioritize:

* semantic relevance to the observed conversation
* conversational obligations created by the preceding turns
* speaker intent, stance, and continuity
* consistency with supported presuppositions and assumptions
* local discourse coherence at the specified future-turn horizon

Do not prefer a candidate merely because it repeats words, entities, or topics from the source representation.

Source representation:
{source_representation}

Candidate A:
{candidate_a_text}

Candidate B:
{candidate_b_text}

Choose exactly one candidate.

Return ONLY a valid JSON object with exactly these two keys:

{{
"answer": "A",
"evidence": "Brief reason why A is more likely than B at the specified future-turn horizon."
}}

Output requirements:

* "answer" must be exactly "A" or "B".
* "evidence" must be a single concise sentence.
* The evidence must compare the candidates based on conversational fit, not summarize or repeat their full content.
* Do not repeat the source representation.
* Do not quote the candidates unless necessary to identify a decisive distinction.
* Do not provide chain-of-thought, step-by-step reasoning, scores, probabilities, explanations outside the JSON, or markdown.
* Do not add any keys.
* Output exactly one JSON object and nothing else.

"""


def build_retry_prompt(
    source_representation: str,
    candidate_a_text: str,
    candidate_b_text: str,
    future_horizon: int,
    previous_output: str,
    parse_error: str | None,
) -> str:
    target_description = (
        "the immediate next turn" if future_horizon == 1 else f"the turn exactly {future_horizon} turns later"
    )
    return f"""You are a strict conversation-continuation judge.

Your previous response did not satisfy the required output format.

Re-evaluate the same source representation and the same two candidates. Determine which
candidate is more likely to be {target_description} in the conversation.

Evaluate Candidate A and Candidate B comparatively. Prioritize:

- semantic relevance to the observed conversation
- conversational obligations created by the preceding turns
- speaker intent, stance, and continuity
- consistency with supported presuppositions and assumptions
- local discourse coherence at the specified future-turn horizon

Do not prefer a candidate merely because it repeats words, entities, or topics from the
source representation.

Source representation:
{source_representation}

Candidate A:
{candidate_a_text}

Candidate B:
{candidate_b_text}

Previous parse error:
{parse_error or "unknown_parse_error"}

Previous invalid response:
{previous_output[:200]}

Choose exactly one candidate.

Return ONLY a valid JSON object with exactly these two keys:

{{
  "answer": "A",
  "evidence": "Brief reason why A is more likely than B at the specified future-turn horizon."
}}

Output requirements:
- "answer" must be exactly "A" or "B".
- "evidence" must be a single concise sentence.
- The evidence must state the decisive conversational distinction between the candidates.
- Do not summarize the source representation.
- Do not repeat or reproduce the previous invalid response.
- Do not quote or restate the candidates unless necessary to identify the decisive distinction.
- Do not provide chain-of-thought, step-by-step reasoning, scores, or probabilities.
- Do not use markdown or code fences.
- Do not add any keys.
- Output exactly one JSON object and nothing else.
"""


def parse_llm_choice(raw_output: str) -> ParsedChoice:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "choice": None,
            "evidence": None,
            "parse_success": False,
            "parse_error": "invalid_json",
            "parse_method": None,
        }
    if not isinstance(payload, dict):
        return {
            "choice": None,
            "evidence": None,
            "parse_success": False,
            "parse_error": "expected_json_object",
            "parse_method": None,
        }
    if set(payload) != {"answer", "evidence"}:
        return {
            "choice": None,
            "evidence": None,
            "parse_success": False,
            "parse_error": "expected_exact_keys_answer_evidence",
            "parse_method": None,
        }
    answer = payload["answer"]
    evidence = payload["evidence"]
    if not isinstance(answer, str) or answer not in {"A", "B"}:
        return {
            "choice": None,
            "evidence": None,
            "parse_success": False,
            "parse_error": "answer_must_be_A_or_B",
            "parse_method": None,
        }
    if not isinstance(evidence, str) or not evidence.strip() or "\n" in evidence:
        return {
            "choice": None,
            "evidence": None,
            "parse_success": False,
            "parse_error": "evidence_must_be_nonempty_single_line_string",
            "parse_method": None,
        }
    return {
        "choice": answer,
        "evidence": evidence.strip(),
        "parse_success": True,
        "parse_error": None,
        "parse_method": "strict_json_answer_evidence",
    }


def normalize_score_choice(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    parsed = parse_llm_choice(str(row.get("raw_output") or ""))
    normalized["choice"] = parsed["choice"]
    normalized["judge_evidence"] = parsed["evidence"]
    normalized["parse_success"] = parsed["parse_success"]
    normalized["parse_error"] = parsed["parse_error"]
    normalized["parse_method"] = parsed["parse_method"]
    if not parsed["parse_success"]:
        normalized["positive_preference"] = None
        return normalized
    required = ("candidate_a_id", "candidate_b_id", "positive_candidate_id")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(
            f"Cannot normalize parsed score row {row.get('pair_id')!r}; missing fields: {missing}"
        )
    chosen_id = row["candidate_a_id"] if parsed["choice"] == "A" else row["candidate_b_id"]
    normalized["positive_preference"] = int(chosen_id == row["positive_candidate_id"])
    return normalized


def task_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["pair_id"]),
        str(row["comparison_id"]),
        str(row["presentation_order"]),
        str(row["condition"]),
        str(row["model_name"]),
        str(row["prompt_version"]),
    )


def score_record_valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("parse_success")
        and row.get("choice") in {"A", "B"}
        and row.get("positive_preference") in {0, 1}
    )


def compact_existing_scores(
    path: Path,
    *,
    model_name: str,
    prompt_version: str,
    overwrite: bool,
    allowed_keys: set[tuple[str, str, str, str, str, str]] | None = None,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str, str, str]]]:
    if not path.exists():
        return [], set()
    observed = read_jsonl(path)
    kept: list[dict[str, Any]] = []
    canonical_by_key: dict[tuple[str, str, str, str, str, str], str] = {}
    completed: set[tuple[str, str, str, str, str, str]] = set()
    changed = False
    identity_fields = {
        "pair_id",
        "comparison_id",
        "presentation_order",
        "condition",
        "model_name",
        "prompt_version",
    }
    for row in observed:
        if not identity_fields.issubset(row):
            changed = True
            continue
        normalized_row = normalize_score_choice(row)
        if canonical_json(normalized_row) != canonical_json(row):
            changed = True
        row = normalized_row
        key = task_key(row)
        if allowed_keys is not None and key not in allowed_keys:
            changed = True
            continue
        selected_config = key[4] == model_name and key[5] == prompt_version
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
            positives = [candidate for candidate in pair["candidates"] if candidate["is_true_next_turn"]]
            negatives = [candidate for candidate in pair["candidates"] if not candidate["is_true_next_turn"]]
            if len(positives) != 1 or len(negatives) != HARD_NEGATIVE_TARGET_COUNT:
                raise ValueError(f"Pair {pair['pair_id']} does not have one positive and 24 negatives")
            positive = positives[0]
            for negative in negatives:
                comparison_digest = sha256_text(
                    f"{pair['pair_id']}:{positive['candidate_id']}:{negative['candidate_id']}"
                )[:16]
                comparison_id = f"{pair['pair_id']}:comparison:{comparison_digest}"
                for presentation_order in ("positive_first", "positive_second"):
                    candidate_a = positive if presentation_order == "positive_first" else negative
                    candidate_b = negative if presentation_order == "positive_first" else positive
                    tasks.append(
                        {
                            "pair": pair,
                            "positive_candidate": positive,
                            "negative_candidate": negative,
                            "candidate_a": candidate_a,
                            "candidate_b": candidate_b,
                            "comparison_id": comparison_id,
                            "presentation_order": presentation_order,
                            "future_horizon": int(pair["future_horizon"]),
                            "condition": condition,
                            "source_representation": metadata["source_representation"],
                        }
                    )
    tasks.sort(
        key=lambda task: (
            task["pair"]["pair_id"],
            task["negative_candidate"]["candidate_order"],
            task["condition"],
            task["presentation_order"],
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
        self.tokenizer = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams
        self.sampling_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        }

    def generate_batch(self, prompts: list[str], *, max_tokens: int) -> list[str]:
        params = self.SamplingParams(max_tokens=max_tokens, **self.sampling_kwargs)
        rendered_prompts: list[str] = []
        for prompt in prompts:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(rendered, str):
                raise TypeError(
                    f"Tokenizer chat template returned {type(rendered).__name__}; expected str"
                )
            rendered_prompts.append(rendered)
        outputs = self.llm.generate(rendered_prompts, params)
        return [output.outputs[0].text.strip() for output in outputs]


def score_row(
    task: dict[str, Any],
    raw_output: str,
    parsed: ParsedChoice,
    args: argparse.Namespace,
    *,
    attempt_outputs: list[str] | None = None,
    attempt_parse_errors: list[str | None] | None = None,
    attempt_token_budgets: list[int] | None = None,
) -> dict[str, Any]:
    pair = task["pair"]
    positive = task["positive_candidate"]
    negative = task["negative_candidate"]
    candidate_a = task["candidate_a"]
    candidate_b = task["candidate_b"]
    condition_metadata = pair["conditions"].get(task["condition"], {})
    donor = pair["donors"].get(condition_donor_key(task["condition"]) or task["condition"], {})
    positive_preference = None
    if parsed["parse_success"]:
        chosen_candidate = candidate_a if parsed["choice"] == "A" else candidate_b
        positive_preference = int(bool(chosen_candidate["is_true_next_turn"]))
    return {
        "pair_id": pair["pair_id"],
        "comparison_id": task["comparison_id"],
        "presentation_order": task["presentation_order"],
        "condition": task["condition"],
        "base_condition": condition_metadata.get("base_condition"),
        "representation_budget": condition_metadata.get("representation_budget"),
        "representation_token_count": condition_metadata.get("representation_token_count"),
        "budget_tokenizer": condition_metadata.get("budget_tokenizer"),
        "future_horizon": int(pair["future_horizon"]),
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "source_turn_id": pair["source_turn_id"],
        "positive_candidate_id": positive["candidate_id"],
        "positive_candidate_turn_id": positive["candidate_turn_id"],
        "positive_candidate_order": positive["candidate_order"],
        "positive_candidate_text": positive["candidate_text"],
        "negative_candidate_id": negative["candidate_id"],
        "negative_candidate_turn_id": negative["candidate_turn_id"],
        "negative_candidate_order": negative["candidate_order"],
        "negative_candidate_text": negative["candidate_text"],
        "negative_source": negative["negative_source"],
        "candidate_a_id": candidate_a["candidate_id"],
        "candidate_b_id": candidate_b["candidate_id"],
        "source_representation": task["source_representation"] if args.save_source_representation else None,
        "source_explicit_json": json.dumps(pair["source_explicit_texts"], ensure_ascii=False),
        "source_assumptions_json": json.dumps(pair["source_assumption_texts"], ensure_ascii=False),
        "source_all_assumptions_json": json.dumps(pair["source_all_assumption_texts"], ensure_ascii=False),
        "donor_turn_id": donor.get("donor_turn_id"),
        "donor_episode_id": donor.get("donor_episode_id"),
        "donor_category": donor.get("donor_category"),
        "donor_fallback_level": donor.get("donor_fallback_level"),
        "candidate_pool_sha256": pair["candidate_pool_sha256"],
        "choice": parsed["choice"],
        "judge_evidence": parsed["evidence"],
        "positive_preference": positive_preference,
        "parse_success": parsed["parse_success"],
        "parse_error": parsed["parse_error"],
        "parse_method": parsed["parse_method"],
        "raw_output": raw_output,
        "judge_attempt_count": len(attempt_outputs or [raw_output]),
        "judge_retry_count": max(0, len(attempt_outputs or [raw_output]) - 1),
        "judge_attempt_outputs_json": json.dumps(attempt_outputs or [raw_output], ensure_ascii=False),
        "judge_attempt_parse_errors_json": json.dumps(attempt_parse_errors or [parsed["parse_error"]], ensure_ascii=False),
        "judge_attempt_token_budgets_json": json.dumps(attempt_token_budgets or [args.max_tokens]),
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
    tasks = build_tasks(selected_pairs, args)
    selected_keys = {
        (
            task["pair"]["pair_id"],
            task["comparison_id"],
            task["presentation_order"],
            task["condition"],
            args.model_name,
            PROMPT_VERSION,
        )
        for task in tasks
    }
    selected_source_paths = sorted({str(pair["source_path"]) for pair in selected_pairs})
    config = {
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "conditions": args.conditions,
        "representation_budgets": args.representation_budgets,
        "representation_budget_tokenizer": REPRESENTATION_BUDGET_TOKENIZER,
        "future_horizons": args.future_horizons,
        "history_turns": args.history_turns,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "max_score_retries": args.max_score_retries,
        "max_retry_tokens": args.max_retry_tokens,
        "scoring_mode": "order_swapped_binary_future_turn_accuracy",
        "source_tail_words": args.source_tail_words,
        "candidate_head_words": args.candidate_head_words,
        "assumption_budget": args.assumption_budget,
        "seed": args.seed,
        "strict_all_conditions": args.strict_all_conditions,
        "dry_run": args.dry_run,
    }
    config_sha256 = stable_hash(config)
    output = patch_dir(args.output_dir, args.patch_index, args.num_patches)
    output.mkdir(parents=True, exist_ok=True)
    scores_file = score_path(output)
    patch_manifest_path = output / "patch_manifest.json"
    if patch_manifest_path.exists():
        previous_manifest = json.loads(patch_manifest_path.read_text(encoding="utf-8"))
        previous_identity = {
            "prepared_pairs_sha256": previous_manifest.get("prepared_pairs_sha256"),
            "config_sha256": previous_manifest.get("config_sha256"),
            "patch_index": previous_manifest.get("patch_index"),
            "num_patches": previous_manifest.get("num_patches"),
            "episodes_per_patch": previous_manifest.get("episodes_per_patch"),
            "selected_source_paths": previous_manifest.get("selected_source_paths"),
        }
        current_identity = {
            "prepared_pairs_sha256": prepare_manifest["prepared_pairs_sha256"],
            "config_sha256": config_sha256,
            "patch_index": args.patch_index,
            "num_patches": args.num_patches,
            "episodes_per_patch": args.episodes_per_patch,
            "selected_source_paths": selected_source_paths,
        }
        if canonical_json(previous_identity) != canonical_json(current_identity):
            logger.warning(
                "Patch identity changed for %s; discarding stale score rows before scoring",
                output,
            )
            write_jsonl(scores_file, [])
    existing, completed = compact_existing_scores(
        scores_file,
        model_name=args.model_name,
        prompt_version=PROMPT_VERSION,
        overwrite=args.overwrite_scores,
        allowed_keys=selected_keys,
    )
    pending = []
    for task in tasks:
        key = (
            task["pair"]["pair_id"],
            task["comparison_id"],
            task["presentation_order"],
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
    in_progress_manifest = {
        "stage": "score_patch",
        "complete": False,
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "patch_index": args.patch_index,
        "num_patches": args.num_patches,
        "episodes_per_patch": args.episodes_per_patch,
        "prepared_pairs_sha256": prepare_manifest["prepared_pairs_sha256"],
        "config": config,
        "config_sha256": config_sha256,
        "selected_source_paths": selected_source_paths,
        "selected_pair_count": len(selected_pairs),
        "expected_task_count": len(tasks),
    }
    write_json(patch_manifest_path, in_progress_manifest)
    llm = None if args.dry_run or not pending else LLMInterface(args)
    attempted = 0
    generation_attempts = 0
    retry_generation_count = 0
    parse_failures_before_retry = 0
    parse_failures = 0
    for start in range(0, len(pending), args.prompt_batch_size):
        batch = pending[start:start + args.prompt_batch_size]
        if args.dry_run:
            raw_outputs = []
            for task in batch:
                choice = "A" if task["candidate_a"]["is_true_next_turn"] else "B"
                other_choice = "B" if choice == "A" else "A"
                raw_outputs.append(
                    json.dumps(
                        {
                            "answer": choice,
                            "evidence": (
                                f"Candidate {choice} fits the specified future-turn horizon "
                                f"better than Candidate {other_choice}."
                            ),
                        }
                    )
                )
        else:
            assert llm is not None
            prompts = [
                build_scoring_prompt(
                    task["source_representation"],
                    task["candidate_a"]["candidate_text"],
                    task["candidate_b"]["candidate_text"],
                    task["future_horizon"],
                )
                for task in batch
            ]
            raw_outputs = llm.generate_batch(prompts, max_tokens=args.max_tokens)
        if len(raw_outputs) != len(batch):
            raise RuntimeError("Model returned a different number of outputs than prompts")

        generation_attempts += len(raw_outputs)
        attempt_outputs: list[list[str]] = [[raw_output] for raw_output in raw_outputs]
        attempt_token_budgets: list[list[int]] = [[args.max_tokens] for _ in raw_outputs]
        parsed_outputs = [parse_llm_choice(raw_output) for raw_output in raw_outputs]
        attempt_parse_errors: list[list[str | None]] = [[parsed["parse_error"]] for parsed in parsed_outputs]
        parse_failures_before_retry += sum(not parsed["parse_success"] for parsed in parsed_outputs)

        for retry_index in range(args.max_score_retries):
            failed_indices = [index for index, parsed in enumerate(parsed_outputs) if not parsed["parse_success"]]
            if not failed_indices or args.dry_run:
                break
            assert llm is not None
            retry_tokens = min(args.max_retry_tokens, args.max_tokens * (2 ** (retry_index + 1)))
            retry_prompts = [
                build_retry_prompt(
                    batch[index]["source_representation"],
                    batch[index]["candidate_a"]["candidate_text"],
                    batch[index]["candidate_b"]["candidate_text"],
                    batch[index]["future_horizon"],
                    attempt_outputs[index][-1],
                    parsed_outputs[index]["parse_error"],
                )
                for index in failed_indices
            ]
            logger.info(
                "Retrying %d malformed judge outputs with max_tokens=%d (retry %d/%d)",
                len(failed_indices),
                retry_tokens,
                retry_index + 1,
                args.max_score_retries,
            )
            retry_outputs = llm.generate_batch(retry_prompts, max_tokens=retry_tokens)
            if len(retry_outputs) != len(failed_indices):
                raise RuntimeError("Model returned a different number of retry outputs than retry prompts")
            generation_attempts += len(retry_outputs)
            retry_generation_count += len(retry_outputs)
            for index, retry_output in zip(failed_indices, retry_outputs):
                attempt_outputs[index].append(retry_output)
                attempt_token_budgets[index].append(retry_tokens)
                raw_outputs[index] = retry_output
                parsed_outputs[index] = parse_llm_choice(retry_output)
                attempt_parse_errors[index].append(parsed_outputs[index]["parse_error"])

        rows: list[dict[str, Any]] = []
        for task, raw_output, parsed, outputs_for_task, errors_for_task, budgets_for_task in zip(
            batch,
            raw_outputs,
            parsed_outputs,
            attempt_outputs,
            attempt_parse_errors,
            attempt_token_budgets,
        ):
            parse_failures += int(not parsed["parse_success"])
            rows.append(
                score_row(
                    task,
                    raw_output,
                    parsed,
                    args,
                    attempt_outputs=outputs_for_task,
                    attempt_parse_errors=errors_for_task,
                    attempt_token_budgets=budgets_for_task,
                )
            )
        append_jsonl(scores_file, rows)
        attempted += len(rows)
    if not scores_file.exists():
        write_jsonl(scores_file, [])
    all_rows = read_jsonl(scores_file) if scores_file.exists() else existing
    valid_count = sum(score_record_valid(row) and task_key(row) in selected_keys for row in all_rows)
    manifest = {
        "stage": "score_patch",
        "complete": valid_count == len(tasks),
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "patch_index": args.patch_index,
        "num_patches": args.num_patches,
        "episodes_per_patch": args.episodes_per_patch,
        "prepared_pairs_sha256": prepare_manifest["prepared_pairs_sha256"],
        "config": config,
        "config_sha256": config_sha256,
        "selected_source_paths": selected_source_paths,
        "selected_pair_count": len(selected_pairs),
        "expected_task_count": len(tasks),
        "valid_task_count": valid_count,
        "attempted_this_run": attempted,
        "generation_attempts_this_run": generation_attempts,
        "retry_generation_count_this_run": retry_generation_count,
        "parse_failures_before_retry_this_run": parse_failures_before_retry,
        "parse_failures_this_run": parse_failures,
        "scores_sha256": file_hash(scores_file) if scores_file.exists() else None,
    }
    write_json(patch_manifest_path, manifest)
    if valid_count != len(tasks):
        raise RuntimeError(
            f"Patch {args.patch_index} has {len(tasks) - valid_count} unresolved forced-choice "
            "rows after retries. Valid rows were checkpointed; rerun this patch to retry only "
            "the unresolved tasks."
        )
    return manifest


def aggregate_pairwise_condition(rows: list[dict[str, Any]]) -> dict[str, float | int] | None:
    valid = [row for row in rows if score_record_valid(row)]
    if len(valid) != EXPECTED_COMPARISONS_PER_CONDITION:
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[str(row["comparison_id"])].append(row)
    if len(grouped) != HARD_NEGATIVE_TARGET_COUNT:
        return None
    comparison_preferences: list[float] = []
    decision_correctness: list[int] = []
    order_consistent_count = 0
    for comparison_id, comparison_rows in grouped.items():
        orders = {str(row["presentation_order"]) for row in comparison_rows}
        if len(comparison_rows) != 2 or orders != {"positive_first", "positive_second"}:
            return None
        positive_orders = {int(row["positive_candidate_order"]) for row in comparison_rows}
        negative_orders = {int(row["negative_candidate_order"]) for row in comparison_rows}
        if len(positive_orders) != 1 or len(negative_orders) != 1:
            raise ValueError(f"Inconsistent candidate order metadata for comparison {comparison_id}")
        preferences = [int(row["positive_preference"]) for row in comparison_rows]
        preference = float(np.mean(preferences))
        comparison_preferences.append(preference)
        decision_correctness.extend(preferences)
        order_consistent_count += int(preferences[0] == preferences[1])
    return {
        "accuracy": float(np.mean(decision_correctness)),
        "order_consistency_rate": order_consistent_count / HARD_NEGATIVE_TARGET_COUNT,
        "complete_comparison_count": len(comparison_preferences),
    }


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
                "future_horizon": int(pair["future_horizon"]),
                "true_future_turn_idx": pair["true_next_turn_idx"],
                "true_future_turn_move_label": pair["true_next_turn_move_label"],
                "condition": condition,
                "base_condition": metadata.get("base_condition"),
                "representation_budget": metadata.get("representation_budget"),
                "representation_token_count": metadata.get("representation_token_count"),
                "budget_tokenizer": metadata.get("budget_tokenizer"),
                "explicit_count": len(pair["source_explicit_texts"]),
                "assumption_count": len(pair["source_assumption_texts"]),
                "context_explicit_count": sum(
                    len(turn.get("explicit_texts", [])) for turn in pair.get("context_turns", [])
                ),
                "context_assumption_count": sum(
                    len(turn.get("assumption_texts", [])) for turn in pair.get("context_turns", [])
                ),
                "all_assumption_count": len(pair["source_all_assumption_texts"]),
                "source_word_count": pair["source_word_count"],
                "original_boundary_verified": pair["original_boundary_verified"],
                "analysis_subset_flags": "[]",
                "accuracy": None,
                "order_consistency_rate": None,
                "candidate_count": len(pair["candidates"]),
                "expected_comparison_row_count": EXPECTED_COMPARISONS_PER_CONDITION,
                "parsed_comparison_row_count": len(parsed),
                "complete_comparison_count": 0,
                "condition_available": bool(metadata.get("available")),
                "control_unavailable_reason": metadata.get("control_unavailable_reason"),
                "candidate_pool_complete": bool(pair["candidate_pool_complete"]),
                "full_retained": False,
                "assumption_eligible": False,
                "sparse_explicit": False,
                "dense_explicit": False,
                "complete_case": False,
            }
            if (
                pair["candidate_pool_complete"]
                and metric["condition_available"]
                and len(parsed) == EXPECTED_COMPARISONS_PER_CONDITION
            ):
                aggregated = aggregate_pairwise_condition(parsed)
                if aggregated is not None:
                    metric.update(aggregated)
                    metric["full_retained"] = True
                    metric["assumption_eligible"] = bool(
                        any(turn.get("assumption_texts") for turn in pair.get("context_turns", []))
                    )
                    metric["sparse_explicit"] = bool(
                        pair["source_assumption_texts"] and len(pair["source_explicit_texts"]) <= 4
                    )
                    metric["dense_explicit"] = bool(
                        pair["source_assumption_texts"] and len(pair["source_explicit_texts"]) >= 5
                    )
            long_rows.append(metric)
    long_df = pd.DataFrame(long_rows)
    long_df["complete_case"] = False
    selected_rows = long_df[long_df["condition"].isin(args.conditions)]
    for (pair_id, budget), group in selected_rows.groupby(
        ["pair_id", "representation_budget"], sort=False, dropna=False
    ):
        if pd.isna(budget):
            expected = [condition for condition in args.conditions if parse_matched_condition(condition)[1] is None]
            mask = (long_df["pair_id"] == pair_id) & long_df["representation_budget"].isna()
        else:
            expected = [
                condition
                for condition in args.conditions
                if parse_matched_condition(condition)[1] == int(budget)
            ]
            mask = (long_df["pair_id"] == pair_id) & (long_df["representation_budget"] == budget)
        complete = bool(
            expected
            and len(group) == len(expected)
            and set(group["condition"]) == set(expected)
            and group["full_retained"].all()
        )
        long_df.loc[mask, "complete_case"] = complete
    for index, row in long_df.iterrows():
        flags = []
        if bool(row["full_retained"]):
            flags.append("full")
        if bool(row["assumption_eligible"]):
            flags.append("assumption_eligible")
        if bool(row["sparse_explicit"]):
            flags.append("sparse_explicit")
        if bool(row["dense_explicit"]):
            flags.append("dense_explicit")
        if bool(row["complete_case"]):
            flags.append("complete_case")
        long_df.at[index, "analysis_subset_flags"] = json.dumps(flags)
    metadata_columns = [
        "pair_id", "category", "episode_id", "source_turn_idx", "future_horizon",
        "true_future_turn_idx", "true_future_turn_move_label", "explicit_count", "assumption_count",
        "all_assumption_count", "context_explicit_count", "context_assumption_count",
        "source_word_count", "original_boundary_verified",
    ]
    wide_parts = [long_df[metadata_columns].drop_duplicates("pair_id").set_index("pair_id")]
    wide_value_columns = (
        "accuracy", "order_consistency_rate", "parsed_comparison_row_count", "complete_comparison_count",
        "condition_available", "control_unavailable_reason", "full_retained",
        "assumption_eligible", "sparse_explicit", "dense_explicit", "complete_case",
        "representation_budget", "representation_token_count", "budget_tokenizer",
    )
    for condition in args.conditions:
        part = long_df[long_df["condition"] == condition].set_index("pair_id")
        selected = part[list(wide_value_columns)].rename(
            columns={column: f"{condition}__{column}" for column in wide_value_columns}
        )
        wide_parts.append(selected)
    return long_df, pd.concat(wide_parts, axis=1).reset_index()


def condition_summary(
    long_df: pd.DataFrame,
    group_column: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "full": "full_retained",
        "assumption_eligible": "assumption_eligible",
        "sparse_explicit": "sparse_explicit",
        "dense_explicit": "dense_explicit",
        "complete_case": "complete_case",
    }
    metrics = ("accuracy", "order_consistency_rate")
    for subset, flag in subsets.items():
        eligible = long_df[long_df[flag] == True].copy()
        if subset == "complete_case":
            eligible = eligible[eligible["full_retained"] == True]
        if eligible.empty:
            continue
        for (condition, future_horizon, group_value), group in eligible.groupby(
            ["condition", "future_horizon", group_column], sort=False, dropna=False
        ):
            row: dict[str, Any] = {
                "analysis_subset": subset,
                "condition": condition,
                "base_condition": parse_matched_condition(str(condition))[0],
                "representation_budget": parse_matched_condition(str(condition))[1],
                "future_horizon": int(future_horizon),
                group_column: group_value,
                "pair_count": len(group),
            }
            for metric in metrics:
                result = cluster_bootstrap(
                    group,
                    metric,
                    seed_label=(
                        f"{subset}:h{int(future_horizon)}:{condition}:"
                        f"{group_column}:{group_value}:{metric}"
                    ),
                    seed=args.seed,
                    draws=args.bootstrap_draws,
                )
                prefix = {
                    "accuracy": "accuracy",
                    "order_consistency_rate": "mean_order_consistency_rate",
                }[metric]
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
        "sparse_explicit": "sparse_explicit",
        "dense_explicit": "dense_explicit",
        "complete_case": "complete_case",
    }
    metric_names = {
        "accuracy": "accuracy",
        "order_consistency_rate": "mean_order_consistency_rate",
    }
    for subset, flag in subsets.items():
        for future_horizon in args.future_horizons:
            for condition in args.conditions:
                group = long_df[
                    (long_df["condition"] == condition)
                    & (long_df["future_horizon"] == future_horizon)
                    & (long_df[flag] == True)
                ].copy()
                if subset == "complete_case":
                    group = group[group["full_retained"] == True]
                row: dict[str, Any] = {
                    "analysis_subset": subset,
                    "condition": condition,
                    "base_condition": parse_matched_condition(str(condition))[0],
                    "representation_budget": parse_matched_condition(str(condition))[1],
                    "future_horizon": future_horizon,
                    "pair_count": int(len(group)),
                }
                for metric, output_name in metric_names.items():
                    result = cluster_bootstrap(
                        group,
                        metric,
                        seed_label=f"overall:{subset}:h{future_horizon}:{condition}:{metric}",
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
    budgets = sorted(
        {
            budget
            for condition in conditions
            for _, budget in [parse_matched_condition(condition)]
            if budget is not None
        }
    )
    for budget in budgets:
        raw = matched_condition_id("raw_history", budget)
        true = matched_condition_id("raw_history_true_assumptions", budget)
        different_episode = matched_condition_id("raw_history_different_episode_assumptions", budget)
        same_episode_random_turn = matched_condition_id("raw_history_same_episode_random_turn_assumptions", budget)
        explicit = matched_condition_id("raw_history_explicit", budget)
        for contrast in (
            (true, different_episode),
            (true, same_episode_random_turn),
            (true, raw),
            (true, explicit),
            (explicit, raw),
            (different_episode, raw),
            (same_episode_random_turn, raw),
        ):
            if contrast[0] in conditions and contrast[1] in conditions:
                rows.append(contrast)
    for contrast in REQUIRED_CONTRASTS:
        if contrast[0] in conditions and contrast[1] in conditions and contrast not in rows:
            rows.append(contrast)
    return rows


def build_pairwise(long_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, baseline in contrast_list(args.conditions):
        for future_horizon in args.future_horizons:
            target_df = long_df[
                (long_df["condition"] == target)
                & (long_df["future_horizon"] == future_horizon)
            ].set_index("pair_id")
            baseline_df = long_df[
                (long_df["condition"] == baseline)
                & (long_df["future_horizon"] == future_horizon)
            ].set_index("pair_id")
            common_ids = target_df.index.intersection(baseline_df.index)
            for subset in (
                "full",
                "assumption_eligible",
                "sparse_explicit",
                "dense_explicit",
                "complete_case",
            ):
                subset_ids = []
                for pair_id in common_ids:
                    target_row = target_df.loc[pair_id]
                    baseline_row = baseline_df.loc[pair_id]
                    if subset == "full":
                        keep = bool(target_row["full_retained"] and baseline_row["full_retained"])
                    elif subset == "assumption_eligible":
                        keep = bool(target_row["assumption_eligible"] and baseline_row["assumption_eligible"])
                    elif subset == "sparse_explicit":
                        keep = bool(target_row["sparse_explicit"] and baseline_row["sparse_explicit"])
                    elif subset == "dense_explicit":
                        keep = bool(target_row["dense_explicit"] and baseline_row["dense_explicit"])
                    else:
                        keep = bool(
                            target_row["complete_case"]
                            and baseline_row["complete_case"]
                            and target_row["full_retained"]
                            and baseline_row["full_retained"]
                        )
                    if keep:
                        subset_ids.append(pair_id)
                delta_rows = []
                for pair_id in subset_ids:
                    target_row = target_df.loc[pair_id]
                    baseline_row = baseline_df.loc[pair_id]
                    delta_rows.append(
                        {
                            "pair_id": pair_id,
                            "category": target_row["category"],
                            "episode_id": target_row["episode_id"],
                            "delta": float(target_row["accuracy"]) - float(baseline_row["accuracy"]),
                        }
                    )
                delta_df = pd.DataFrame(delta_rows, columns=["pair_id", "category", "episode_id", "delta"])
                result = cluster_bootstrap(
                    delta_df,
                    "delta",
                    seed_label=f"paired:{subset}:h{future_horizon}:{target}:{baseline}:accuracy",
                    seed=args.seed,
                    draws=args.bootstrap_draws,
                )
                values = delta_df["delta"] if not delta_df.empty else pd.Series(dtype=float)
                rows.append(
                    {
                        "analysis_subset": subset,
                        "future_horizon": future_horizon,
                        "target_condition": target,
                        "baseline_condition": baseline,
                        "target_base_condition": parse_matched_condition(target)[0],
                        "baseline_base_condition": parse_matched_condition(baseline)[0],
                        "representation_budget": (
                            parse_matched_condition(target)[1]
                            if parse_matched_condition(target)[1] == parse_matched_condition(baseline)[1]
                            else None
                        ),
                        "metric": "accuracy",
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



def build_flip_analysis(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Candidate-level paired flip analysis after averaging the two A/B presentation orders."""
    pair_lookup = {str(pair["pair_id"]): pair for pair in pairs}
    selected = [
        row
        for row in scores
        if row.get("model_name") == args.model_name
        and row.get("prompt_version") == PROMPT_VERSION
        and score_record_valid(row)
        and row.get("condition") in args.conditions
    ]
    if not selected:
        return pd.DataFrame()
    frame = pd.DataFrame(selected)
    grouped = (
        frame.groupby(["pair_id", "comparison_id", "condition"], sort=False)
        .agg(
            positive_preference=("positive_preference", "mean"),
            presentation_count=("presentation_order", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["presentation_count"] == 2].copy()
    rows: list[dict[str, Any]] = []
    for target, baseline in contrast_list(args.conditions):
        target_base, target_budget = parse_matched_condition(target)
        baseline_base, baseline_budget = parse_matched_condition(baseline)
        if target_base != "raw_history_true_assumptions" or target_budget != baseline_budget:
            continue
        target_df = grouped[grouped["condition"] == target].set_index(["pair_id", "comparison_id"])
        baseline_df = grouped[grouped["condition"] == baseline].set_index(["pair_id", "comparison_id"])
        common = target_df.index.intersection(baseline_df.index)
        if common.empty:
            continue
        details: list[dict[str, Any]] = []
        for pair_id, comparison_id in common:
            pair = pair_lookup[str(pair_id)]
            target_pref = float(target_df.loc[(pair_id, comparison_id), "positive_preference"])
            baseline_pref = float(baseline_df.loc[(pair_id, comparison_id), "positive_preference"])
            details.append(
                {
                    "pair_id": str(pair_id),
                    "comparison_id": str(comparison_id),
                    "category": pair["category"],
                    "episode_id": pair["episode_id"],
                    "future_horizon": int(pair["future_horizon"]),
                    "delta": target_pref - baseline_pref,
                    "helpful_flip": int(baseline_pref == 0.0 and target_pref == 1.0),
                    "harmful_flip": int(baseline_pref == 1.0 and target_pref == 0.0),
                    "partial_improvement": int(target_pref > baseline_pref),
                    "partial_harm": int(target_pref < baseline_pref),
                }
            )
        detail_df = pd.DataFrame(details)
        for horizon in args.future_horizons:
            group = detail_df[detail_df["future_horizon"] == horizon].copy()
            if group.empty:
                continue
            boot = cluster_bootstrap(
                group,
                "delta",
                seed_label=f"flip:h{horizon}:{target}:{baseline}",
                seed=args.seed,
                draws=args.bootstrap_draws,
            )
            n = len(group)
            helpful = int(group["helpful_flip"].sum())
            harmful = int(group["harmful_flip"].sum())
            rows.append(
                {
                    "future_horizon": int(horizon),
                    "representation_budget": target_budget,
                    "target_condition": target,
                    "baseline_condition": baseline,
                    "candidate_comparison_count": n,
                    "helpful_flip_count": helpful,
                    "harmful_flip_count": harmful,
                    "helpful_flip_rate": helpful / n,
                    "harmful_flip_rate": harmful / n,
                    "net_strict_flip_rate": (helpful - harmful) / n,
                    "partial_improvement_count": int(group["partial_improvement"].sum()),
                    "partial_harm_count": int(group["partial_harm"].sum()),
                    "mean_order_averaged_preference_delta": boot["mean"],
                    "delta_ci95_low": boot["ci95_low"],
                    "delta_ci95_high": boot["ci95_high"],
                    "delta_ci_unstable": boot["ci_unstable"],
                    "cluster_count": boot["cluster_count"],
                }
            )
    return pd.DataFrame(rows)

def coverage_summary(long_df: pd.DataFrame, pairs: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for future_horizon in args.future_horizons:
        total = sum(int(pair["future_horizon"]) == future_horizon for pair in pairs)
        for condition in args.conditions:
            group = long_df[
                (long_df["condition"] == condition)
                & (long_df["future_horizon"] == future_horizon)
            ]
            parse_eligible = (group["condition_available"] == True) & (group["candidate_pool_complete"] == True)
            parse_failures = parse_eligible & (group["full_retained"] == False)
            parse_eligible_count = int(parse_eligible.sum())
            rows.append(
                {
                    "future_horizon": future_horizon,
                    "condition": condition,
                    "base_condition": parse_matched_condition(condition)[0],
                    "representation_budget": parse_matched_condition(condition)[1],
                    "pair_count": total,
                    "candidate_complete_pair_count": int(group["candidate_pool_complete"].sum()),
                    "condition_available_pair_count": int(group["condition_available"].sum()),
                    "fully_parsed_pair_count": int(group["full_retained"].sum()),
                    "assumption_eligible_pair_count": int(group["assumption_eligible"].sum()),
                    "sparse_explicit_pair_count": int(group["sparse_explicit"].sum()),
                    "dense_explicit_pair_count": int(group["dense_explicit"].sum()),
                    "complete_case_pair_count": int(group["complete_case"].sum()),
                    "retained_pair_rate": float(group["full_retained"].mean()) if total else None,
                    "comparison_parse_eligible_pair_count": parse_eligible_count,
                    "comparison_parse_failure_pair_count": int(parse_failures.sum()),
                    "comparison_parse_failure_rate": (
                        float(parse_failures.sum() / parse_eligible_count) if parse_eligible_count else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_decomposition_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    questions: dict[tuple[str, str], str] = {}
    if not pairwise.empty and "representation_budget" in pairwise:
        budgets = sorted(int(value) for value in pairwise["representation_budget"].dropna().unique().tolist())
        for budget in budgets:
            true = matched_condition_id("raw_history_true_assumptions", budget)
            questions.update({
                (true, matched_condition_id("raw_history_different_episode_assumptions", budget)):
                    "true_assumptions_vs_different_episode_control",
                (true, matched_condition_id("raw_history_same_episode_random_turn_assumptions", budget)):
                    "true_assumptions_vs_same_episode_random_turn_control",
                (true, matched_condition_id("raw_history", budget)):
                    "incremental_assumption_value_over_raw_history",
                (true, matched_condition_id("raw_history_explicit", budget)):
                    "implicit_vs_length_matched_explicit_augmentation",
            })
    questions.update({
        ("explicit_plus_top3_assumptions", "explicit_only"): "incremental_implicit_value_after_abstraction",
        ("explicit_plus_top3_assumptions", "explicit_plus_different_episode_assumptions"): "true_assumptions_vs_different_episode_control_legacy",
        ("explicit_plus_top3_assumptions", "explicit_plus_same_episode_random_turn_assumptions"): "true_assumptions_vs_same_episode_random_turn_control_legacy",
        ("raw_turn_plus_assumptions", "raw_turn"): "incremental_implicit_value_with_lexical_context_legacy",
    })
    if pairwise.empty:
        return pd.DataFrame(columns=[*pairwise.columns, "diagnostic_question", "contrast"])
    selected = pairwise[pairwise.apply(
        lambda row: (str(row["target_condition"]), str(row["baseline_condition"])) in questions, axis=1
    )].copy()
    selected["diagnostic_question"] = selected.apply(
        lambda row: questions[(str(row["target_condition"]), str(row["baseline_condition"]))], axis=1
    )
    selected["contrast"] = selected["target_condition"].astype(str) + " - " + selected["baseline_condition"].astype(str)
    order = {contrast: index for index, contrast in enumerate(questions)}
    selected["_contrast_order"] = selected.apply(
        lambda row: order[(str(row["target_condition"]), str(row["baseline_condition"]))], axis=1
    )
    subset_order = {"assumption_eligible": 0, "complete_case": 1, "full": 2, "sparse_explicit": 3, "dense_explicit": 4}
    selected["_subset_order"] = selected["analysis_subset"].map(subset_order).fillna(99)
    return selected.sort_values(
        ["_contrast_order", "future_horizon", "_subset_order"], kind="stable"
    ).drop(columns=["_contrast_order", "_subset_order"])


def build_audit_sample(
    pairs: list[dict[str, Any]],
    long_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    columns = [
        "audit_outcome", "audit_priority", "pair_id", "category", "episode_id",
        "future_horizon", "true_future_turn_move_label", "explicit_count", "assumption_count",
        "representation_budget", "target_condition", "baseline_condition",
        "accuracy_delta", "baseline_accuracy", "target_accuracy",
        "original_boundary_verified", "source_turn_text", "source_tail_text",
        "source_explicit_json", "source_assumptions_json", "source_all_assumptions_json",
        "history_turns_json", "raw_history_representation", "structured_state_representation",
        "true_future_turn_text",
    ]
    available_budgets = sorted(
        budget
        for condition in args.conditions
        for _, budget in [parse_matched_condition(condition)]
        if budget is not None
    )
    if available_budgets:
        budget = 256 if 256 in available_budgets else available_budgets[len(available_budgets) // 2]
        baseline_condition = matched_condition_id("raw_history_different_episode_assumptions", budget)
        target_condition = matched_condition_id("raw_history_true_assumptions", budget)
    else:
        budget = None
        baseline_condition = "explicit_only"
        target_condition = "explicit_plus_top3_assumptions"
    if not {baseline_condition, target_condition}.issubset(set(args.conditions)):
        return pd.DataFrame(columns=columns)
    lookup = {pair["pair_id"]: pair for pair in pairs}
    baseline_rows = long_df[long_df["condition"] == baseline_condition].set_index("pair_id")
    target_rows = long_df[long_df["condition"] == target_condition].set_index("pair_id")
    rows: list[dict[str, Any]] = []
    for pair_id in baseline_rows.index.intersection(target_rows.index):
        baseline_row = baseline_rows.loc[pair_id]
        target_row = target_rows.loc[pair_id]
        if not bool(baseline_row["assumption_eligible"] and target_row["assumption_eligible"]):
            continue
        pair = lookup[str(pair_id)]
        accuracy_delta = float(target_row["accuracy"]) - float(baseline_row["accuracy"])
        outcome = "win" if accuracy_delta > 0 else "loss" if accuracy_delta < 0 else "tie"
        rows.append(
            {
                "audit_outcome": outcome,
                "pair_id": pair_id,
                "category": pair["category"],
                "episode_id": pair["episode_id"],
                "future_horizon": int(pair["future_horizon"]),
                "true_future_turn_move_label": pair["true_next_turn_move_label"],
                "explicit_count": len(pair["source_explicit_texts"]),
                "assumption_count": len(pair["source_assumption_texts"]),
                "representation_budget": budget,
                "target_condition": target_condition,
                "baseline_condition": baseline_condition,
                "accuracy_delta": accuracy_delta,
                "baseline_accuracy": float(baseline_row["accuracy"]),
                "target_accuracy": float(target_row["accuracy"]),
                "original_boundary_verified": pair["original_boundary_verified"],
                "source_turn_text": pair["source_turn_text"],
                "source_tail_text": pair["source_tail_text"],
                "source_explicit_json": json.dumps(pair["source_explicit_texts"], ensure_ascii=False),
                "source_assumptions_json": json.dumps(pair["source_assumption_texts"], ensure_ascii=False),
                "source_all_assumptions_json": json.dumps(
                    pair["source_all_assumption_texts"], ensure_ascii=False
                ),
                "history_turns_json": json.dumps(pair["history_turn_texts"], ensure_ascii=False),
                "raw_history_representation": pair["conditions"][baseline_condition]["source_representation"],
                "structured_state_representation": pair["conditions"][target_condition]["source_representation"],
                "true_future_turn_text": pair["true_next_turn_text"],
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
            group = group.sort_values(["accuracy_delta", "pair_id"], ascending=[False, True])
        elif outcome == "loss":
            group = group.sort_values(["accuracy_delta", "pair_id"], ascending=[True, True])
        else:
            group = group.sort_values(["_tie_order"], ascending=[True])
        group = group.head(args.audit_sample_size_per_outcome).copy()
        group["audit_priority"] = np.arange(1, len(group) + 1)
        samples.append(group)
    result = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=columns)
    return result.reindex(columns=columns)


def legacy_diagnostic_gate(
    pairwise: pd.DataFrame,
    long_df: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    primary_horizon = 1

    def contrast_row(target: str, baseline: str, subset: str) -> dict[str, Any] | None:
        match = pairwise[
            (pairwise["analysis_subset"] == subset)
            & (pairwise["future_horizon"] == primary_horizon)
            & (pairwise["target_condition"] == target)
            & (pairwise["baseline_condition"] == baseline)
            & (pairwise["metric"] == "accuracy")
        ]
        return None if match.empty else match.iloc[0].to_dict()

    def category_deltas(target: str, baseline: str, subset_flag: str) -> dict[str, float]:
        target_rows = long_df[long_df["condition"] == target].set_index("pair_id")
        baseline_rows = long_df[long_df["condition"] == baseline].set_index("pair_id")
        values: list[dict[str, Any]] = []
        for pair_id in target_rows.index.intersection(baseline_rows.index):
            target_row = target_rows.loc[pair_id]
            baseline_row = baseline_rows.loc[pair_id]
            if (
                int(target_row["future_horizon"]) != primary_horizon
                or not bool(target_row[subset_flag] and baseline_row[subset_flag])
            ):
                continue
            values.append(
                {
                    "category": str(target_row["category"]),
                    "delta": float(target_row["accuracy"]) - float(baseline_row["accuracy"]),
                }
            )
        if not values:
            return {}
        frame = pd.DataFrame(values)
        return {str(key): float(value) for key, value in frame.groupby("category")["delta"].mean().items()}

    primary = contrast_row("explicit_plus_top3_assumptions", "explicit_only", "sparse_explicit")
    exploratory_primary = contrast_row(
        "explicit_plus_top3_assumptions",
        "explicit_only",
        "assumption_eligible",
    )
    different_episode_control = contrast_row(
        "explicit_plus_top3_assumptions",
        "explicit_plus_different_episode_assumptions",
        "sparse_explicit",
    )
    same_episode_random_turn_control = contrast_row(
        "explicit_plus_top3_assumptions",
        "explicit_plus_same_episode_random_turn_assumptions",
        "sparse_explicit",
    )
    raw_increment = contrast_row("raw_turn_plus_assumptions", "raw_turn", "assumption_eligible")
    primary_categories = category_deltas(
        "explicit_plus_top3_assumptions",
        "explicit_only",
        "sparse_explicit",
    )
    raw_categories = category_deltas(
        "raw_turn_plus_assumptions",
        "raw_turn",
        "assumption_eligible",
    )

    def supported(row: dict[str, Any] | None) -> bool:
        return bool(row and row.get("ci95_low") is not None and float(row["ci95_low"]) > 0.0)

    minimum_retained = float(coverage["retained_pair_rate"].min()) if not coverage.empty else 0.0
    primary_supported = supported(primary)
    different_episode_supported = supported(different_episode_control)
    same_episode_random_turn_supported = supported(same_episode_random_turn_control)
    raw_supported = supported(raw_increment)
    positive_primary_categories = sum(value > 0 for value in primary_categories.values())
    positive_raw_categories = sum(value > 0 for value in raw_categories.values())
    category_breadth = positive_primary_categories >= 2
    coverage_acceptable = minimum_retained >= 0.98
    specificity_supported = different_episode_supported and same_episode_random_turn_supported
    if primary_supported and specificity_supported and raw_supported:
        interpretation = "assumptions_add_signal_beyond_raw_lexical_context"
    elif primary_supported and specificity_supported:
        interpretation = "assumptions_help_sparse_explicit_representations_with_control_specificity"
    elif primary_supported:
        interpretation = "sparse_explicit_gain_without_control_specificity"
    else:
        interpretation = "no_robust_incremental_assumption_signal"
    ready_for_cross_model = bool(
        primary_supported and specificity_supported and category_breadth and coverage_acceptable
    )
    return {
        "gate_version": "confirmatory-gate-v4-accuracy-horizon",
        "primary_future_horizon": primary_horizon,
        "primary_contrast": primary,
        "exploratory_all_assumption_eligible_contrast": exploratory_primary,
        "different_episode_control_contrast": different_episode_control,
        "same_episode_random_turn_control_contrast": same_episode_random_turn_control,
        "raw_context_contrast": raw_increment,
        "primary_category_accuracy_deltas": primary_categories,
        "raw_context_category_accuracy_deltas": raw_categories,
        "criteria": {
            "primary_accuracy_ci_excludes_zero": primary_supported,
            "different_episode_control_accuracy_ci_excludes_zero": different_episode_supported,
            "same_episode_random_turn_control_accuracy_ci_excludes_zero": same_episode_random_turn_supported,
            "control_specificity_supported": specificity_supported,
            "raw_context_accuracy_ci_excludes_zero": raw_supported,
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


def diagnostic_gate(
    pairwise: pd.DataFrame,
    long_df: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    budgets = []
    if not pairwise.empty and "representation_budget" in pairwise:
        budgets = sorted(int(value) for value in pairwise["representation_budget"].dropna().unique())
    if not budgets:
        return legacy_diagnostic_gate(pairwise, long_df, coverage)

    primary_horizon = 1
    primary_budget = 256 if 256 in budgets else budgets[len(budgets) // 2]

    def condition(base: str, budget: int = primary_budget) -> str:
        return matched_condition_id(base, budget)

    def contrast_row(target: str, baseline: str) -> dict[str, Any] | None:
        match = pairwise[
            (pairwise["analysis_subset"] == "assumption_eligible")
            & (pairwise["future_horizon"] == primary_horizon)
            & (pairwise["target_condition"] == target)
            & (pairwise["baseline_condition"] == baseline)
            & (pairwise["metric"] == "accuracy")
        ]
        return None if match.empty else match.iloc[0].to_dict()

    raw = condition("raw_history")
    true = condition("raw_history_true_assumptions")
    different_episode = condition("raw_history_different_episode_assumptions")
    same_episode_random_turn = condition("raw_history_same_episode_random_turn_assumptions")
    explicit = condition("raw_history_explicit")

    primary = contrast_row(true, different_episode)
    same_episode_random_turn_control = contrast_row(true, same_episode_random_turn)
    raw_increment = contrast_row(true, raw)
    explicit_control = contrast_row(true, explicit)

    budget_curve = {
        str(budget): {
            "true_vs_different_episode": contrast_row(
                condition("raw_history_true_assumptions", budget),
                condition("raw_history_different_episode_assumptions", budget),
            ),
            "true_vs_same_episode_random_turn": contrast_row(
                condition("raw_history_true_assumptions", budget),
                condition("raw_history_same_episode_random_turn_assumptions", budget),
            ),
            "true_vs_raw": contrast_row(
                condition("raw_history_true_assumptions", budget),
                condition("raw_history", budget),
            ),
            "true_vs_explicit": contrast_row(
                condition("raw_history_true_assumptions", budget),
                condition("raw_history_explicit", budget),
            ),
        }
        for budget in budgets
    }

    target_rows = long_df[long_df["condition"] == true].set_index("pair_id")
    baseline_rows = long_df[long_df["condition"] == different_episode].set_index("pair_id")
    category_values: list[dict[str, Any]] = []
    for pair_id in target_rows.index.intersection(baseline_rows.index):
        target_row = target_rows.loc[pair_id]
        baseline_row = baseline_rows.loc[pair_id]
        if int(target_row["future_horizon"]) == primary_horizon and bool(
            target_row["assumption_eligible"] and baseline_row["assumption_eligible"]
        ):
            category_values.append({
                "category": str(target_row["category"]),
                "delta": float(target_row["accuracy"]) - float(baseline_row["accuracy"]),
            })
    category_deltas = (
        {str(key): float(value) for key, value in pd.DataFrame(category_values).groupby("category")["delta"].mean().items()}
        if category_values else {}
    )

    def supported(row: dict[str, Any] | None) -> bool:
        return bool(row and row.get("ci95_low") is not None and float(row["ci95_low"]) > 0.0)

    relevant_coverage = coverage
    if "representation_budget" in coverage:
        relevant_coverage = coverage[coverage["representation_budget"] == primary_budget]
    minimum_retained = float(relevant_coverage["retained_pair_rate"].min()) if not relevant_coverage.empty else 0.0

    primary_supported = supported(primary)
    same_episode_random_turn_supported = supported(same_episode_random_turn_control)
    raw_supported = supported(raw_increment)
    explicit_supported = supported(explicit_control)
    specificity_supported = primary_supported and same_episode_random_turn_supported
    positive_categories = sum(value > 0 for value in category_deltas.values())
    category_breadth = positive_categories >= 2
    coverage_acceptable = minimum_retained >= 0.95

    if primary_supported and same_episode_random_turn_supported and explicit_supported and raw_supported:
        interpretation = "true_assumptions_are_specific_and_incrementally_useful_over_raw_and_explicit_controls"
    elif primary_supported and same_episode_random_turn_supported and explicit_supported:
        interpretation = "true_assumptions_show_state_specificity_beyond_generated_explicit_augmentation"
    elif primary_supported and same_episode_random_turn_supported:
        interpretation = "true_assumptions_show_local_state_specificity_but_not_clear_increment_over_raw"
    elif primary_supported:
        interpretation = "true_assumptions_beat_different_episode_control_without_same_episode_random_turn_specificity"
    else:
        interpretation = "no_robust_true_assumption_advantage_over_different_episode_control"

    ready_for_cross_model = bool(
        primary_supported and same_episode_random_turn_supported and category_breadth and coverage_acceptable
    )
    return {
        "gate_version": "raw-augmentation-gate-v1",
        "primary_future_horizon": primary_horizon,
        "primary_raw_history_budget": primary_budget,
        "budget_tokenizer": REPRESENTATION_BUDGET_TOKENIZER,
        "primary_contrast": primary,
        "same_episode_random_turn_control_contrast": same_episode_random_turn_control,
        "raw_increment_contrast": raw_increment,
        "explicit_augmentation_control_contrast": explicit_control,
        "budget_curve": budget_curve,
        "primary_category_accuracy_deltas": category_deltas,
        "criteria": {
            "true_vs_different_episode_ci_excludes_zero": primary_supported,
            "true_vs_same_episode_random_turn_ci_excludes_zero": same_episode_random_turn_supported,
            "true_vs_raw_ci_excludes_zero": raw_supported,
            "true_vs_explicit_ci_excludes_zero": explicit_supported,
            "control_specificity_supported": specificity_supported,
            "positive_category_count": positive_categories,
            "category_breadth_at_least_two": category_breadth,
            "minimum_condition_retained_rate": minimum_retained,
            "coverage_at_least_95_percent": coverage_acceptable,
        },
        "interpretation": interpretation,
        "ready_for_cross_model_smoke": ready_for_cross_model,
        "ready_for_full_corpus": False,
        "full_corpus_blocker": "Run a different-family judge smoke test and manually audit true-vs-different-episode/same-episode-random-turn flips before making the full-corpus claim.",
        "recommended_next_stage": "cross_model_smoke" if ready_for_cross_model else "manual_audit_or_control_revision",
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

    matched = long_df[long_df["representation_budget"].notna()].copy()
    if not matched.empty:
        primary_horizon = 1
        budgets = sorted(int(value) for value in matched["representation_budget"].unique())
        base_labels = {
            "raw_history_true_assumptions": "True implicit",
            "raw_history_different_episode_assumptions": "Different-Episode",
            "raw_history_same_episode_random_turn_assumptions": "Same-Episode Random Turn",
            "raw_history_explicit": "Explicit augmentation",
        }
        plotted_base_conditions = tuple(
            condition for condition in MATCHED_BASE_CONDITIONS
            if condition != PLOT_REFERENCE_BASE_CONDITION
        )
        complete = matched[
            (matched["complete_case"] == True)
            & (matched["full_retained"] == True)
            & (matched["future_horizon"] == primary_horizon)
        ]
        rows: list[dict[str, Any]] = []
        for base_condition in plotted_base_conditions:
            for budget in budgets:
                group = complete[
                    (complete["base_condition"] == base_condition)
                    & (complete["representation_budget"] == budget)
                ]
                result = cluster_bootstrap(
                    group,
                    "accuracy",
                    seed_label=f"plot:matched:h{primary_horizon}:{base_condition}:b{budget}",
                    seed=args.seed,
                    draws=args.bootstrap_draws,
                )
                rows.append({"base_condition": base_condition, "budget": budget, **result})
        plot_df = pd.DataFrame(rows)
        fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
        if complete.empty:
            axis.text(0.5, 0.5, "No complete-case data available", ha="center", va="center")
            axis.set_axis_off()
        else:
            for base_condition in plotted_base_conditions:
                group = plot_df[plot_df["base_condition"] == base_condition].set_index("budget").reindex(budgets)
                means = group["mean"].to_numpy(dtype=float)
                lows = group["ci95_low"].to_numpy(dtype=float)
                highs = group["ci95_high"].to_numpy(dtype=float)
                x = np.asarray(budgets, dtype=float)
                axis.plot(x, means, marker="o", linewidth=1.8, label=base_labels[base_condition])
                finite = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
                if finite.any():
                    axis.errorbar(
                        x[finite],
                        means[finite],
                        yerr=np.vstack((means[finite] - lows[finite], highs[finite] - means[finite])),
                        fmt="none",
                        color="black",
                        alpha=0.35,
                        capsize=3,
                    )
            axis.axhline(0.5, color="black", linewidth=0.9, linestyle="--", label="Chance")
            axis.set_xticks(budgets)
            axis.set_xlabel(f"Raw-history budget ({REPRESENTATION_BUDGET_TOKENIZER})")
            axis.set_ylabel("Binary accuracy")
            axis.set_ylim(0.0, 1.0)
            axis.set_title("Assumption-augmented future-turn accuracy (horizon 1)")
            axis.legend(loc="best", fontsize=8)
            axis.grid(axis="y", alpha=0.25)
        fig.savefig(paths["diagnostic_pdf"], bbox_inches="tight")
        fig.savefig(paths["diagnostic_png"], dpi=args.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        # Keep one fixed performance reference for every plotted lift.  Using raw history as
        # the common baseline makes vertical distances directly comparable across augmentations;
        # the zero line is the performance of the identical raw-dialogue representation.
        reference_base = PLOT_REFERENCE_BASE_CONDITION
        lift_labels = {
            "raw_history_true_assumptions": "True implicit",
            "raw_history_different_episode_assumptions": "Different-Episode",
            "raw_history_same_episode_random_turn_assumptions": "Same-Episode Random Turn",
            "raw_history_explicit": "Explicit augmentation",
        }
        lift = pairwise[
            (pairwise["analysis_subset"] == "assumption_eligible")
            & (pairwise["future_horizon"] == primary_horizon)
            & (pairwise["target_base_condition"].isin(lift_labels))
            & (pairwise["baseline_base_condition"] == reference_base)
            & (pairwise["metric"] == "accuracy")
        ].copy()
        fig, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
        if lift.empty or lift["mean_improvement"].isna().all():
            axis.text(0.5, 0.5, "No raw-reference lift data available", ha="center", va="center")
            axis.set_axis_off()
        else:
            for target_base, group in lift.groupby("target_base_condition", sort=False):
                ordered = group.set_index("representation_budget").reindex(budgets)
                means = ordered["mean_improvement"].to_numpy(dtype=float)
                lows = ordered["ci95_low"].to_numpy(dtype=float)
                highs = ordered["ci95_high"].to_numpy(dtype=float)
                x = np.asarray(budgets, dtype=float)
                axis.plot(x, means, marker="o", linewidth=1.8, label=lift_labels[str(target_base)])
                finite = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
                if finite.any():
                    axis.errorbar(
                        x[finite],
                        means[finite],
                        yerr=np.vstack((means[finite] - lows[finite], highs[finite] - means[finite])),
                        fmt="none",
                        color="black",
                        alpha=0.35,
                        capsize=3,
                    )
            # Zero is the fixed Raw-history baseline; keep the visual guide out of the legend.
            axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--", label="_nolegend_")
            axis.set_xticks(budgets)
            axis.set_xlabel(f"Raw-history budget ({REPRESENTATION_BUDGET_TOKENIZER})")
            axis.set_ylabel("Paired accuracy change vs. Raw history")
            axis.set_title("Augmentation performance relative to a fixed raw-history reference (horizon 1)")
            axis.legend(loc="best", fontsize=8)
            axis.grid(axis="y", alpha=0.25)
        fig.savefig(paths["decomposition_pdf"], bbox_inches="tight")
        fig.savefig(paths["decomposition_png"], dpi=args.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        return

    labels = {
        "raw_turn": "Raw",
        "raw_turn_with_history": "Raw + history",
        "raw_turn_plus_assumptions": "Raw + top 3 assumptions",
        "explicit_only": "Explicit",
        "explicit_plus_top1_assumption": "Explicit + first 1",
        "explicit_plus_top3_assumptions": "Explicit + grounded top 3",
        "assumptions_only": "Assumptions",
        "explicit_plus_assumptions": "Explicit + all",
        "explicit_plus_different_episode_assumptions": "Explicit + Different-Episode top 3",
        "explicit_plus_same_episode_random_turn_assumptions": "Explicit + Same-Episode Random Turn top 3",
    }
    complete = long_df[(long_df["complete_case"] == True) & (long_df["full_retained"] == True)]
    condition_rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        for future_horizon in args.future_horizons:
            group = complete[
                (complete["condition"] == condition)
                & (complete["future_horizon"] == future_horizon)
            ]
            result = cluster_bootstrap(
                group,
                "accuracy",
                seed_label=f"plot:complete:h{future_horizon}:{condition}:accuracy",
                seed=args.seed,
                draws=args.bootstrap_draws,
            )
            condition_rows.append(
                {"condition": condition, "future_horizon": future_horizon, **result}
            )
    plot_df = pd.DataFrame(condition_rows)
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    if complete.empty:
        axis.text(0.5, 0.5, "No complete-case data available", ha="center", va="center")
        axis.set_axis_off()
    else:
        for condition in args.conditions:
            metric_df = plot_df[plot_df["condition"] == condition].set_index(
                "future_horizon"
            ).reindex(args.future_horizons)
            means = metric_df["mean"].to_numpy(dtype=float)
            x = np.asarray(args.future_horizons, dtype=float)
            lows = metric_df["ci95_low"].to_numpy(dtype=float)
            highs = metric_df["ci95_high"].to_numpy(dtype=float)
            finite_intervals = np.isfinite(lows) & np.isfinite(highs) & np.isfinite(means)
            axis.plot(x, means, marker="o", linewidth=1.8, label=labels[condition])
            if finite_intervals.any():
                axis.errorbar(
                    x[finite_intervals],
                    means[finite_intervals],
                    yerr=np.vstack(
                        (
                            means[finite_intervals] - lows[finite_intervals],
                            highs[finite_intervals] - means[finite_intervals],
                        )
                    ),
                    fmt="none",
                    color="black",
                    alpha=0.45,
                    capsize=3,
                )
        axis.axhline(0.5, color="black", linewidth=0.9, linestyle="--", label="Chance")
        axis.set_xticks(args.future_horizons)
        axis.set_xlabel("Future-turn horizon n")
        axis.set_ylabel("Binary accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Which candidate is the future turn? - complete cases")
        axis.legend(loc="best", fontsize=8, ncol=2)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(paths["diagnostic_pdf"], bbox_inches="tight")
    fig.savefig(paths["diagnostic_png"], dpi=args.plot_dpi, bbox_inches="tight")
    plt.close(fig)

    decomposition_contrasts = (
        ("explicit_plus_top3_assumptions", "explicit_only"),
        ("explicit_plus_top3_assumptions", "explicit_plus_different_episode_assumptions"),
        ("explicit_plus_top3_assumptions", "explicit_plus_same_episode_random_turn_assumptions"),
        ("raw_turn_plus_assumptions", "raw_turn"),
        ("raw_turn", "explicit_only"),
        ("raw_turn_with_history", "raw_turn"),
    )
    lift_rows: list[dict[str, Any]] = []
    for target, baseline in decomposition_contrasts:
        subset = "sparse_explicit" if target == "explicit_plus_top3_assumptions" else "assumption_eligible"
        match = pairwise[
            (pairwise["analysis_subset"] == subset)
            & (pairwise["target_condition"] == target)
            & (pairwise["baseline_condition"] == baseline)
            & (pairwise["metric"] == "accuracy")
        ]
        for _, row in match.iterrows():
            lift_rows.append(
                dict(row, contrast=f"{labels[target]} - {labels[baseline]}")
            )
    lift_df = pd.DataFrame(lift_rows)
    fig, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    if lift_df.empty or lift_df["mean_improvement"].isna().all():
        axis.text(0.5, 0.5, "No diagnostic lift data available", ha="center", va="center")
        axis.set_axis_off()
    else:
        for contrast, group in lift_df.groupby("contrast", sort=False):
            ordered = group.set_index("future_horizon").reindex(args.future_horizons)
            means = ordered["mean_improvement"].to_numpy(dtype=float)
            x = np.asarray(args.future_horizons, dtype=float)
            lows = ordered["ci95_low"].to_numpy(dtype=float)
            highs = ordered["ci95_high"].to_numpy(dtype=float)
            finite_intervals = np.isfinite(lows) & np.isfinite(highs) & np.isfinite(means)
            axis.plot(x, means, marker="o", linewidth=1.8, label=contrast)
            if finite_intervals.any():
                axis.errorbar(
                    x[finite_intervals],
                    means[finite_intervals],
                    yerr=np.vstack(
                        (
                            means[finite_intervals] - lows[finite_intervals],
                            highs[finite_intervals] - means[finite_intervals],
                        )
                    ),
                    fmt="none",
                    color="black",
                    alpha=0.45,
                    capsize=3,
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(args.future_horizons)
        axis.set_xlabel("Future-turn horizon n")
        axis.set_ylabel("Paired accuracy improvement (target - baseline)")
        axis.set_title("Representation accuracy improvements across future horizons")
        axis.legend(loc="best", fontsize=8)
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
    original_scores = read_jsonl(scores_file)
    scores: list[dict[str, Any]] = []
    repaired_score_count = 0
    score_file_changed = False
    parse_method_counts: Counter[str] = Counter()
    for original_row in original_scores:
        normalized_row = normalize_score_choice(original_row)
        repaired_score_count += int(
            any(
                original_row.get(field) != normalized_row.get(field)
                for field in ("choice", "positive_preference", "parse_success", "parse_error")
            )
        )
        score_file_changed = score_file_changed or (
            canonical_json(original_row) != canonical_json(normalized_row)
        )
        parse_method_counts[str(normalized_row.get("parse_method") or "unparsed")] += 1
        scores.append(normalized_row)
    unresolved_scores = [row for row in scores if not score_record_valid(row)]
    if unresolved_scores:
        examples = [str(row.get("raw_output"))[:120] for row in unresolved_scores[:5]]
        raise RuntimeError(
            f"Analysis found {len(unresolved_scores)} unresolved forced-choice rows after "
            f"reparsing the merged scores. Example outputs: {examples}"
        )
    seen: dict[tuple[str, str, str, str, str, str], str] = {}
    for row in scores:
        key = task_key(row)
        canonical = canonical_json(row)
        previous = seen.setdefault(key, canonical)
        if previous != canonical:
            raise RuntimeError(f"Conflicting duplicate score task key during analysis: {key}")
    merge_manifest_path = args.output_dir / "exp1_representation_merge_manifest.json"
    if score_file_changed:
        merge_manifest = None
        if merge_manifest_path.exists():
            merge_manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
            expected_count = int(merge_manifest.get("merged_score_count", -1))
            if expected_count != len(scores):
                raise RuntimeError(
                    f"Merged score count changed before analysis repair: manifest={expected_count}, "
                    f"scores={len(scores)}"
                )
            expected_hash = merge_manifest.get("scores_sha256")
            observed_hash = file_hash(scores_file)
            if expected_hash and str(expected_hash) != observed_hash:
                raise RuntimeError(
                    "Merged score hash does not match its manifest before analysis repair"
                )
        write_jsonl(scores_file, scores)
        if merge_manifest is not None:
            merge_manifest["valid_score_count"] = len(scores)
            merge_manifest["repaired_score_count"] = repaired_score_count
            merge_manifest["parse_method_counts"] = dict(sorted(parse_method_counts.items()))
            merge_manifest["scores_sha256"] = file_hash(scores_file)
            write_json(merge_manifest_path, merge_manifest)
        logger.info(
            "Repaired %d/%d merged score rows from saved raw outputs",
            repaired_score_count,
            len(scores),
        )
    long_df, wide_df = build_metrics(pairs, scores, args)
    scorable_pair_condition_count = sum(
        bool(pair["candidate_pool_complete"])
        and bool(pair["conditions"].get(condition, {}).get("available"))
        for pair in pairs
        for condition in args.conditions
    )
    if scorable_pair_condition_count and not bool(long_df["full_retained"].any()):
        raise RuntimeError(
            "Analysis retained zero pair-conditions despite having "
            f"{scorable_pair_condition_count} scorable pair-conditions. Re-merge scores with "
            "the current parser or inspect unresolved forced-choice outputs."
        )
    category_df = condition_summary(long_df, "category", args)
    move_df = condition_summary(long_df, "true_future_turn_move_label", args)
    pairwise_df = build_pairwise(long_df, args)
    flip_df = build_flip_analysis(pairs, scores, args)
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
    flip_df.to_csv(paths["flips"], index=False)
    decomposition_df.to_csv(paths["decomposition"], index=False)
    audit_df.to_csv(paths["audit_sample"], index=False)
    write_json(paths["diagnostic_gate"], gate)
    coverage_df.to_csv(paths["coverage"], index=False)
    plot_results(long_df, pairwise_df, paths, args)
    hash_candidates = dict(paths)
    hash_candidates["prepared_pairs"] = prepared
    hash_candidates["prepare_manifest"] = prepare_manifest_path(args)
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
        (row["condition"], int(row["future_horizon"])): row
        for row in overall_metrics
        if row["analysis_subset"] == "complete_case"
    }
    complete_accuracy_lifts = {
        (str(row["target_condition"]), int(row["future_horizon"])): row["mean_improvement"]
        for _, row in pairwise_df.iterrows()
        if row["analysis_subset"] == "complete_case"
        and row["metric"] == "accuracy"
        and str(row["target_condition"]) in args.conditions
        and str(row["baseline_condition"]) in args.conditions
    }
    diagnostic_table = [
        {
            "condition": condition,
            "future_horizon": future_horizon,
            "accuracy": complete_lookup.get((condition, future_horizon), {}).get("accuracy"),
            "mean_order_consistency_rate": complete_lookup.get((condition, future_horizon), {}).get(
                "mean_order_consistency_rate"
            ),
            "accuracy_lift_vs_prespecified_baseline": complete_accuracy_lifts.get(
                (condition, future_horizon)
            ),
            "pairs": complete_lookup.get((condition, future_horizon), {}).get("pair_count", 0),
        }
        for future_horizon in args.future_horizons
        for condition in args.conditions
    ]
    summary = {
        "experiment": "Experiment 1: Assumption-Augmented Future-Turn Accuracy",
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
            "max_score_retries": args.max_score_retries,
            "max_retry_tokens": args.max_retry_tokens,
            "scoring_mode": "order_swapped_binary_future_turn_accuracy",
            "valid_outputs": ["A", "B"],
            "parser": "strict_json_answer_evidence",
            "comparison_rows_per_pair_condition": EXPECTED_COMPARISONS_PER_CONDITION,
        },
        "seed": args.seed,
        "candidate_count_target": EXPECTED_CANDIDATE_COUNT,
        "conditions": args.conditions,
        "representation_budgets": prepare_manifest["representation_budgets"],
        "representation_budget_tokenizer": prepare_manifest["representation_budget_tokenizer"],
        "future_horizons": prepare_manifest["future_horizons"],
        "history_turns": prepare_manifest["history_turns"],
        "source_tail_words": prepare_manifest["source_tail_words"],
        "candidate_head_words": prepare_manifest["candidate_head_words"],
        "assumption_budget": prepare_manifest["assumption_budget"],
        "pair_count": len(pairs),
        "candidate_complete_pair_count": prepare_manifest["candidate_complete_pair_count"],
        "filter_counts": {
            "episode_file_count": prepare_manifest["episode_file_count"],
            "source_episode_count": prepare_manifest["source_episode_count"],
            "turn_count": prepare_manifest["turn_count"],
            "pair_count_before_candidate_filter": prepare_manifest["pair_count_before_candidate_filter"],
            "pair_count_by_horizon": prepare_manifest["pair_count_by_horizon"],
            "candidate_complete_pair_count": prepare_manifest["candidate_complete_pair_count"],
            "candidate_incomplete_pair_count": prepare_manifest["candidate_incomplete_pair_count"],
            "assumption_eligible_pair_count": prepare_manifest["assumption_eligible_pair_count"],
            "boundary_provenance_counts": prepare_manifest["boundary_provenance_counts"],
        },
        "full_retained_by_horizon_condition": {
            f"h{int(row['future_horizon'])}:{row['condition']}": int(row["fully_parsed_pair_count"])
            for _, row in coverage_df.iterrows()
        },
        "complete_case_pair_count": int(long_df.loc[long_df["complete_case"] == True, "pair_id"].nunique()),
        "complete_case_removed_pair_count": len(pairs) - int(long_df.loc[long_df["complete_case"] == True, "pair_id"].nunique()),
        "complete_case_pair_count_by_horizon": {
            str(horizon): int(
                long_df.loc[
                    (long_df["future_horizon"] == horizon) & (long_df["complete_case"] == True),
                    "pair_id",
                ].nunique()
            )
            for horizon in prepare_manifest["future_horizons"]
        },
        "unavailable_controls": prepare_manifest["unavailable_controls"],
        "unavailable_control_reasons": prepare_manifest["unavailable_control_reasons"],
        "parse_failures_by_horizon_condition": {
            f"h{int(row['future_horizon'])}:{row['condition']}": int(
                row["comparison_parse_failure_pair_count"]
            )
            for _, row in coverage_df.iterrows()
        },
        "score_repair": {
            "score_file_changed": score_file_changed,
            "repaired_score_count": repaired_score_count,
            "valid_score_count": len(scores),
            "parse_method_counts": dict(sorted(parse_method_counts.items())),
        },
        "condition_metrics": overall_metrics,
        "complete_case_diagnostic_table": diagnostic_table,
        "diagnostic_gate": gate,
        "flip_analysis_rows": len(flip_df),
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
    for key in ("prepared_pairs_sha256", "config_sha256", "num_patches", "episodes_per_patch"):
        values = {canonical_json(manifest.get(key)) for manifest in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Mixed {key} values across patches")

    source_path_owner: dict[str, int] = {}
    for patch_index, manifest in enumerate(manifests):
        for source_path in manifest.get("selected_source_paths", []):
            source_path = str(source_path)
            previous_patch = source_path_owner.get(source_path)
            if previous_patch is not None and previous_patch != patch_index:
                raise RuntimeError(
                    f"Overlapping source path across patches {previous_patch} and {patch_index}: {source_path}"
                )
            source_path_owner[source_path] = patch_index

    prepared_pairs = read_jsonl(prepared_path(args))
    pair_source_paths: dict[str, set[str]] = defaultdict(set)
    for pair in prepared_pairs:
        pair_source_paths[str(pair["pair_id"])].add(str(pair["source_path"]))
    duplicate_pair_ids = {
        pair_id: sorted(paths)
        for pair_id, paths in pair_source_paths.items()
        if len(paths) > 1
    }
    if duplicate_pair_ids:
        pair_id, paths = next(iter(sorted(duplicate_pair_ids.items())))
        raise RuntimeError(
            "Prepared data contains a pair_id shared by multiple source files; "
            f"pair_id={pair_id!r}, source_paths={paths}. Episode IDs must be unique within a category."
        )
    pair_source_path = {pair_id: next(iter(paths)) for pair_id, paths in pair_source_paths.items()}

    merged: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str, str, str], str] = {}
    repaired_score_count = 0
    parse_method_counts: Counter[str] = Counter()
    for patch_index, directory in enumerate(expected_dirs):
        manifest = manifests[patch_index]
        selected_source_paths = {str(path) for path in manifest.get("selected_source_paths", [])}
        expected_model_name = str(manifest.get("config", {}).get("model_name"))
        expected_prompt_version = str(manifest.get("config", {}).get("prompt_version"))
        expected_conditions = {str(value) for value in manifest.get("config", {}).get("conditions", [])}
        for original_row in read_jsonl(score_path(directory)):
            row = normalize_score_choice(original_row)
            was_repaired = any(
                original_row.get(field) != row.get(field)
                for field in ("choice", "positive_preference", "parse_success", "parse_error")
            )
            pair_id = str(row["pair_id"])
            source_path = pair_source_path.get(pair_id)
            if source_path is None:
                raise RuntimeError(
                    f"Stale score row in patch {patch_index}: pair_id {pair_id!r} is absent from the prepared dataset"
                )
            if source_path not in selected_source_paths:
                raise RuntimeError(
                    "Stale score row belongs to a different patch selection: "
                    f"patch={patch_index}, pair_id={pair_id!r}, source_path={source_path!r}"
                )
            if str(row.get("model_name")) != expected_model_name or str(row.get("prompt_version")) != expected_prompt_version:
                raise RuntimeError(
                    f"Stale score configuration in patch {patch_index} for pair_id {pair_id!r}"
                )
            if str(row.get("condition")) not in expected_conditions:
                raise RuntimeError(
                    f"Stale score condition in patch {patch_index} for pair_id {pair_id!r}: {row.get('condition')!r}"
                )
            key = task_key(row)
            canonical = canonical_json(row)
            previous = seen.get(key)
            if previous is not None:
                if previous != canonical:
                    raise RuntimeError(f"Conflicting duplicate score task key during merge: {key}")
                continue
            seen[key] = canonical
            repaired_score_count += int(was_repaired)
            parse_method_counts[str(row.get("parse_method") or "unparsed")] += 1
            merged.append(row)
    expected_score_count = sum(int(manifest.get("expected_task_count", -1)) for manifest in manifests)
    if expected_score_count < 0 or len(merged) != expected_score_count:
        raise RuntimeError(
            f"Merged score count mismatch: expected {expected_score_count}, observed {len(merged)}"
        )
    unresolved = [row for row in merged if not score_record_valid(row)]
    if unresolved:
        examples = [str(row.get("raw_output"))[:120] for row in unresolved[:5]]
        raise RuntimeError(
            f"Merge found {len(unresolved)} unresolved forced-choice rows after reparsing. "
            f"Example outputs: {examples}"
        )
    presentation_order = {"positive_first": 0, "positive_second": 1}
    merged.sort(
        key=lambda row: (
            row["pair_id"],
            row["condition"],
            int(row["negative_candidate_order"]),
            presentation_order[str(row["presentation_order"])],
        )
    )
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
        "valid_score_count": len(merged) - len(unresolved),
        "repaired_score_count": repaired_score_count,
        "parse_method_counts": dict(sorted(parse_method_counts.items())),
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
