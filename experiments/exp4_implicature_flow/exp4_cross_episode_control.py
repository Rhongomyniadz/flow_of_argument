from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from tqdm import tqdm


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.entailment_labeler import (  # noqa: E402
    LLMInterface,
    MODEL_NAME,
    build_entailment_prompt,
    context_window,
    extract_turns,
    infer_category_from_lookup,
    infer_category_from_turns,
    keywords,
    normalize_category,
    overlap_score,
    safe_json_extract,
)


DEFAULT_PAIR_DIR: Path = Path("data/implicature_flow/entailment_pairs_1to10")
DEFAULT_TURN_DIR: Path = Path("data/stance_labeled/512")
DEFAULT_CATEGORY_LOOKUP_DIR: Path = Path("data/conversation_moves_labeled")
DEFAULT_OUTPUT_DIR: Path = Path("experiments/exp4_implicature_flow/results")
DEFAULT_SAMPLE_SIZE: int = 1000
DEFAULT_SEED: int = 42
DEFAULT_MIN_OVERLAP: int = 2
DEFAULT_MAX_CLAIMS_PER_ASSUMPTION: int = 15
DEFAULT_ENTAILMENT_THRESHOLD: int = 7
DEFAULT_CONTEXT_W: int = 2
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_TENSOR_PARALLEL_SIZE: int = 2
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.9
DEFAULT_MAX_TOKENS: int = 500
DEFAULT_RETRY_MAX_TOKENS: int = 4000
DEFAULT_DOWNLOAD_DIR: Path = Path("/shared/4/models")


class AssumptionRecord(TypedDict):
    assumption_id: str
    category: str
    source_episode_id: str
    source_path: str
    source_turn_idx: int
    source_assumption_idx: int
    source_time: float | None
    source_speaker_id: str
    assumption_text: str


class EpisodeSummary(TypedDict):
    episode_id: str
    category: str
    path: str
    explicit_claim_count: int


class ControlAssignment(TypedDict):
    assumption_id: str
    control_episode_id: str
    control_path: str
    control_explicit_claim_count: int


class RankedClaim(TypedDict):
    control_turn_idx: int
    control_claim_idx: int
    claim_text: str
    overlap_score: int


class BaseAssumptionResult(TypedDict):
    assumption_id: str
    category: str
    source_episode_id: str
    source_turn_idx: int
    source_assumption_idx: int
    source_speaker_id: str
    assumption_text: str
    control_episode_id: str
    control_explicit_claim_count: int
    matched_control_claims: int
    selected_control_claims: int


class PromptRecord(TypedDict):
    prompt_id: str
    assumption_id: str
    category: str
    source_episode_id: str
    source_turn_idx: int
    source_assumption_idx: int
    assumption_text: str
    control_episode_id: str
    control_turn_idx: int
    control_claim_idx: int
    claim_text: str
    overlap_score: int
    prompt: str


class PairResult(TypedDict):
    prompt_id: str
    assumption_id: str
    category: str
    source_episode_id: str
    source_turn_idx: int
    source_assumption_idx: int
    assumption_text: str
    control_episode_id: str
    control_turn_idx: int
    control_claim_idx: int
    claim_text: str
    overlap_score: int
    entailment_score: int
    confidence: float
    raw: str


class AssumptionResult(TypedDict):
    assumption_id: str
    category: str
    source_episode_id: str
    source_turn_idx: int
    source_assumption_idx: int
    source_speaker_id: str
    assumption_text: str
    control_episode_id: str
    control_explicit_claim_count: int
    matched_control_claims: int
    selected_control_claims: int
    scored_pairs: int
    max_entailment_score: int
    mean_entailment_score: float
    entailed: bool
    best_claim_text: str
    best_claim_turn_idx: str
    best_claim_idx: str
    best_overlap_score: str
    best_confidence: str


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the Exp4 cross-episode entailment control baseline."
    )
    parser.add_argument("--pair_dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--turn_dir", type=Path, default=DEFAULT_TURN_DIR)
    parser.add_argument("--category_lookup_dir", type=Path, default=DEFAULT_CATEGORY_LOOKUP_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min_overlap", type=int, default=DEFAULT_MIN_OVERLAP)
    parser.add_argument(
        "--max_claims_per_assumption",
        type=int,
        default=DEFAULT_MAX_CLAIMS_PER_ASSUMPTION,
    )
    parser.add_argument(
        "--entailment_threshold",
        type=int,
        default=DEFAULT_ENTAILMENT_THRESHOLD,
    )
    parser.add_argument("--context_w", type=int, default=DEFAULT_CONTEXT_W)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--retry_max_tokens",
        type=int,
        default=DEFAULT_RETRY_MAX_TOKENS,
    )
    parser.add_argument("--download_dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    return parser.parse_args()


def require_existing_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}")


def validate_args(args: argparse.Namespace) -> None:
    require_existing_directory(args.pair_dir, "pair")
    require_existing_directory(args.turn_dir, "turn")
    require_existing_directory(args.category_lookup_dir, "category lookup")
    if args.sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {args.sample_size}")
    if args.min_overlap < 0:
        raise ValueError(f"min_overlap must be nonnegative, got {args.min_overlap}")
    if args.max_claims_per_assumption < 1:
        raise ValueError(
            "max_claims_per_assumption must be positive, "
            f"got {args.max_claims_per_assumption}"
        )
    if args.entailment_threshold < 0 or args.entailment_threshold > 10:
        raise ValueError(
            "entailment_threshold must be between 0 and 10, "
            f"got {args.entailment_threshold}"
        )
    if args.context_w < 0:
        raise ValueError(f"context_w must be nonnegative, got {args.context_w}")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.tensor_parallel_size < 1:
        raise ValueError(
            "tensor_parallel_size must be positive, "
            f"got {args.tensor_parallel_size}"
        )
    if args.gpu_memory_utilization <= 0.0 or args.gpu_memory_utilization > 1.0:
        raise ValueError(
            "gpu_memory_utilization must be in (0, 1], "
            f"got {args.gpu_memory_utilization}"
        )
    if args.max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {args.max_tokens}")
    if args.retry_max_tokens < args.max_tokens:
        raise ValueError(
            "retry_max_tokens must be at least max_tokens, "
            f"got retry_max_tokens={args.retry_max_tokens}, max_tokens={args.max_tokens}"
        )


def read_json_any(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def read_json_object(path: Path) -> dict[str, Any]:
    obj: Any = read_json_any(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(obj).__name__}")
    return obj


def read_turns(path: Path) -> list[dict[str, Any]]:
    obj: Any = read_json_any(path)
    try:
        return extract_turns(obj)
    except ValueError as exc:
        raise ValueError(f"Invalid turn JSON in {path}: {exc}") from exc


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value).__name__}")
    stripped: str = value.strip()
    if not stripped:
        raise ValueError(f"{label} must be nonempty")
    return stripped


def optional_string(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Expected optional string, got {type(value).__name__}")
    return value.strip()


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if stripped:
            numeric: float = float(stripped)
            if numeric.is_integer():
                return int(numeric)
    raise ValueError(f"{label} must be an integer, got {value!r}")


def optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number or null, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if stripped:
            return float(stripped)
    raise ValueError(f"{label} must be a number or null, got {value!r}")


def require_category(value: Any, path: Path) -> str:
    category: str | None = normalize_category(value if isinstance(value, str) else None)
    if category is None:
        raise ValueError(f"Missing category metadata in {path}")
    return category


def infer_category_for_episode(
    metadata_value: Any,
    episode_id: str,
    episode_path: Path,
    category_lookup_dir: Path,
    metadata_path: Path,
) -> str:
    metadata_category: str | None = normalize_category(
        metadata_value if isinstance(metadata_value, str) else None
    )
    if metadata_category is not None:
        return metadata_category

    lookup_category: str | None = infer_category_from_lookup(
        str(episode_path),
        str(category_lookup_dir),
    )
    if lookup_category is not None:
        return lookup_category

    turns: list[dict[str, Any]] = read_turns(episode_path)
    turn_category: str | None = infer_category_from_turns(turns)
    if turn_category is not None:
        return turn_category

    raise ValueError(
        "Missing category metadata and lookup entry for "
        f"episode_id={episode_id}, metadata_path={metadata_path}, "
        f"episode_path={episode_path}, category_lookup_dir={category_lookup_dir}"
    )


def assumption_id(
    category: str,
    episode_id: str,
    turn_idx: int,
    assumption_idx: int,
    assumption_text: str,
) -> str:
    text_key: str = hashlib.sha1(assumption_text.encode("utf-8")).hexdigest()[:16]
    return f"{category}|{episode_id}|{turn_idx}|{assumption_idx}|{text_key}"


def pair_json_paths(pair_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(pair_dir.glob("*.json"))
        if not path.name.startswith("_LABELING_META")
    ]


def collect_unique_assumptions(
    pair_dir: Path,
    turn_dir: Path,
    category_lookup_dir: Path,
) -> list[AssumptionRecord]:
    records: list[AssumptionRecord] = []
    seen: set[tuple[str, str, int, int, str]] = set()

    for path in tqdm(pair_json_paths(pair_dir), desc="Collect assumptions", unit="file"):
        obj: dict[str, Any] = read_json_object(path)
        episode_id: str = require_nonempty_string(obj.get("episode_id"), f"{path}: episode_id")
        pairs_value: Any = obj.get("pairs")
        if not isinstance(pairs_value, list):
            raise ValueError(f"{path}: pairs must be a list")

        source_path: Path = turn_dir / f"{episode_id}.json"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing source turn file for episode_id={episode_id}: {source_path}"
            )
        category: str = infer_category_for_episode(
            obj.get("category"),
            episode_id,
            source_path,
            category_lookup_dir,
            path,
        )

        for pair_idx, pair_value in enumerate(pairs_value):
            if not isinstance(pair_value, dict):
                raise ValueError(f"{path}: pair {pair_idx} must be an object")

            source_turn_idx: int = require_int(
                pair_value.get("a_turn_idx"),
                f"{path}: pair {pair_idx} a_turn_idx",
            )
            source_assumption_idx: int = require_int(
                pair_value.get("a_idx_in_turn"),
                f"{path}: pair {pair_idx} a_idx_in_turn",
            )
            assumption_text_value: str = require_nonempty_string(
                pair_value.get("assumption_text"),
                f"{path}: pair {pair_idx} assumption_text",
            )
            key: tuple[str, str, int, int, str] = (
                category,
                episode_id,
                source_turn_idx,
                source_assumption_idx,
                assumption_text_value,
            )
            if key in seen:
                continue
            seen.add(key)

            speaker_value: Any = pair_value.get("a_turn_speaker_id")
            if speaker_value is None:
                speaker_value = pair_value.get("a_speaker_id")

            records.append(
                {
                    "assumption_id": assumption_id(
                        category,
                        episode_id,
                        source_turn_idx,
                        source_assumption_idx,
                        assumption_text_value,
                    ),
                    "category": category,
                    "source_episode_id": episode_id,
                    "source_path": str(source_path),
                    "source_turn_idx": source_turn_idx,
                    "source_assumption_idx": source_assumption_idx,
                    "source_time": optional_float(
                        pair_value.get("a_time"),
                        f"{path}: pair {pair_idx} a_time",
                    ),
                    "source_speaker_id": optional_string(speaker_value),
                    "assumption_text": assumption_text_value,
                }
            )

    return records


def explicit_claim_count(turns: list[dict[str, Any]], path: Path) -> int:
    count: int = 0
    for turn_idx, turn in enumerate(turns):
        explicit_value: Any = turn.get("explicit_propositions")
        if explicit_value is None:
            continue
        if not isinstance(explicit_value, list):
            raise ValueError(f"{path}: turn {turn_idx} explicit_propositions must be a list")
        for claim_idx, claim_value in enumerate(explicit_value):
            if not isinstance(claim_value, dict):
                raise ValueError(
                    f"{path}: turn {turn_idx} explicit claim {claim_idx} must be an object"
                )
            text_value: Any = claim_value.get("text")
            if isinstance(text_value, str) and text_value.strip():
                count += 1
    return count


def collect_control_episode_index(
    turn_dir: Path,
    category_lookup_dir: Path,
) -> dict[str, list[EpisodeSummary]]:
    by_category: dict[str, list[EpisodeSummary]] = defaultdict(list)
    turn_paths: list[Path] = [
        path
        for path in sorted(turn_dir.glob("*.json"))
        if not path.name.startswith("_LABELING_META")
    ]

    for path in tqdm(turn_paths, desc="Index control episodes", unit="file"):
        turns: list[dict[str, Any]] = read_turns(path)
        category: str = infer_category_for_episode(
            None,
            path.stem,
            path,
            category_lookup_dir,
            path,
        )
        claim_count: int = explicit_claim_count(turns, path)
        if claim_count < 1:
            continue
        by_category[category].append(
            {
                "episode_id": path.stem,
                "category": category,
                "path": str(path),
                "explicit_claim_count": claim_count,
            }
        )

    return dict(by_category)


def sample_assumptions(
    records: list[AssumptionRecord],
    sample_size: int,
    seed: int,
) -> list[AssumptionRecord]:
    if len(records) < sample_size:
        raise ValueError(
            "Not enough eligible assumptions for requested control sample: "
            f"eligible={len(records)}, requested={sample_size}"
        )
    rng: random.Random = random.Random(seed)
    return rng.sample(records, sample_size)


def assign_control_episodes(
    assumptions: list[AssumptionRecord],
    episode_index: dict[str, list[EpisodeSummary]],
    seed: int,
) -> list[ControlAssignment]:
    rng: random.Random = random.Random(seed + 1)
    assignments: list[ControlAssignment] = []

    for assumption in assumptions:
        category: str = assumption["category"]
        candidates: list[EpisodeSummary] = [
            episode
            for episode in episode_index.get(category, [])
            if episode["episode_id"] != assumption["source_episode_id"]
        ]
        if not candidates:
            raise ValueError(
                "No different same-category control episode with explicit claims exists for "
                f"source_episode_id={assumption['source_episode_id']}, category={category}"
            )
        selected: EpisodeSummary = rng.choice(candidates)
        assignments.append(
            {
                "assumption_id": assumption["assumption_id"],
                "control_episode_id": selected["episode_id"],
                "control_path": selected["path"],
                "control_explicit_claim_count": selected["explicit_claim_count"],
            }
        )

    return assignments


def load_needed_turns(
    assumptions: list[AssumptionRecord],
    assignments: list[ControlAssignment],
) -> dict[str, list[dict[str, Any]]]:
    paths: set[str] = {assumption["source_path"] for assumption in assumptions}
    paths.update(assignment["control_path"] for assignment in assignments)
    return {path: read_turns(Path(path)) for path in tqdm(sorted(paths), desc="Load turns", unit="file")}


def source_context(
    turns_by_path: dict[str, list[dict[str, Any]]],
    assumption: AssumptionRecord,
    context_w: int,
) -> str:
    source_path: str = assumption["source_path"]
    turns: list[dict[str, Any]] = turns_by_path[source_path]
    turn_idx: int = assumption["source_turn_idx"]
    if turn_idx < 0 or turn_idx >= len(turns):
        raise ValueError(
            "Source turn index out of range for "
            f"episode_id={assumption['source_episode_id']}, "
            f"category={assumption['category']}, turn_idx={turn_idx}, "
            f"num_turns={len(turns)}, source_path={source_path}"
        )
    return context_window(turns, turn_idx, context_w)


def ranked_control_claims(
    turns: list[dict[str, Any]],
    control_path: str,
    assumption_text: str,
    min_overlap: int,
    max_claims: int,
) -> tuple[list[RankedClaim], int]:
    assumption_keywords: set[str] = keywords(assumption_text)
    candidates: list[RankedClaim] = []

    for turn_idx, turn in enumerate(turns):
        explicit_value: Any = turn.get("explicit_propositions")
        if explicit_value is None:
            continue
        if not isinstance(explicit_value, list):
            raise ValueError(
                f"{control_path}: turn {turn_idx} explicit_propositions must be a list"
            )
        for claim_idx, claim_value in enumerate(explicit_value):
            if not isinstance(claim_value, dict):
                raise ValueError(
                    f"{control_path}: turn {turn_idx} explicit claim {claim_idx} "
                    "must be an object"
                )
            claim_text: Any = claim_value.get("text")
            if not isinstance(claim_text, str) or not claim_text.strip():
                continue
            score: int = overlap_score(assumption_keywords, keywords(claim_text))
            if score >= min_overlap:
                candidates.append(
                    {
                        "control_turn_idx": turn_idx,
                        "control_claim_idx": claim_idx,
                        "claim_text": claim_text.strip(),
                        "overlap_score": score,
                    }
                )

    ranked: list[RankedClaim] = sorted(
        candidates,
        key=lambda claim: (
            -claim["overlap_score"],
            claim["control_turn_idx"],
            claim["control_claim_idx"],
        ),
    )
    return ranked[:max_claims], len(candidates)


def build_prompt_records(
    assumptions: list[AssumptionRecord],
    assignments: list[ControlAssignment],
    turns_by_path: dict[str, list[dict[str, Any]]],
    min_overlap: int,
    max_claims: int,
    context_w: int,
) -> tuple[list[BaseAssumptionResult], list[PromptRecord]]:
    assignment_by_id: dict[str, ControlAssignment] = {
        assignment["assumption_id"]: assignment for assignment in assignments
    }
    base_rows: list[BaseAssumptionResult] = []
    prompt_records: list[PromptRecord] = []

    for assumption in tqdm(assumptions, desc="Build control prompts", unit="assumption"):
        assignment: ControlAssignment = assignment_by_id[assumption["assumption_id"]]
        control_path: str = assignment["control_path"]
        control_turns: list[dict[str, Any]] = turns_by_path[control_path]
        ranked_claims, matched_claims = ranked_control_claims(
            control_turns,
            control_path,
            assumption["assumption_text"],
            min_overlap,
            max_claims,
        )
        a_context: str = source_context(turns_by_path, assumption, context_w)

        base_rows.append(
            {
                "assumption_id": assumption["assumption_id"],
                "category": assumption["category"],
                "source_episode_id": assumption["source_episode_id"],
                "source_turn_idx": assumption["source_turn_idx"],
                "source_assumption_idx": assumption["source_assumption_idx"],
                "source_speaker_id": assumption["source_speaker_id"],
                "assumption_text": assumption["assumption_text"],
                "control_episode_id": assignment["control_episode_id"],
                "control_explicit_claim_count": assignment["control_explicit_claim_count"],
                "matched_control_claims": matched_claims,
                "selected_control_claims": len(ranked_claims),
            }
        )

        for prompt_idx, claim in enumerate(ranked_claims):
            c_context: str = context_window(control_turns, claim["control_turn_idx"], context_w)
            prompt_records.append(
                {
                    "prompt_id": f"{assumption['assumption_id']}|{prompt_idx}",
                    "assumption_id": assumption["assumption_id"],
                    "category": assumption["category"],
                    "source_episode_id": assumption["source_episode_id"],
                    "source_turn_idx": assumption["source_turn_idx"],
                    "source_assumption_idx": assumption["source_assumption_idx"],
                    "assumption_text": assumption["assumption_text"],
                    "control_episode_id": assignment["control_episode_id"],
                    "control_turn_idx": claim["control_turn_idx"],
                    "control_claim_idx": claim["control_claim_idx"],
                    "claim_text": claim["claim_text"],
                    "overlap_score": claim["overlap_score"],
                    "prompt": build_entailment_prompt(
                        assumption_text=assumption["assumption_text"],
                        claim_text=claim["claim_text"],
                        a_turn_idx=assumption["source_turn_idx"],
                        c_turn_idx=claim["control_turn_idx"],
                        a_context=a_context,
                        c_context=c_context,
                    ),
                }
            )

    return base_rows, prompt_records


def output_has_required_fields(parsed: dict[str, Any] | None) -> bool:
    return (
        isinstance(parsed, dict)
        and "entailment_score" in parsed
        and "confidence" in parsed
    )


def parse_outputs(raw_outputs: list[str]) -> tuple[list[dict[str, Any] | None], list[int]]:
    parsed_outputs: list[dict[str, Any] | None] = []
    failed_indices: list[int] = []
    for idx, raw_output in enumerate(raw_outputs):
        parsed: dict[str, Any] | None = safe_json_extract(raw_output)
        parsed_outputs.append(parsed)
        if not output_has_required_fields(parsed):
            failed_indices.append(idx)
    return parsed_outputs, failed_indices


def generate_raw_outputs(
    prompts: list[str],
    llm: LLMInterface,
    batch_size: int,
    max_tokens: int,
    retry_max_tokens: int,
) -> list[str]:
    raw_outputs: list[str] = [""] * len(prompts)
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Score cross-episode pairs",
        unit="batch",
    ):
        end: int = min(start + batch_size, len(prompts))
        batch_prompts: list[str] = prompts[start:end]
        batch_outputs: list[str] = llm.generate_batch(batch_prompts, max_tokens=max_tokens)
        for offset, output in enumerate(batch_outputs):
            raw_outputs[start + offset] = output

    _, failed_indices = parse_outputs(raw_outputs)
    if not failed_indices:
        return raw_outputs

    retry_prompts: list[str] = [prompts[idx] for idx in failed_indices]
    retry_outputs: list[str] = []
    print(
        "Retrying malformed model outputs with larger max_tokens: "
        f"failed={len(failed_indices)}, total={len(prompts)}"
    )
    for start in tqdm(
        range(0, len(retry_prompts), batch_size),
        desc="Retry malformed outputs",
        unit="batch",
    ):
        end = min(start + batch_size, len(retry_prompts))
        retry_outputs.extend(
            llm.generate_batch(retry_prompts[start:end], max_tokens=retry_max_tokens)
        )

    for failed_idx, retry_output in zip(failed_indices, retry_outputs):
        raw_outputs[failed_idx] = retry_output

    parsed_outputs, remaining_failed_indices = parse_outputs(raw_outputs)
    if remaining_failed_indices:
        first_idx: int = remaining_failed_indices[0]
        raise ValueError(
            "Failed to parse model outputs after retry: "
            f"remaining_failed={len(remaining_failed_indices)}, "
            f"first_failed_index={first_idx}, "
            f"first_raw_output={raw_outputs[first_idx]!r}, "
            f"first_parsed={parsed_outputs[first_idx]!r}"
        )

    return raw_outputs


def numeric_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if stripped:
            return float(stripped)
    raise ValueError(f"{label} must be numeric, got {value!r}")


def model_score(value: Any, label: str) -> int:
    numeric: float = numeric_float(value, label)
    if not numeric.is_integer():
        raise ValueError(f"{label} must be integer-valued, got {value!r}")
    score: int = int(numeric)
    if score < 0 or score > 10:
        raise ValueError(f"{label} must be between 0 and 10, got {score}")
    return score


def model_confidence(value: Any, label: str) -> float:
    confidence: float = numeric_float(value, label)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"{label} must be between 0 and 1, got {confidence}")
    return confidence


def build_pair_result(
    prompt_record: PromptRecord,
    parsed: dict[str, Any],
    raw_output: str,
) -> PairResult:
    return {
        "prompt_id": prompt_record["prompt_id"],
        "assumption_id": prompt_record["assumption_id"],
        "category": prompt_record["category"],
        "source_episode_id": prompt_record["source_episode_id"],
        "source_turn_idx": prompt_record["source_turn_idx"],
        "source_assumption_idx": prompt_record["source_assumption_idx"],
        "assumption_text": prompt_record["assumption_text"],
        "control_episode_id": prompt_record["control_episode_id"],
        "control_turn_idx": prompt_record["control_turn_idx"],
        "control_claim_idx": prompt_record["control_claim_idx"],
        "claim_text": prompt_record["claim_text"],
        "overlap_score": prompt_record["overlap_score"],
        "entailment_score": model_score(
            parsed.get("entailment_score"),
            f"{prompt_record['prompt_id']} entailment_score",
        ),
        "confidence": model_confidence(
            parsed.get("confidence"),
            f"{prompt_record['prompt_id']} confidence",
        ),
        "raw": raw_output,
    }


def score_prompt_records(
    prompt_records: list[PromptRecord],
    model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    download_dir: Path,
    batch_size: int,
    max_tokens: int,
    retry_max_tokens: int,
) -> list[PairResult]:
    if not prompt_records:
        return []

    llm: LLMInterface = LLMInterface(
        model_name=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        repetition_penalty=1.05,
        download_dir=str(download_dir),
        default_max_tokens=max_tokens,
    )
    prompts: list[str] = [record["prompt"] for record in prompt_records]
    raw_outputs: list[str] = generate_raw_outputs(
        prompts,
        llm,
        batch_size,
        max_tokens,
        retry_max_tokens,
    )
    parsed_outputs, failed_indices = parse_outputs(raw_outputs)
    if failed_indices:
        raise ValueError(
            "Internal parse validation failed after retry: "
            f"failed_indices={failed_indices[:10]}"
        )

    pair_results: list[PairResult] = []
    for prompt_record, parsed_output, raw_output in zip(
        prompt_records,
        parsed_outputs,
        raw_outputs,
    ):
        if parsed_output is None:
            raise ValueError(f"Missing parsed output for prompt_id={prompt_record['prompt_id']}")
        pair_results.append(build_pair_result(prompt_record, parsed_output, raw_output))
    return pair_results


def best_pair(pairs: list[PairResult]) -> PairResult:
    return max(
        pairs,
        key=lambda pair: (
            pair["entailment_score"],
            pair["confidence"],
            pair["overlap_score"],
            -pair["control_turn_idx"],
            -pair["control_claim_idx"],
        ),
    )


def build_assumption_results(
    base_rows: list[BaseAssumptionResult],
    pair_results: list[PairResult],
    entailment_threshold: int,
) -> list[AssumptionResult]:
    pairs_by_assumption: dict[str, list[PairResult]] = defaultdict(list)
    for pair in pair_results:
        pairs_by_assumption[pair["assumption_id"]].append(pair)

    assumption_results: list[AssumptionResult] = []
    for base_row in base_rows:
        pairs: list[PairResult] = pairs_by_assumption.get(base_row["assumption_id"], [])
        if pairs:
            scores: list[int] = [pair["entailment_score"] for pair in pairs]
            max_score: int = max(scores)
            mean_score: float = sum(scores) / len(scores)
            best: PairResult = best_pair(pairs)
            best_claim_text: str = best["claim_text"]
            best_claim_turn_idx: str = str(best["control_turn_idx"])
            best_claim_idx: str = str(best["control_claim_idx"])
            best_overlap: str = str(best["overlap_score"])
            best_confidence: str = f"{best['confidence']:.6f}"
        else:
            max_score = 0
            mean_score = 0.0
            best_claim_text = ""
            best_claim_turn_idx = ""
            best_claim_idx = ""
            best_overlap = ""
            best_confidence = ""

        assumption_results.append(
            {
                **base_row,
                "scored_pairs": len(pairs),
                "max_entailment_score": max_score,
                "mean_entailment_score": mean_score,
                "entailed": max_score >= entailment_threshold,
                "best_claim_text": best_claim_text,
                "best_claim_turn_idx": best_claim_turn_idx,
                "best_claim_idx": best_claim_idx,
                "best_overlap_score": best_overlap,
                "best_confidence": best_confidence,
            }
        )

    return assumption_results


def rate_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def summarize_category(
    category: str,
    rows: list[AssumptionResult],
    pair_results: list[PairResult],
    entailment_threshold: int,
) -> dict[str, Any]:
    category_rows: list[AssumptionResult] = [
        row for row in rows if row["category"] == category
    ]
    category_pairs: list[PairResult] = [
        pair for pair in pair_results if pair["category"] == category
    ]
    entailed_assumptions: int = sum(1 for row in category_rows if row["entailed"])
    assumptions_with_candidates: int = sum(
        1 for row in category_rows if row["selected_control_claims"] > 0
    )
    entailed_pairs: int = sum(
        1 for pair in category_pairs if pair["entailment_score"] >= entailment_threshold
    )
    return {
        "category": category,
        "sampled_assumptions": len(category_rows),
        "assumptions_with_candidate_claims": assumptions_with_candidates,
        "entailed_assumptions": entailed_assumptions,
        "cross_episode_entailment_rate_percent": rate_percent(
            entailed_assumptions,
            len(category_rows),
        ),
        "scored_pairs": len(category_pairs),
        "entailed_pairs": entailed_pairs,
        "pair_entailment_rate_percent": rate_percent(entailed_pairs, len(category_pairs)),
    }


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "exp4_cross_episode_control_summary.json",
        "assumptions": output_dir / "exp4_cross_episode_control_assumptions.csv",
        "pairs": output_dir / "exp4_cross_episode_control_pairs.csv",
    }


def build_summary(
    rows: list[AssumptionResult],
    pair_results: list[PairResult],
    eligible_assumptions: int,
    episode_index: dict[str, list[EpisodeSummary]],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    entailed_assumptions: int = sum(1 for row in rows if row["entailed"])
    assumptions_with_candidates: int = sum(
        1 for row in rows if row["selected_control_claims"] > 0
    )
    entailed_pairs: int = sum(
        1
        for pair in pair_results
        if pair["entailment_score"] >= args.entailment_threshold
    )
    categories: list[str] = sorted({row["category"] for row in rows})
    control_episode_counts: dict[str, int] = {
        category: len(episodes) for category, episodes in sorted(episode_index.items())
    }
    return {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "pair_dir": str(args.pair_dir),
        "turn_dir": str(args.turn_dir),
        "output_dir": str(args.output_dir),
        "params": {
            "category_lookup_dir": str(args.category_lookup_dir),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "min_overlap": args.min_overlap,
            "max_claims_per_assumption": args.max_claims_per_assumption,
            "entailment_threshold": args.entailment_threshold,
            "context_w": args.context_w,
            "model_name": args.model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "retry_max_tokens": args.retry_max_tokens,
            "download_dir": str(args.download_dir),
        },
        "sampling_frame": {
            "eligible_assumptions": eligible_assumptions,
            "sampled_assumptions": len(rows),
            "control_episode_counts_by_category": control_episode_counts,
        },
        "global_metrics": {
            "sampled_assumptions": len(rows),
            "assumptions_with_candidate_claims": assumptions_with_candidates,
            "assumptions_without_candidate_claims": len(rows) - assumptions_with_candidates,
            "entailed_assumptions": entailed_assumptions,
            "cross_episode_entailment_rate_percent": rate_percent(
                entailed_assumptions,
                len(rows),
            ),
            "scored_pairs": len(pair_results),
            "entailed_pairs": entailed_pairs,
            "pair_entailment_rate_percent": rate_percent(entailed_pairs, len(pair_results)),
        },
        "by_category": [
            summarize_category(category, rows, pair_results, args.entailment_threshold)
            for category in categories
        ],
        "outputs": {name: str(path) for name, path in paths.items()},
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    metrics: dict[str, Any] = summary["global_metrics"]
    print("\nEXP4 CROSS-EPISODE CONTROL COMPLETE")
    print(f"Sampled assumptions: {metrics['sampled_assumptions']}")
    print(f"Scored pairs: {metrics['scored_pairs']}")
    print(
        "Cross-episode entailment rate: "
        f"{metrics['entailed_assumptions']}/{metrics['sampled_assumptions']} "
        f"({metrics['cross_episode_entailment_rate_percent']}%)"
    )
    print(f"Output directory: {summary['output_dir']}")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    assumptions: list[AssumptionRecord] = collect_unique_assumptions(
        args.pair_dir,
        args.turn_dir,
        args.category_lookup_dir,
    )
    sampled_assumptions: list[AssumptionRecord] = sample_assumptions(
        assumptions,
        args.sample_size,
        args.seed,
    )
    episode_index: dict[str, list[EpisodeSummary]] = collect_control_episode_index(
        args.turn_dir,
        args.category_lookup_dir,
    )
    assignments: list[ControlAssignment] = assign_control_episodes(
        sampled_assumptions,
        episode_index,
        args.seed,
    )
    turns_by_path: dict[str, list[dict[str, Any]]] = load_needed_turns(
        sampled_assumptions,
        assignments,
    )
    base_rows, prompt_records = build_prompt_records(
        sampled_assumptions,
        assignments,
        turns_by_path,
        args.min_overlap,
        args.max_claims_per_assumption,
        args.context_w,
    )
    pair_results: list[PairResult] = score_prompt_records(
        prompt_records,
        args.model_name,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.download_dir,
        args.batch_size,
        args.max_tokens,
        args.retry_max_tokens,
    )
    assumption_results: list[AssumptionResult] = build_assumption_results(
        base_rows,
        pair_results,
        args.entailment_threshold,
    )
    paths: dict[str, Path] = output_paths(args.output_dir)
    summary: dict[str, Any] = build_summary(
        assumption_results,
        pair_results,
        len(assumptions),
        episode_index,
        args,
        paths,
    )

    write_json(paths["summary"], summary)
    write_csv(paths["assumptions"], list(assumption_results), list(AssumptionResult.__annotations__))
    write_csv(paths["pairs"], list(pair_results), list(PairResult.__annotations__))
    print_summary(summary)


if __name__ == "__main__":
    run(parse_args())
