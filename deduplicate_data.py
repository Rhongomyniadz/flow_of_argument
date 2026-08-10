#!/usr/bin/env python3
"""Clean, deduplicate, and enforce two-speaker ABAB episode data.

The script scans a data tree recursively and processes JSON episode files that
contain turn records. It deliberately skips manifests, entailment-pair outputs,
and extraction ``raw/`` files that do not contain speaker identities.

For each episode it performs these operations in order:

1. Delete turns whose ``turn_text`` contains fewer than ``--min-words`` words.
2. Require exactly two remaining speakers.
3. Merge adjacent remaining turns from the same speaker.
4. Keep at most ``--max-statements`` explicit propositions and assumptions,
   ranked by confidence whenever truncation is required.
5. Renumber turns and validate strict two-speaker alternation (ABAB or BABA).
6. Deduplicate episodes by their normalized cleaned speaker/text sequence.

The original data is never modified. Cleaned files retain their paths relative
to the input root under a separate output root. When turns are merged, metadata
from the last component is retained because it represents the most recent
dialogue state; text, annotations, timing, counts, and original-turn provenance
are recomputed. Provenance is retained for singleton and merged turns.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

LOG = logging.getLogger("deduplicate-data")

SCRIPT_VERSION = "1.1.1"
DEFAULT_INPUT_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("data_cleaned")
DEFAULT_MIN_WORDS = 50
DEFAULT_MAX_STATEMENTS = 10

TURN_LIST_KEYS = ("turns", "records", "utterances")
TURN_TEXT_KEYS = ("turn_text", "turnText", "transcript", "text")
SPEAKER_KEYS = ("speaker_id", "speaker")
EXPLICIT_KEYS = ("explicit_propositions", "explicit_statements")
ASSUMPTION_KEYS = ("assumptions", "implicit_assumptions")
ANNOTATION_KEYS = EXPLICIT_KEYS + ASSUMPTION_KEYS
START_TIME_KEYS = ("start_time", "startTime")
END_TIME_KEYS = ("end_time", "endTime")

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

MANIFEST_NAME = "deduplication_manifest.json"
AUDIT_NAME = "deduplication_file_audit.jsonl"


@dataclass
class CleaningStats:
    original_turns: int = 0
    final_turns: int = 0
    short_turns_removed: int = 0
    missing_text_turns_removed: int = 0
    merge_groups: int = 0
    turns_absorbed_by_merging: int = 0
    annotation_duplicates_removed: int = 0
    annotation_items_removed_by_cap: int = 0

    def add(self, other: "CleaningStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class CleanedEpisode:
    root_kind: str
    turn_container_key: str | None
    root_metadata: dict[str, Any] | None
    turns: list[dict[str, Any]]
    speakers: list[str]
    stats: CleaningStats


@dataclass
class DatasetStats:
    json_files_seen: int = 0
    episode_files_detected: int = 0
    files_written: int = 0
    duplicate_files_removed: int = 0
    skipped_files: Counter[str] = field(default_factory=Counter)
    cleaning: CleaningStats = field(default_factory=CleaningStats)

    def to_json(self) -> dict[str, Any]:
        return {
            "json_files_seen": self.json_files_seen,
            "episode_files_detected": self.episode_files_detected,
            "files_written": self.files_written,
            "duplicate_files_removed": self.duplicate_files_removed,
            "skipped_files": dict(sorted(self.skipped_files.items())),
            "cleaning": asdict(self.cleaning),
        }


class EpisodeRejected(Exception):
    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate episode JSON data and enforce two-speaker ABAB turns."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Data tree to scan recursively (default: data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Separate destination tree (default: data_cleaned).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help="Delete turns with fewer words than this value (default: 50).",
    )
    parser.add_argument(
        "--max-statements",
        type=int,
        default=DEFAULT_MAX_STATEMENTS,
        help="Maximum explicit propositions or assumptions per final turn (default: 10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root after safety validation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-file progress and skip logging.",
    )
    return parser.parse_args(argv)


def canonical_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_roots(input_root: Path, output_root: Path) -> None:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {input_root}")
    if input_root == output_root:
        raise ValueError("Input and output roots must be different")
    if is_relative_to(output_root, input_root):
        raise ValueError("Output root must not be inside the input data tree")
    if output_root.parent == output_root:
        raise ValueError("Refusing to use a filesystem root as the output directory")


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite to replace it."
            )
        if output_root.parent == output_root or len(output_root.parts) < 2:
            raise ValueError(f"Refusing to remove unsafe output path: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return WHITESPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def turn_text(turn: dict[str, Any]) -> str:
    for key in TURN_TEXT_KEYS:
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def speaker_value(turn: dict[str, Any]) -> str:
    for key in SPEAKER_KEYS:
        value = turn.get(key)
        if value is not None and str(value).strip():
            return normalize_text(value)
    return ""


def finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def annotation_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("value", "list", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        if "text" in value:
            return [value]
        return []
    if isinstance(value, str):
        return [value]
    return []


def normalize_annotation(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        text = normalize_text(item.get("text"))
        if not text:
            return None
        normalized = copy.deepcopy(item)
        normalized["text"] = text
        confidence = item.get("confidence", item.get("confidence_score", 0.0))
        normalized["confidence"] = finite_float(confidence)
        return normalized
    text = normalize_text(item)
    if not text:
        return None
    return {"text": text, "confidence": 0.0}


def clean_annotations(
    values: Iterable[Any],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], int, int]:
    ordered: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    duplicate_count = 0
    for raw_item in values:
        item = normalize_annotation(raw_item)
        if item is None:
            continue
        key = normalize_text(item["text"]).casefold()
        existing_position = positions.get(key)
        if existing_position is None:
            positions[key] = len(ordered)
            ordered.append(item)
            continue
        duplicate_count += 1
        existing = ordered[existing_position]
        if finite_float(item.get("confidence")) > finite_float(existing.get("confidence")):
            ordered[existing_position] = item

    removed_by_cap = max(0, len(ordered) - limit)
    if removed_by_cap:
        ranked = sorted(
            enumerate(ordered),
            key=lambda row: (-finite_float(row[1].get("confidence")), row[0]),
        )
        ordered = [item for _, item in ranked[:limit]]
    return ordered, duplicate_count, removed_by_cap


def numeric_value(record: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def original_turn_identifier(turn: dict[str, Any], fallback: int) -> Any:
    for key in ("turn_idx", "turn_id", "id"):
        if turn.get(key) is not None:
            return turn[key]
    return fallback


def merge_turn_group(
    group: list[tuple[int, dict[str, Any]]],
    *,
    max_statements: int,
) -> tuple[dict[str, Any], int, int]:
    # Last-component metadata best represents the state immediately before the
    # next speaker responds. Structural fields are recomputed below.
    merged = copy.deepcopy(group[-1][1])
    texts = [turn_text(record) for _, record in group]
    combined_text = " ".join(text for text in texts if text)
    merged["turn_text"] = combined_text
    if any("transcript" in record for _, record in group):
        merged["transcript"] = combined_text
    if any("turnText" in record for _, record in group):
        merged["turnText"] = combined_text

    duplicate_total = 0
    capped_total = 0
    for key in ANNOTATION_KEYS:
        if not any(key in record for _, record in group):
            continue
        values: list[Any] = []
        for _, record in group:
            values.extend(annotation_values(record.get(key)))
        cleaned, duplicates, capped = clean_annotations(values, limit=max_statements)
        merged[key] = cleaned
        duplicate_total += duplicates
        capped_total += capped

    starts = [
        value
        for _, record in group
        if (value := numeric_value(record, START_TIME_KEYS)) is not None
    ]
    ends = [
        value
        for _, record in group
        if (value := numeric_value(record, END_TIME_KEYS)) is not None
    ]
    durations = [
        value
        for _, record in group
        if (value := numeric_value(record, ("duration",))) is not None and value >= 0
    ]
    if starts:
        start = min(starts)
        for key in START_TIME_KEYS:
            if any(key in record for _, record in group):
                merged[key] = start
    if ends:
        end = max(ends)
        for key in END_TIME_KEYS:
            if any(key in record for _, record in group):
                merged[key] = end
    duration = sum(durations) if durations else None
    if duration is not None:
        merged["duration"] = duration

    count = word_count(combined_text)
    merged["wordCount"] = count
    merged["word_count"] = count
    if duration is not None and duration > 0:
        merged["words_per_second"] = count / duration

    merged["merged_from_turn_indices"] = [
        original_turn_identifier(record, position)
        for position, record in group
    ]
    if len(group) > 1:
        merged["merged_turn_count"] = len(group)
    else:
        merged.pop("merged_turn_count", None)
    return merged, duplicate_total, capped_total


def extract_episode_root(value: Any) -> tuple[str, str | None, dict[str, Any] | None, list[Any]] | None:
    if isinstance(value, list):
        return "list", None, None, value
    if not isinstance(value, dict):
        return None
    for key in TURN_LIST_KEYS:
        turns = value.get(key)
        if isinstance(turns, list):
            metadata = copy.deepcopy(value)
            metadata.pop(key, None)
            return "object", key, metadata, turns
    return None


def clean_episode(
    value: Any,
    *,
    min_words: int,
    max_statements: int,
) -> CleanedEpisode | None:
    extracted = extract_episode_root(value)
    if extracted is None:
        return None
    root_kind, container_key, root_metadata, raw_turns = extracted
    stats = CleaningStats(original_turns=len(raw_turns))
    retained: list[tuple[int, dict[str, Any]]] = []
    for position, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            stats.missing_text_turns_removed += 1
            continue
        text = turn_text(raw_turn)
        if not text:
            stats.missing_text_turns_removed += 1
            continue
        if word_count(text) < min_words:
            stats.short_turns_removed += 1
            continue
        speaker = speaker_value(raw_turn)
        if not speaker:
            raise EpisodeRejected(
                "missing_speaker_after_length_filter",
                f"Turn at source position {position} has no speaker identity",
            )
        retained_turn = copy.deepcopy(raw_turn)
        retained_turn["turn_text"] = text
        retained.append((position, retained_turn))

    speakers = list(dict.fromkeys(speaker_value(turn) for _, turn in retained))
    if len(speakers) != 2:
        raise EpisodeRejected(
            "not_exactly_two_speakers_after_filter",
            f"Found {len(speakers)} speaker(s): {speakers}",
        )

    groups: list[list[tuple[int, dict[str, Any]]]] = []
    for item in retained:
        if groups and speaker_value(groups[-1][-1][1]) == speaker_value(item[1]):
            groups[-1].append(item)
        else:
            groups.append([item])

    merged_turns: list[dict[str, Any]] = []
    for group in groups:
        merged, duplicates, capped = merge_turn_group(
            group,
            max_statements=max_statements,
        )
        merged_turns.append(merged)
        stats.annotation_duplicates_removed += duplicates
        stats.annotation_items_removed_by_cap += capped
        if len(group) > 1:
            stats.merge_groups += 1
            stats.turns_absorbed_by_merging += len(group) - 1

    final_speakers = [speaker_value(turn) for turn in merged_turns]
    if len(set(final_speakers)) != 2 or any(
        left == right for left, right in zip(final_speakers, final_speakers[1:])
    ):
        raise EpisodeRejected("abab_validation_failed")
    for index, turn in enumerate(merged_turns):
        turn["turn_idx"] = index
        if word_count(turn_text(turn)) < min_words:
            raise EpisodeRejected("minimum_word_validation_failed")
        for key in ANNOTATION_KEYS:
            if len(annotation_values(turn.get(key))) > max_statements:
                raise EpisodeRejected("annotation_cap_validation_failed", key)

    stats.final_turns = len(merged_turns)
    return CleanedEpisode(
        root_kind=root_kind,
        turn_container_key=container_key,
        root_metadata=root_metadata,
        turns=merged_turns,
        speakers=list(dict.fromkeys(final_speakers)),
        stats=stats,
    )


def serialize_episode(episode: CleanedEpisode) -> Any:
    if episode.root_kind == "list":
        return episode.turns
    output = copy.deepcopy(episode.root_metadata or {})
    output[episode.turn_container_key or "turns"] = episode.turns
    return output


def episode_fingerprint(episode: CleanedEpisode) -> str:
    speaker_map: dict[str, str] = {}
    sequence: list[list[str]] = []
    for turn in episode.turns:
        speaker = speaker_value(turn)
        if speaker not in speaker_map:
            speaker_map[speaker] = chr(ord("A") + len(speaker_map))
        sequence.append(
            [
                speaker_map[speaker],
                normalize_text(turn_text(turn)).casefold(),
            ]
        )
    encoded = json.dumps(sequence, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dataset_key(relative_path: Path) -> str:
    parts = relative_path.parts
    if not parts:
        return "root"
    if len(parts) == 1:
        return "root_metadata"
    if len(parts) >= 3 and parts[1] == "parsed":
        return "raw_parsed"
    if len(parts) >= 3 and parts[1] == "raw":
        return "extraction_raw_outputs"
    if parts[0] == "stance_labeled" and len(parts) >= 3:
        return f"stance_labeled/{parts[1]}"
    if parts[0] == "implicature_flow" and len(parts) >= 3:
        return f"implicature_flow/{parts[1]}"
    return parts[0]


def likely_copy_suffix(path: Path) -> bool:
    stem = path.stem
    return bool(re.search(r"(?:_copy|_duplicate|\(copy\)|_[2-9]\d*)$", stem, re.IGNORECASE))


def input_json_files(input_root: Path) -> list[Path]:
    return sorted(
        input_root.rglob("*.json"),
        key=lambda path: (
            dataset_key(path.relative_to(input_root)),
            likely_copy_suffix(path),
            path.relative_to(input_root).as_posix().casefold(),
        ),
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl_row(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_words < 1:
        raise ValueError("--min-words must be at least 1")
    if args.max_statements < 1:
        raise ValueError("--max-statements must be at least 1")

    input_root = canonical_path(args.input_root)
    output_root = canonical_path(args.output_root)
    validate_roots(input_root, output_root)
    if output_root.exists() and args.overwrite:
        LOG.info("Removing existing output tree: %s", output_root)
    else:
        LOG.info("Preparing output tree: %s", output_root)
    prepare_output_root(output_root, overwrite=args.overwrite)
    LOG.info("Output tree is ready: %s", output_root)

    LOG.info("Scanning JSON files under %s", input_root)
    files = input_json_files(input_root)
    LOG.info("Discovered %d JSON files; cleaning into %s", len(files), output_root)
    dataset_stats: defaultdict[str, DatasetStats] = defaultdict(DatasetStats)
    kept_by_fingerprint: dict[tuple[str, str], str] = {}
    overall_status = Counter()
    audit_path = output_root / AUDIT_NAME

    with audit_path.open("w", encoding="utf-8") as audit_handle:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(input_root)
            if args.verbose:
                LOG.info("Processing %d/%d: %s", index, len(files), relative)
            key = dataset_key(relative)
            stats = dataset_stats[key]
            stats.json_files_seen += 1
            audit: dict[str, Any] = {
                "input_file": relative.as_posix(),
                "dataset_key": key,
            }
            try:
                value = read_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                status = "invalid_json"
                stats.skipped_files[status] += 1
                audit.update(status=status, error=str(exc))
                write_jsonl_row(audit_handle, audit)
                overall_status[status] += 1
                continue

            try:
                episode = clean_episode(
                    value,
                    min_words=args.min_words,
                    max_statements=args.max_statements,
                )
            except EpisodeRejected as exc:
                stats.episode_files_detected += 1
                stats.skipped_files[exc.reason] += 1
                audit.update(status=exc.reason, detail=exc.detail)
                write_jsonl_row(audit_handle, audit)
                overall_status[exc.reason] += 1
                continue
            if episode is None:
                status = "non_episode_json"
                stats.skipped_files[status] += 1
                audit.update(status=status)
                write_jsonl_row(audit_handle, audit)
                overall_status[status] += 1
                continue

            stats.episode_files_detected += 1
            fingerprint = episode_fingerprint(episode)
            duplicate_key = (key, fingerprint)
            kept = kept_by_fingerprint.get(duplicate_key)
            if kept is not None:
                status = "duplicate_episode_removed"
                stats.duplicate_files_removed += 1
                stats.skipped_files[status] += 1
                audit.update(
                    status=status,
                    duplicate_of=kept,
                    fingerprint_sha256=fingerprint,
                    cleaning=asdict(episode.stats),
                )
                write_jsonl_row(audit_handle, audit)
                overall_status[status] += 1
                continue

            output_path = output_root / relative
            write_json(output_path, serialize_episode(episode))
            kept_by_fingerprint[duplicate_key] = relative.as_posix()
            stats.files_written += 1
            stats.cleaning.add(episode.stats)
            status = "written"
            audit.update(
                status=status,
                output_file=relative.as_posix(),
                fingerprint_sha256=fingerprint,
                speakers=episode.speakers,
                cleaning=asdict(episode.stats),
            )
            write_jsonl_row(audit_handle, audit)
            overall_status[status] += 1

    manifest = {
        "script": "deduplicate_data.py",
        "script_version": SCRIPT_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "configuration": {
            "min_words": args.min_words,
            "max_statements": args.max_statements,
            "turn_filter_order": "minimum_words_then_merge_adjacent_same_speaker",
            "duplicate_identity": "dataset_key_plus_cleaned_normalized_speaker_text_sha256",
            "merge_metadata_policy": "last_component_metadata_with_recomputed_structural_fields",
            "turn_provenance_policy": "original_index_list_on_every_cleaned_turn",
            "speaker_policy": "skip_episode_unless_exactly_two_speakers_remain",
        },
        "input_json_file_count": len(files),
        "status_counts": dict(sorted(overall_status.items())),
        "dataset_summaries": {
            key: stats.to_json()
            for key, stats in sorted(dataset_stats.items())
        },
        "audit_jsonl": AUDIT_NAME,
    }
    write_json(output_root / MANIFEST_NAME, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    try:
        manifest = run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    counts = manifest["status_counts"]
    LOG.info(
        "Complete: written=%d duplicates_removed=%d skipped=%d output=%s",
        counts.get("written", 0),
        counts.get("duplicate_episode_removed", 0),
        sum(value for key, value in counts.items() if key not in {"written", "duplicate_episode_removed"}),
        manifest["output_root"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
