#!/usr/bin/env python3
"""Copy one representative JSON file from each dataset directory.

The repository's data products are commonly arranged as directories such as::

    data/<category>/<stage>/*.json

This script selects one data JSON from every directory containing JSON files,
copies it below ``examples/`` while preserving its relative path, and writes a
manifest describing the detected JSON structure. Metadata files such as
``manifest_by_episode.json`` are ignored by default because they do not expose
the turn schema that downstream preprocessing needs to inspect.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger("extract-json-examples")

DEFAULT_INPUT_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("examples")
MANIFEST_NAME = "examples_manifest.json"

METADATA_FILENAMES = {
    "ai_episode_names.json",
    "conversation_moves_labeled_host_guest_summary.json",
    "episode_names.json",
    "manifest.json",
    "manifest_by_category.json",
    "manifest_by_episode.json",
    "summary.json",
}

TURN_TEXT_KEYS = ("turn_text", "turnText", "transcript", "text")
TURN_LIST_KEYS = ("turns", "records", "utterances")


@dataclass(frozen=True)
class JsonDescription:
    json_root_type: str
    detected_kind: str
    turn_container_key: str | None
    turn_count: int | None
    top_level_keys: list[str]
    first_turn_keys: list[str]


@dataclass(frozen=True)
class ExampleRecord:
    source_directory: str
    source_file: str
    example_file: str
    json_root_type: str
    detected_kind: str
    turn_container_key: str | None
    turn_count: int | None
    top_level_keys: list[str]
    first_turn_keys: list[str]


@dataclass(frozen=True)
class DirectoryRecord:
    directory: str
    direct_json_count: int
    status: str
    selected_example: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one representative JSON from every JSON-containing directory "
            "under a data root into examples/."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Dataset root to scan recursively (default: data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination root for copied examples (default: examples).",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help=(
            "Allow manifest/summary JSON files to be selected when a directory "
            "contains no episode-like JSON."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing copied example at the same destination path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed selection and skip logging.",
    )
    return parser.parse_args(argv)


def canonical_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def looks_like_turn(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(isinstance(value.get(key), str) for key in TURN_TEXT_KEYS)


def find_turn_list(value: Any) -> tuple[str | None, list[Any] | None]:
    if isinstance(value, list):
        return None, value
    if isinstance(value, dict):
        for key in TURN_LIST_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, list):
                return key, candidate
    return None, None


def describe_json(value: Any) -> JsonDescription:
    if isinstance(value, list):
        root_type = "list"
        top_level_keys: list[str] = []
    elif isinstance(value, dict):
        root_type = "object"
        top_level_keys = sorted(str(key) for key in value)
    else:
        return JsonDescription(
            json_root_type=type(value).__name__,
            detected_kind="other",
            turn_container_key=None,
            turn_count=None,
            top_level_keys=[],
            first_turn_keys=[],
        )

    container_key, turns = find_turn_list(value)
    first_turn = next((turn for turn in turns or [] if isinstance(turn, dict)), None)
    first_turn_keys = sorted(str(key) for key in first_turn) if first_turn else []
    episode_like = bool(turns) and any(looks_like_turn(turn) for turn in turns or [])
    return JsonDescription(
        json_root_type=root_type,
        detected_kind="episode" if episode_like else "metadata_or_other",
        turn_container_key=container_key,
        turn_count=len(turns) if turns is not None else None,
        top_level_keys=top_level_keys,
        first_turn_keys=first_turn_keys,
    )


def load_and_describe(path: Path) -> JsonDescription:
    with path.open("r", encoding="utf-8-sig") as handle:
        return describe_json(json.load(handle))


def is_known_metadata(path: Path) -> bool:
    name = path.name.casefold()
    return name in METADATA_FILENAMES or "manifest" in name or "summary" in name


def has_unsuffixed_sibling(path: Path, siblings: set[str]) -> bool:
    """Identify obvious ``name_2.json`` copies without penalizing lone names."""
    stem = path.stem
    head, separator, suffix = stem.rpartition("_")
    if not separator or not suffix.isdigit() or int(suffix) < 2:
        return False
    return f"{head}{path.suffix}".casefold() in siblings


def candidate_order(paths: Iterable[Path]) -> list[Path]:
    rows = list(paths)
    sibling_names = {path.name.casefold() for path in rows}
    return sorted(
        rows,
        key=lambda path: (
            is_known_metadata(path),
            has_unsuffixed_sibling(path, sibling_names),
            path.name.casefold(),
        ),
    )


def choose_example(
    directory: Path,
    *,
    include_metadata: bool,
) -> tuple[Path, JsonDescription] | None:
    candidates = candidate_order(directory.glob("*.json"))
    metadata_fallback: tuple[Path, JsonDescription] | None = None
    for path in candidates:
        try:
            description = load_and_describe(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            LOG.warning("Skipping unreadable JSON %s: %s", path, exc)
            continue
        if description.detected_kind == "episode":
            return path, description
        if include_metadata and metadata_fallback is None:
            metadata_fallback = (path, description)
    return metadata_fallback


def all_data_directories(input_root: Path, output_root: Path) -> list[Path]:
    directories = [input_root]
    directories.extend(
        path
        for path in input_root.rglob("*")
        if path.is_dir() and not is_relative_to(path.resolve(), output_root)
    )
    return sorted(set(directories), key=lambda path: path.as_posix().casefold())


def copy_examples(args: argparse.Namespace) -> list[ExampleRecord]:
    input_root = canonical_path(args.input_root)
    output_root = canonical_path(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")
    if input_root == output_root:
        raise ValueError("Input and output roots must be different directories")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[ExampleRecord] = []
    directory_records: list[DirectoryRecord] = []
    for directory in all_data_directories(input_root, output_root):
        direct_json_files = sorted(directory.glob("*.json"))
        relative_directory = directory.relative_to(input_root).as_posix() or "."
        if not direct_json_files:
            directory_records.append(
                DirectoryRecord(
                    directory=relative_directory,
                    direct_json_count=0,
                    status="no_json_files_directly_in_directory",
                    selected_example=None,
                )
            )
            LOG.info("Directory %s: no direct JSON files", relative_directory)
            continue
        chosen = choose_example(directory, include_metadata=args.include_metadata)
        if chosen is None:
            directory_records.append(
                DirectoryRecord(
                    directory=relative_directory,
                    direct_json_count=len(direct_json_files),
                    status="no_episode_like_json",
                    selected_example=None,
                )
            )
            LOG.info(
                "Directory %s: %d JSON file(s), but none is episode-like",
                relative_directory,
                len(direct_json_files),
            )
            continue
        source, description = chosen
        relative_path = directory.relative_to(input_root)
        destination = output_root / relative_path / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Example already exists: {destination}. Re-run with --overwrite to replace it."
            )
        shutil.copy2(source, destination)
        records.append(
            ExampleRecord(
                source_directory=directory.relative_to(input_root).as_posix() or ".",
                source_file=source.relative_to(input_root).as_posix(),
                example_file=destination.relative_to(output_root).as_posix(),
                **asdict(description),
            )
        )
        directory_records.append(
            DirectoryRecord(
                directory=relative_directory,
                direct_json_count=len(direct_json_files),
                status="example_selected",
                selected_example=destination.relative_to(output_root).as_posix(),
            )
        )
        LOG.info("Selected %s -> %s", source, destination)

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "include_metadata": bool(args.include_metadata),
        "directory_count": len(directory_records),
        "directories": [asdict(record) for record in directory_records],
        "sample_count": len(records),
        "samples": [asdict(record) for record in records],
    }
    manifest_path = output_root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        records = copy_examples(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    LOG.info("Copied %d example JSON file(s) to %s", len(records), args.output_root)
    if not records:
        LOG.warning(
            "No episode-like JSON files were found. If this checkout contains only "
            "manifests, run the script where the full dataset is mounted; use "
            "--include-metadata only when metadata samples are desired."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
