from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from ..common.candidates import build_anchors, validate_anchor
from ..common.data import episode_input_hash, load_show_map, normalize_episode
from ..common.splits import assert_show_disjoint, assign_show_splits, balanced_anchor_sample
from ..common.utils import (
    file_hash,
    list_episode_paths,
    make_manifest,
    manifest_matches,
    patch_directory,
    read_json,
    read_jsonl,
    shard_slice,
    stable_hash,
    validate_patch_manifests,
    write_json,
    write_jsonl,
)

STAGE = "exp8_prepare_data"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Prepare show-disjoint Exp8 pilot data.")
    value.add_argument("--mode", choices=("worker", "merge", "count"), required=True)
    value.add_argument("--input-dir", type=Path, default=Path("data/conversation_moves_labeled"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--show-map", type=Path)
    value.add_argument("--allow-episode-fallback", action="store_true")
    value.add_argument("--num-patches", type=int, default=1)
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--episodes-per-task", type=int, default=250)
    value.add_argument("--candidate-count", type=int, default=25)
    value.add_argument("--development-limit", type=int, default=10000)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    return value


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_dir": str(args.input_dir),
        "show_map": str(args.show_map) if args.show_map else None,
        "allow_episode_fallback": bool(args.allow_episode_fallback),
        "episodes_per_task": int(args.episodes_per_task),
        "candidate_count": int(args.candidate_count),
        "development_limit": int(args.development_limit),
        "seed": int(args.seed),
    }


def count(args: argparse.Namespace) -> None:
    total = len(list_episode_paths(args.input_dir))
    patches = math.ceil(total / args.episodes_per_task) if total else 0
    print(f"TOTAL_ITEMS={total}")
    print(f"NUM_PATCHES={patches}")


def worker(args: argparse.Namespace) -> None:
    paths = list_episode_paths(args.input_dir)
    if not paths:
        raise RuntimeError(f"No episode JSON files found under {args.input_dir}")
    if args.num_patches != math.ceil(len(paths) / args.episodes_per_task):
        raise ValueError("--num-patches does not match the current input size")
    input_hash = episode_input_hash(args.input_dir)
    config = configuration(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, args.num_patches)
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=args.num_patches,
        row_count=0,
        input_hash=input_hash,
        split_hash="unassigned",
        config=config,
    )
    manifest_path = patch_dir / "patch_manifest.json"
    if not args.force and manifest_matches(manifest_path, expected):
        print(f"Reusing completed patch {patch_dir}")
        return
    selected = paths[shard_slice(len(paths), args.patch_index, args.episodes_per_task)]
    show_map = load_show_map(args.show_map)
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in selected:
        try:
            episodes.append(
                normalize_episode(
                    path,
                    show_map=show_map,
                    allow_episode_fallback=args.allow_episode_fallback,
                )
            )
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    if errors and not episodes:
        raise RuntimeError(f"Every episode in patch {args.patch_index} failed normalization; first error: {errors[0]}")
    patch_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(patch_dir / "episodes.jsonl", episodes)
    write_jsonl(patch_dir / "errors.jsonl", errors)
    manifest = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=args.num_patches,
        row_count=len(episodes),
        input_hash=input_hash,
        split_hash="unassigned",
        config=config,
        extra={"error_count": len(errors), "source_file_count": len(selected)},
    )
    write_json(manifest_path, manifest)


def merge(args: argparse.Namespace) -> None:
    manifests = validate_patch_manifests(args.output_dir, STAGE, args.num_patches)
    episodes: list[dict[str, Any]] = []
    for index in range(args.num_patches):
        episodes.extend(read_jsonl(patch_directory(args.output_dir, index, args.num_patches) / "episodes.jsonl"))
    keys = [str(episode["episode_key"]) for episode in episodes]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate episode keys found while merging preparation patches")
    assignments = assign_show_splits(episodes, args.seed)
    split_rows = [
        {"show_id": show_id, "split": split, "split_hash_key": stable_hash({"show_id": show_id, "seed": args.seed})}
        for show_id, split in sorted(assignments.items())
    ]
    split_hash = stable_hash(split_rows)
    turns = [turn for episode in episodes for turn in episode["turns"]]
    turn_ids = [str(turn["turn_id"]) for turn in turns]
    if len(turn_ids) != len(set(turn_ids)):
        raise RuntimeError("Duplicate turn IDs found while merging preparation patches")
    anchors = build_anchors(episodes, assignments, args.candidate_count)
    for anchor in anchors:
        validate_anchor(anchor)
    assert_show_disjoint(anchors)
    development = balanced_anchor_sample(
        anchors,
        split="validation",
        limit=args.development_limit,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "episodes.jsonl", episodes)
    write_jsonl(args.output_dir / "turns.jsonl", turns)
    write_jsonl(args.output_dir / "anchors.jsonl", anchors)
    write_jsonl(args.output_dir / "development_anchors.jsonl", development)
    write_jsonl(args.output_dir / "split_manifest.jsonl", split_rows)
    summary = {
        "stage": STAGE,
        "input_hash": manifests[0]["input_hash"],
        "split_hash": split_hash,
        "config": configuration(args),
        "episode_count": len(episodes),
        "turn_count": len(turns),
        "anchor_count": len(anchors),
        "development_anchor_count": len(development),
        "show_count": len(assignments),
        "error_count": sum(int(manifest.get("error_count", 0)) for manifest in manifests),
        "split_counts": {
            split: sum(1 for value in assignments.values() if value == split)
            for split in ("train", "validation", "test")
        },
        "outputs": {
            "episodes": str(args.output_dir / "episodes.jsonl"),
            "turns": str(args.output_dir / "turns.jsonl"),
            "anchors": str(args.output_dir / "anchors.jsonl"),
            "development_anchors": str(args.output_dir / "development_anchors.jsonl"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)


def main() -> None:
    args = parser().parse_args()
    if args.episodes_per_task < 1 or args.num_patches < 1:
        raise ValueError("Patch sizes and counts must be positive")
    if args.mode == "count":
        count(args)
    elif args.mode == "worker":
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()

