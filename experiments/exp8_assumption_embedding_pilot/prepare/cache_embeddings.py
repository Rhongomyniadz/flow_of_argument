from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..common.embeddings import DEFAULT_INSTRUCTION, TextEmbedder, save_embedding_patch
from ..common.utils import (
    file_hash,
    make_manifest,
    manifest_matches,
    patch_directory,
    read_json,
    read_jsonl,
    stable_hash,
    validate_patch_manifests,
    write_json,
)

STAGE = "exp8_cache_embeddings"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Cache Qwen or deterministic smoke embeddings by episode shard.")
    value.add_argument("--mode", choices=("worker", "merge", "count"), required=True)
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--model-name", default="Qwen/Qwen3-Embedding-4B")
    value.add_argument("--model-revision", default="main")
    value.add_argument("--backend", choices=("sentence_transformer", "hash"), default="sentence_transformer")
    value.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--device")
    value.add_argument("--hash-dim", type=int, default=64)
    value.add_argument("--episodes-per-task", type=int, default=50)
    value.add_argument("--num-patches", type=int, default=1)
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--force", action="store_true")
    return value


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "backend": args.backend,
        "instruction": args.instruction,
        "batch_size": args.batch_size,
        "hash_dim": args.hash_dim,
        "episodes_per_task": args.episodes_per_task,
    }


def episode_count(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def count(args: argparse.Namespace) -> None:
    total = episode_count(args.data_dir / "episodes.jsonl")
    print(f"TOTAL_ITEMS={total}")
    print(f"NUM_PATCHES={math.ceil(total / args.episodes_per_task) if total else 0}")


def worker(args: argparse.Namespace) -> None:
    episodes_path = args.data_dir / "episodes.jsonl"
    data_summary = read_json(args.data_dir / "summary.json")
    total = int(data_summary["episode_count"])
    expected_patches = math.ceil(total / args.episodes_per_task)
    if args.num_patches != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {args.num_patches}")
    input_hash = file_hash(episodes_path)
    split_hash = str(data_summary["split_hash"])
    config = configuration(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, args.num_patches)
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=args.num_patches,
        row_count=0,
        input_hash=input_hash,
        split_hash=split_hash,
        config=config,
    )
    manifest_path = patch_dir / "patch_manifest.json"
    if not args.force and manifest_matches(manifest_path, expected):
        print(f"Reusing completed embedding patch {patch_dir}")
        return
    start = args.patch_index * args.episodes_per_task
    episodes = list(itertools.islice(read_jsonl(episodes_path), start, start + args.episodes_per_task))
    turns = [turn for episode in episodes for turn in episode["turns"]]
    if not turns:
        raise RuntimeError(f"Embedding patch {args.patch_index} contains no turns")
    embedder = TextEmbedder(
        args.model_name,
        model_revision=args.model_revision,
        backend=args.backend,
        instruction=args.instruction,
        batch_size=args.batch_size,
        device=args.device,
        hash_dim=args.hash_dim,
    )
    patch_dir.mkdir(parents=True, exist_ok=True)
    output_path = patch_dir / "embeddings.npz"
    row_count = save_embedding_patch(output_path, turns, embedder)
    with np.load(output_path, allow_pickle=False) as data:
        dimension = int(data["query"].shape[1])
    manifest = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=args.num_patches,
        row_count=row_count,
        input_hash=input_hash,
        split_hash=split_hash,
        config=config,
        extra={"episode_count": len(episodes), "dimension": dimension, "embedding_file": str(output_path)},
    )
    write_json(manifest_path, manifest)


def merge(args: argparse.Namespace) -> None:
    manifests = validate_patch_manifests(args.output_dir, STAGE, args.num_patches)
    dimensions = {int(manifest["dimension"]) for manifest in manifests}
    if len(dimensions) != 1:
        raise RuntimeError(f"Mixed embedding dimensions: {sorted(dimensions)}")
    seen: set[str] = set()
    patch_files: list[str] = []
    for index in range(args.num_patches):
        path = patch_directory(args.output_dir, index, args.num_patches) / "embeddings.npz"
        with np.load(path, allow_pickle=False) as data:
            ids = [str(value) for value in data["turn_ids"].tolist()]
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError(f"Duplicate embedded turn IDs; first duplicate: {sorted(overlap)[0]}")
        seen.update(ids)
        patch_files.append(os.path.relpath(path, args.output_dir))
    index = {
        "stage": STAGE,
        "input_hash": manifests[0]["input_hash"],
        "split_hash": manifests[0]["split_hash"],
        "config_hash": manifests[0]["config_hash"],
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "backend": args.backend,
        "instruction": args.instruction,
        "dimension": dimensions.pop(),
        "turn_count": len(seen),
        "patch_files": patch_files,
    }
    write_json(args.output_dir / "cache_index.json", index)
    write_json(args.output_dir / "summary.json", index)


def main() -> None:
    args = parser().parse_args()
    if args.episodes_per_task < 1 or args.batch_size < 1:
        raise ValueError("Batch and shard sizes must be positive")
    if args.mode == "count":
        count(args)
    elif args.mode == "worker":
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
