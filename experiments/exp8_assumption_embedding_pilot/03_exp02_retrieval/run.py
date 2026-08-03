from __future__ import annotations

"""Stage 03: evaluate frozen next-turn retrieval."""

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.controls import build_control_map
from ..common.embeddings import EmbeddingStore, load_turn_lookup
from ..common.evaluation import FROZEN_CONDITIONS, evaluate_anchor
from ..common.metrics import aggregate_rows, clustered_delta_interval
from ..common.progress import run_parallel, run_single
from ..common.utils import (
    file_hash,
    make_manifest,
    manifest_matches,
    patch_directory,
    read_json,
    read_jsonl,
    shard_slice,
    stable_hash,
    validate_patch_manifests,
    write_json,
)

STAGE = "exp8_exp02_frozen_retrieval"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run or merge frozen next-turn retrieval shards.")
    value.add_argument("--mode", choices=("local", "worker", "merge"), default="local")
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--cache-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp02_results"))
    value.add_argument("--anchors-per-task", type=int, default=1000)
    value.add_argument("--num-patches", type=int, default=1)
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--bootstrap-draws", type=int, default=1000)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    value.add_argument("--jobs", type=int, default=8)
    return value


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "anchors_per_task": args.anchors_per_task,
        "conditions": list(FROZEN_CONDITIONS),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }


def worker(args: argparse.Namespace) -> None:
    anchors_path = args.data_dir / "development_anchors.jsonl"
    anchors = list(read_jsonl(anchors_path))
    expected_patches = math.ceil(len(anchors) / args.anchors_per_task)
    if args.num_patches != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {args.num_patches}")
    cache_index = read_json(args.cache_dir / "cache_index.json")
    input_hash = stable_hash({"anchors": file_hash(anchors_path), "cache": cache_index})
    split_hash = str(cache_index["split_hash"])
    configuration = config(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, args.num_patches)
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=args.num_patches,
        row_count=0,
        input_hash=input_hash,
        split_hash=split_hash,
        config=configuration,
    )
    manifest_path = patch_dir / "patch_manifest.json"
    if not args.force and manifest_matches(manifest_path, expected):
        return
    selected = anchors[shard_slice(len(anchors), args.patch_index, args.anchors_per_task)]
    store = EmbeddingStore(args.cache_dir / "cache_index.json")
    turns = list(read_jsonl(args.data_dir / "turns.jsonl"))
    source_ids = {str(anchor["anchor_id"]) for anchor in selected}
    donors = build_control_map(turns, "same_episode", source_ids)
    rows: list[dict[str, Any]] = []
    for anchor in selected:
        for condition in FROZEN_CONDITIONS:
            donor = donors.get(str(anchor["anchor_id"])) if condition == "shuffled" else None
            if condition == "shuffled" and donor is None:
                continue
            rows.append(evaluate_anchor(anchor, store, condition, donor_id=donor))
    patch_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(patch_dir / "rows.csv", index=False)
    manifest = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=args.num_patches,
        row_count=len(rows),
        input_hash=input_hash,
        split_hash=split_hash,
        config=configuration,
        extra={"anchor_count": len(selected)},
    )
    write_json(manifest_path, manifest)


def merge(args: argparse.Namespace) -> None:
    manifests = validate_patch_manifests(args.output_dir, STAGE, args.num_patches)
    frames = [
        pd.read_csv(patch_directory(args.output_dir, index, args.num_patches) / "rows.csv")
        for index in range(args.num_patches)
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["anchor_id", "condition"]).any():
        raise RuntimeError("Duplicate anchor/condition rows in Exp02 merge")
    rows = frame.to_dict("records")
    metrics = aggregate_rows(rows)
    deltas = [
        clustered_delta_interval(rows, "full", "current_history", draws=args.bootstrap_draws, seed=args.seed),
        clustered_delta_interval(rows, "full", "shuffled", draws=args.bootstrap_draws, seed=args.seed + 1),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    frame.to_csv(args.output_dir / "rows.csv", index=False)
    write_json(args.output_dir / "config.json", {**config(args), "config_hash": manifests[0]["config_hash"]})
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp02_frozen_retrieval",
            "status": "complete",
            "input_hash": manifests[0]["input_hash"],
            "split_hash": manifests[0]["split_hash"],
            "anchor_count": int(frame["anchor_id"].nunique()),
            "metrics": metrics,
            "paired_deltas": deltas,
            "patch_count": args.num_patches,
        },
    )


def local(args: argparse.Namespace) -> None:
    total = sum(1 for _ in read_jsonl(args.data_dir / "development_anchors.jsonl"))
    patches = math.ceil(total / args.anchors_per_task) if total else 0
    if patches < 1:
        raise RuntimeError("Stage 03 has no development anchors")

    def run_patch(index: int) -> None:
        child = argparse.Namespace(**vars(args))
        child.mode = "worker"
        child.patch_index = index
        child.num_patches = patches
        worker(child)

    run_parallel(range(patches), run_patch, args.jobs, "stage 03 workers")
    merged = argparse.Namespace(**vars(args))
    merged.mode = "merge"
    merged.num_patches = patches
    run_single(lambda: merge(merged), "stage 03 merge")


def main() -> None:
    args = parser().parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.mode == "local":
        local(args)
    elif args.mode == "worker":
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
