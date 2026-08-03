from __future__ import annotations

"""Stage 05: evaluate counterfactual assumption controls."""

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.exp8_assumption_embedding_pilot.common.controls import build_control_map
from experiments.exp8_assumption_embedding_pilot.common.embeddings import EmbeddingStore
from experiments.exp8_assumption_embedding_pilot.common.evaluation import evaluate_anchor
from experiments.exp8_assumption_embedding_pilot.common.metrics import aggregate_rows, clustered_delta_interval
from experiments.exp8_assumption_embedding_pilot.common.progress import run_parallel, run_single
from experiments.exp8_assumption_embedding_pilot.common.utils import (
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

STAGE = "exp8_exp04_counterfactual_controls"
CONTROL_TYPES = ("same_episode", "same_category", "explicit_matched")
ROW_COLUMNS = (
    "anchor_id", "condition", "control_type", "category", "show_id", "episode_id",
    "target_id", "candidate_count", "candidate_pool_hash", "rank", "reciprocal_rank",
    "top1", "top5", "margin",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Evaluate counterfactual controls in anchor shards.")
    value.add_argument("--mode", choices=("local", "worker", "merge"), default="local")
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--cache-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp04_results"))
    value.add_argument("--anchors-per-task", type=int, default=1000)
    value.add_argument("--num-patches", type=int, default=1)
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--bootstrap-draws", type=int, default=1000)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    value.add_argument("--jobs", type=int, default=8)
    return value


def counts(args: argparse.Namespace) -> tuple[int, int]:
    anchors = sum(1 for _ in read_jsonl(args.data_dir / "development_anchors.jsonl"))
    shards = math.ceil(anchors / args.anchors_per_task) if anchors else 0
    return anchors, shards


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "anchors_per_task": args.anchors_per_task,
        "control_types": list(CONTROL_TYPES),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }


def execute(args: argparse.Namespace) -> None:
    total_anchors, anchor_shards = counts(args)
    total_patches = anchor_shards * len(CONTROL_TYPES)
    if args.mode == "merge":
        manifests = validate_patch_manifests(args.output_dir, STAGE, total_patches)
        frames = [pd.read_csv(patch_directory(args.output_dir, index, total_patches) / "rows.csv") for index in range(total_patches)]
        frame = pd.concat(frames, ignore_index=True)
        if frame.duplicated(["anchor_id", "control_type", "condition"]).any():
            raise RuntimeError("Duplicate Exp04 anchor/control/condition rows")
        mixed_pools = frame.groupby("anchor_id")["candidate_pool_hash"].nunique()
        if (mixed_pools != 1).any():
            raise RuntimeError("Exp04 controls did not preserve the original candidate pool")
        rows = frame.to_dict("records")
        metrics = []
        deltas = []
        for control_type in CONTROL_TYPES:
            subset = [row for row in rows if row["control_type"] == control_type]
            for metric in aggregate_rows(subset):
                metrics.append({"control_type": control_type, **metric})
            deltas.append(
                {
                    "control_type": control_type,
                    **clustered_delta_interval(
                        subset,
                        "correct",
                        "control",
                        draws=args.bootstrap_draws,
                        seed=args.seed,
                    ),
                }
            )
        frame.to_csv(args.output_dir / "rows.csv", index=False)
        pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
        write_json(args.output_dir / "config.json", {**configuration(args), "config_hash": manifests[0]["config_hash"]})
        write_json(
            args.output_dir / "summary.json",
            {
                "experiment": "exp04_counterfactual_controls",
                "status": "complete",
                "input_hash": manifests[0]["input_hash"],
                "split_hash": manifests[0]["split_hash"],
                "metrics": metrics,
                "paired_deltas": deltas,
            },
        )
        return
    if args.num_patches != total_patches:
        raise ValueError(f"Expected {total_patches} total patches, got {args.num_patches}")
    control_index = args.patch_index // anchor_shards
    anchor_shard = args.patch_index % anchor_shards
    control_type = CONTROL_TYPES[control_index]
    anchors_path = args.data_dir / "development_anchors.jsonl"
    anchors = list(read_jsonl(anchors_path))
    selected = anchors[shard_slice(len(anchors), anchor_shard, args.anchors_per_task)]
    cache_index = read_json(args.cache_dir / "cache_index.json")
    input_hash = stable_hash({"anchors": file_hash(anchors_path), "cache": cache_index})
    configuration_value = configuration(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, total_patches)
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=total_patches,
        row_count=0,
        input_hash=input_hash,
        split_hash=str(cache_index["split_hash"]),
        config=configuration_value,
    )
    if not args.force and manifest_matches(patch_dir / "patch_manifest.json", expected):
        return
    turns = list(read_jsonl(args.data_dir / "turns.jsonl"))
    source_ids = {str(anchor["anchor_id"]) for anchor in selected}
    donor_map = build_control_map(turns, control_type, source_ids)
    store = EmbeddingStore(args.cache_dir / "cache_index.json")
    rows: list[dict[str, Any]] = []
    for anchor in selected:
        source_id = str(anchor["anchor_id"])
        donor_id = donor_map.get(source_id)
        if donor_id is None:
            continue
        correct = evaluate_anchor(anchor, store, "full")
        correct["condition"] = "correct"
        correct["control_type"] = control_type
        control = evaluate_anchor(
            anchor,
            store,
            "control",
            donor_id=None if control_type == "explicit_matched" else donor_id,
            explicit_as_assumption=control_type == "explicit_matched",
        )
        control["condition"] = "control"
        control["control_type"] = control_type
        rows.extend([correct, control])
    patch_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ROW_COLUMNS).to_csv(patch_dir / "rows.csv", index=False)
    write_json(
        patch_dir / "patch_manifest.json",
        make_manifest(
            stage=STAGE,
            patch_index=args.patch_index,
            num_patches=total_patches,
            row_count=len(rows),
            input_hash=input_hash,
            split_hash=str(cache_index["split_hash"]),
            config=configuration_value,
            extra={"control_type": control_type, "anchor_shard": anchor_shard},
        ),
    )


def local(args: argparse.Namespace) -> None:
    _, anchor_shards = counts(args)
    patches = anchor_shards * len(CONTROL_TYPES)
    if patches < 1:
        raise RuntimeError("Stage 05 has no development anchors")

    def run_patch(index: int) -> None:
        child = argparse.Namespace(**vars(args))
        child.mode = "worker"
        child.patch_index = index
        child.num_patches = patches
        execute(child)

    run_parallel(range(patches), run_patch, args.jobs, "stage 05 controls")
    merged = argparse.Namespace(**vars(args))
    merged.mode = "merge"
    merged.num_patches = patches
    run_single(lambda: execute(merged), "stage 05 merge")


def main() -> None:
    args = parser().parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.mode == "local":
        local(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
