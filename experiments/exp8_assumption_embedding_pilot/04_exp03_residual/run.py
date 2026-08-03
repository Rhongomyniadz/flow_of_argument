from __future__ import annotations

"""Stage 04: fit the three linear residual conditions."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.controls import build_control_map
from ..common.embeddings import EmbeddingStore, component_vectors
from ..common.metrics import normalize, rank_scores
from ..common.progress import run_parallel, run_single
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

STAGE = "exp8_exp03_linear_residual"
CONDITIONS = ("baseline", "full", "shuffled")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Fit one linear residual condition or merge all conditions.")
    value.add_argument("--mode", choices=("local", "worker", "merge"), default="local")
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--cache-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp03_results"))
    value.add_argument("--num-patches", type=int, default=3)
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--condition", choices=CONDITIONS)
    value.add_argument("--feature-dim", type=int, default=256)
    value.add_argument("--max-train-anchors", type=int, default=50000)
    value.add_argument("--ridge-alpha", type=float, default=10.0)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    value.add_argument("--jobs", type=int, default=3)
    return value


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "conditions": list(CONDITIONS),
        "feature_dim": args.feature_dim,
        "max_train_anchors": args.max_train_anchors,
        "ridge_alpha": args.ridge_alpha,
        "seed": args.seed,
    }


def features(
    anchor: dict[str, Any], store: EmbeddingStore, condition: str, dimension: int, donor: str | None
) -> np.ndarray:
    parts = component_vectors(anchor, store, assumption_source_id=donor, dimension=dimension)
    base = [parts["current"], parts["history"], parts["explicit"]]
    if condition in {"full", "shuffled"}:
        base.append(parts["assumption"])
    return np.concatenate([np.asarray(value, dtype=np.float32) for value in base])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_x = (x - mean) / scale
    augmented = np.column_stack([np.ones(len(normalized_x), dtype=np.float32), normalized_x])
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
    return weights.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def predict(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([np.ones(len(x), dtype=np.float32), (x - mean) / scale])
    return augmented @ weights


def execute(args: argparse.Namespace) -> None:
    if args.mode == "merge":
        manifests = validate_patch_manifests(args.output_dir, STAGE, len(CONDITIONS))
        observed = {str(manifest["condition"]) for manifest in manifests}
        if observed != set(CONDITIONS):
            raise RuntimeError(f"Expected conditions {CONDITIONS}, got {sorted(observed)}")
        frames = [
            pd.read_csv(patch_directory(args.output_dir, index, len(CONDITIONS)) / "metrics.csv")
            for index in range(len(CONDITIONS))
        ]
        metrics = pd.concat(frames, ignore_index=True)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        write_json(args.output_dir / "config.json", {**config(args), "config_hash": manifests[0]["config_hash"]})
        write_json(
            args.output_dir / "summary.json",
            {
                "experiment": "exp03_linear_residual",
                "status": "complete",
                "input_hash": manifests[0]["input_hash"],
                "split_hash": manifests[0]["split_hash"],
                "metrics": metrics.to_dict("records"),
            },
        )
        return

    condition = args.condition or CONDITIONS[args.patch_index]
    if condition != CONDITIONS[args.patch_index] or args.num_patches != len(CONDITIONS):
        raise ValueError("Exp03 patch indices must map exactly to baseline, full, shuffled")
    anchors_path = args.data_dir / "anchors.jsonl"
    dev_path = args.data_dir / "development_anchors.jsonl"
    cache_index = read_json(args.cache_dir / "cache_index.json")
    input_hash = stable_hash({"anchors": file_hash(anchors_path), "development": file_hash(dev_path), "cache": cache_index})
    split_hash = str(cache_index["split_hash"])
    configuration = config(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, len(CONDITIONS))
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=len(CONDITIONS),
        row_count=0,
        input_hash=input_hash,
        split_hash=split_hash,
        config=configuration,
    )
    if not args.force and manifest_matches(patch_dir / "patch_manifest.json", expected):
        return
    all_anchors = list(read_jsonl(anchors_path))
    train = [anchor for anchor in all_anchors if anchor["split"] == "train"]
    rng = np.random.default_rng(args.seed)
    if len(train) > args.max_train_anchors:
        indices = rng.choice(len(train), size=args.max_train_anchors, replace=False)
        train = [train[int(index)] for index in sorted(indices)]
    validation = list(read_jsonl(dev_path))
    store = EmbeddingStore(args.cache_dir / "cache_index.json")
    dimension = min(args.feature_dim, store.dimension)
    turns = list(read_jsonl(args.data_dir / "turns.jsonl"))
    sources = {str(anchor["anchor_id"]) for anchor in train + validation}
    donors = build_control_map(turns, "same_episode", sources) if condition == "shuffled" else {}
    train = [anchor for anchor in train if condition != "shuffled" or str(anchor["anchor_id"]) in donors]
    validation = [anchor for anchor in validation if condition != "shuffled" or str(anchor["anchor_id"]) in donors]
    if not train or not validation:
        raise RuntimeError(
            f"Exp03 condition {condition} requires nonempty train and validation anchors; "
            f"got train={len(train)}, validation={len(validation)}"
        )
    x_train = np.stack([features(anchor, store, condition, dimension, donors.get(str(anchor["anchor_id"]))) for anchor in train])
    y_train = np.stack([normalize(store.get(str(anchor["target_id"]))["document"][:dimension]) for anchor in train])
    weights, mean, scale = fit_ridge(x_train, y_train, args.ridge_alpha)
    x_validation = np.stack([features(anchor, store, condition, dimension, donors.get(str(anchor["anchor_id"]))) for anchor in validation])
    predictions = predict(x_validation, weights, mean, scale)
    target = np.stack([normalize(store.get(str(anchor["target_id"]))["document"][:dimension]) for anchor in validation])
    normalized_predictions = np.stack([normalize(row) for row in predictions])
    cosines = np.sum(normalized_predictions * target, axis=1)
    denominator = float(np.sum((target - target.mean(axis=0)) ** 2))
    r2 = 1.0 - float(np.sum((target - predictions) ** 2)) / denominator if denominator > 0 else float("nan")
    retrieval_rows = []
    for anchor, prediction_vector in zip(validation, normalized_predictions):
        candidate_ids = [str(value) for value in anchor["candidate_ids"]]
        documents = np.stack([normalize(store.get(candidate_id)["document"][:dimension]) for candidate_id in candidate_ids])
        retrieval_rows.append(rank_scores(candidate_ids, documents @ prediction_vector, str(anchor["target_id"])))
    metric = {
        "condition": condition,
        "n_train": len(train),
        "n_validation": len(validation),
        "mean_target_cosine": float(np.mean(cosines)),
        "embedding_r2": r2,
        "recall_at_1": float(np.mean([row["top1"] for row in retrieval_rows])),
        "recall_at_5": float(np.mean([row["top5"] for row in retrieval_rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in retrieval_rows])),
    }
    patch_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric]).to_csv(patch_dir / "metrics.csv", index=False)
    np.savez_compressed(patch_dir / "ridge_model.npz", weights=weights, mean=mean, scale=scale)
    write_json(
        patch_dir / "patch_manifest.json",
        make_manifest(
            stage=STAGE,
            patch_index=args.patch_index,
            num_patches=len(CONDITIONS),
            row_count=len(validation),
            input_hash=input_hash,
            split_hash=split_hash,
            config=configuration,
            extra={"condition": condition},
        ),
    )


def local(args: argparse.Namespace) -> None:
    def run_condition(index: int) -> None:
        child = argparse.Namespace(**vars(args))
        child.mode = "worker"
        child.patch_index = index
        child.num_patches = len(CONDITIONS)
        execute(child)

    run_parallel(range(len(CONDITIONS)), run_condition, args.jobs, "stage 04 conditions")
    merged = argparse.Namespace(**vars(args))
    merged.mode = "merge"
    merged.num_patches = len(CONDITIONS)
    run_single(lambda: execute(merged), "stage 04 merge")


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
