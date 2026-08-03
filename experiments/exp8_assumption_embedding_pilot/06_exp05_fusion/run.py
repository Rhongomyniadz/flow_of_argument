from __future__ import annotations

"""Stage 06: train the nine mini-fusion GPU conditions."""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.exp8_assumption_embedding_pilot.common.controls import build_control_map
from experiments.exp8_assumption_embedding_pilot.common.embeddings import EmbeddingStore, component_vectors
from experiments.exp8_assumption_embedding_pilot.common.evaluation import evaluate_anchor
from experiments.exp8_assumption_embedding_pilot.common.metrics import normalize, rank_scores
from experiments.exp8_assumption_embedding_pilot.common.utils import (
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

STAGE = "exp8_exp05_mini_fusion"
CONDITIONS = ("history", "full", "shuffled")
SEEDS = (42, 43, 44)
TASKS = tuple((condition, seed) for condition in CONDITIONS for seed in SEEDS)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Train or merge the nine Exp05 mini-fusion conditions.")
    value.add_argument("--mode", choices=("count", "worker", "merge"), required=True)
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--cache-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp05_results"))
    value.add_argument("--num-patches", type=int, default=len(TASKS))
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--condition", choices=CONDITIONS)
    value.add_argument("--seed", type=int)
    value.add_argument("--feature-dim", type=int, default=256)
    value.add_argument("--hidden-dim", type=int, default=256)
    value.add_argument("--max-train-anchors", type=int, default=50000)
    value.add_argument("--max-epochs", type=int, default=10)
    value.add_argument("--patience", type=int, default=2)
    value.add_argument("--batch-size", type=int, default=512)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--weight-decay", type=float, default=0.01)
    value.add_argument("--temperature", type=float, default=0.05)
    value.add_argument("--device")
    value.add_argument("--smoke", action="store_true", help="Use deterministic no-training scoring for CPU smoke tests.")
    value.add_argument("--force", action="store_true")
    return value


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_mapping": [{"patch_index": i, "condition": c, "seed": s} for i, (c, s) in enumerate(TASKS)],
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "max_train_anchors": args.max_train_anchors,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "smoke": args.smoke,
    }


def task_for(args: argparse.Namespace) -> tuple[str, int]:
    if args.num_patches != len(TASKS) or not 0 <= args.patch_index < len(TASKS):
        raise ValueError(f"Exp05 requires exactly {len(TASKS)} patches indexed 0..{len(TASKS) - 1}")
    expected = TASKS[args.patch_index]
    observed = (args.condition or expected[0], args.seed if args.seed is not None else expected[1])
    if observed != expected:
        raise ValueError(f"Patch {args.patch_index} must be {expected}, got {observed}")
    return expected


def select_anchors(anchors: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(anchors) <= limit:
        return anchors
    rng = np.random.default_rng(seed)
    selected = sorted(int(value) for value in rng.choice(len(anchors), size=limit, replace=False))
    return [anchors[index] for index in selected]


def feature_rows(
    anchors: list[dict[str, Any]],
    store: EmbeddingStore,
    condition: str,
    dimension: int,
    donors: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    kept = [anchor for anchor in anchors if condition != "shuffled" or str(anchor["anchor_id"]) in donors]
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for anchor in kept:
        parts = component_vectors(
            anchor,
            store,
            assumption_source_id=donors.get(str(anchor["anchor_id"])),
            dimension=dimension,
        )
        if condition == "history":
            parts["assumption"] = np.zeros(dimension, dtype=np.float32)
            parts["explicit"] = np.zeros(dimension, dtype=np.float32)
        features.append(np.concatenate([parts[name] for name in ("current", "history", "explicit", "assumption")]))
        targets.append(normalize(store.get(str(anchor["target_id"]))["document"][:dimension]))
    if not kept:
        raise RuntimeError(f"No anchors remain for Exp05 condition {condition}")
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32), kept


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, list[dict[str, Any]], str]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("Exp05 training requires torch; use --smoke only for the synthetic CPU pipeline") from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    class FusionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_dim, args.hidden_dim)
            self.gate = nn.Linear(input_dim, args.hidden_dim)
            self.output = nn.Linear(args.hidden_dim, output_dim)

        def forward(self, values: Any) -> Any:
            hidden = torch.tanh(self.projection(values)) * torch.sigmoid(self.gate(values))
            return functional.normalize(self.output(hidden), dim=-1)

    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train)
    dev_x = torch.from_numpy(x_dev).to(device)
    dev_y = torch.from_numpy(y_dev).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, Any] | None = None
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_x = train_x[indices].to(device)
            batch_y = train_y[indices].to(device)
            prediction = model(batch_x)
            logits = prediction @ batch_y.T / args.temperature
            labels = torch.arange(len(indices), device=device)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            cosine = float((model(dev_x) * dev_y).sum(dim=1).mean().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_cosine": cosine})
        if cosine > best_score + 1e-6:
            best_score = cosine
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Exp05 did not produce a model checkpoint")
    model.load_state_dict(best_state)
    return model, history, device


def score_model(model: Any, x_dev: np.ndarray, anchors: list[dict[str, Any]], store: EmbeddingStore, dimension: int, device: str) -> dict[str, float]:
    import torch

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_dev).to(device)).detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for anchor, prediction in zip(anchors, predictions):
        candidate_ids = [str(value) for value in anchor["candidate_ids"]]
        documents = np.stack([normalize(store.get(value)["document"][:dimension]) for value in candidate_ids])
        rows.append(rank_scores(candidate_ids, documents @ normalize(prediction), str(anchor["target_id"])))
    return {
        "recall_at_1": float(np.mean([row["top1"] for row in rows])),
        "recall_at_5": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
    }


def smoke_metrics(anchors: list[dict[str, Any]], store: EmbeddingStore, condition: str, donors: dict[str, str]) -> dict[str, float]:
    frozen = "current_history" if condition == "history" else ("shuffled" if condition == "shuffled" else "full")
    rows = [
        evaluate_anchor(anchor, store, frozen, donor_id=donors.get(str(anchor["anchor_id"])))
        for anchor in anchors
        if condition != "shuffled" or str(anchor["anchor_id"]) in donors
    ]
    if not rows:
        raise RuntimeError(f"No development anchors remain for Exp05 smoke condition {condition}")
    return {
        "recall_at_1": float(np.mean([row["top1"] for row in rows])),
        "recall_at_5": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
    }


def worker(args: argparse.Namespace) -> None:
    condition, seed = task_for(args)
    anchors_path = args.data_dir / "anchors.jsonl"
    dev_path = args.data_dir / "development_anchors.jsonl"
    cache_index = read_json(args.cache_dir / "cache_index.json")
    input_hash = stable_hash({"anchors": file_hash(anchors_path), "development": file_hash(dev_path), "cache": cache_index})
    config_value = configuration(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, len(TASKS))
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=len(TASKS),
        row_count=0,
        input_hash=input_hash,
        split_hash=str(cache_index["split_hash"]),
        config=config_value,
    )
    if not args.force and manifest_matches(patch_dir / "patch_manifest.json", expected):
        print(f"Reusing completed Exp05 task {condition}/{seed}")
        return
    all_anchors = list(read_jsonl(anchors_path))
    train = select_anchors([value for value in all_anchors if value["split"] == "train"], args.max_train_anchors, seed)
    dev = list(read_jsonl(dev_path))
    store = EmbeddingStore(args.cache_dir / "cache_index.json")
    dimension = min(store.dimension, args.feature_dim)
    turns = list(read_jsonl(args.data_dir / "turns.jsonl"))
    sources = {str(value["anchor_id"]) for value in train + dev}
    donors = build_control_map(turns, "same_episode", sources) if condition == "shuffled" else {}
    patch_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if args.smoke:
        metrics = smoke_metrics(dev, store, condition, donors)
        epochs = 0
    else:
        x_train, y_train, train = feature_rows(train, store, condition, dimension, donors)
        x_dev, y_dev, dev = feature_rows(dev, store, condition, dimension, donors)
        model, history, device = train_model(x_train, y_train, x_dev, y_dev, args, seed)
        metrics = score_model(model, x_dev, dev, store, dimension, device)
        import torch

        torch.save(model.state_dict(), patch_dir / "model.pt")
        epochs = len(history)
    metric = {
        "condition": condition,
        "seed": seed,
        "n_train": len(train),
        "n_validation": len(dev),
        "epochs": epochs,
        **metrics,
    }
    pd.DataFrame([metric]).to_csv(patch_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(patch_dir / "training_history.csv", index=False)
    write_json(
        patch_dir / "patch_manifest.json",
        make_manifest(
            stage=STAGE,
            patch_index=args.patch_index,
            num_patches=len(TASKS),
            row_count=len(dev),
            input_hash=input_hash,
            split_hash=str(cache_index["split_hash"]),
            config=config_value,
            extra={"condition": condition, "seed": seed},
        ),
    )


def merge(args: argparse.Namespace) -> None:
    if args.num_patches != len(TASKS):
        raise ValueError(f"Exp05 merge requires --num-patches {len(TASKS)}")
    manifests = validate_patch_manifests(args.output_dir, STAGE, len(TASKS))
    observed = [(str(item["condition"]), int(item["seed"])) for item in manifests]
    if observed != list(TASKS):
        raise RuntimeError(f"Exp05 patch/seed mapping mismatch: {observed}")
    frames = [
        pd.read_csv(patch_directory(args.output_dir, index, len(TASKS)) / "metrics.csv")
        for index in range(len(TASKS))
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    aggregated = (
        metrics.groupby("condition")[["recall_at_1", "recall_at_5", "mrr"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = ["condition"] + [f"{metric}_{stat}" for metric, stat in aggregated.columns.tolist()[1:]]
    write_json(args.output_dir / "config.json", {**configuration(args), "config_hash": manifests[0]["config_hash"]})
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp05_mini_fusion",
            "status": "complete",
            "input_hash": manifests[0]["input_hash"],
            "split_hash": manifests[0]["split_hash"],
            "task_count": len(TASKS),
            "metrics": metrics.to_dict("records"),
            "aggregate": aggregated.to_dict("records"),
        },
    )


def main() -> None:
    args = parser().parse_args()
    if args.mode == "count":
        print(f"TOTAL_ITEMS={len(TASKS)}")
        print(f"NUM_PATCHES={len(TASKS)}")
    elif args.mode == "worker":
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
