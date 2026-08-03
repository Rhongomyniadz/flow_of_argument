from __future__ import annotations

"""Stage 02: audit the existing Exp1 pair table."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Stage-local helper functions (utils).
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def list_episode_paths(input_dir: Path) -> list[Path]:
    direct = sorted(input_dir.glob("*.json"))
    return direct if direct else sorted(input_dir.glob("*/*.json"))


def patch_name(index: int, total: int) -> str:
    return f"patch_{index:04d}_of_{total:04d}"


def patch_directory(root: Path, index: int, total: int) -> Path:
    return root / "patches" / patch_name(index, total)


def shard_slice(total_items: int, patch_index: int, items_per_patch: int) -> slice:
    if patch_index < 0 or items_per_patch < 1:
        raise ValueError("patch_index must be non-negative and items_per_patch must be positive")
    start = patch_index * items_per_patch
    return slice(start, min(start + items_per_patch, total_items))


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "torch", "sentence_transformers", "sklearn"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "not-installed"
    return versions


def make_manifest(
    *,
    stage: str,
    patch_index: int,
    num_patches: int,
    row_count: int,
    input_hash: str,
    split_hash: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "patch_index": patch_index,
        "num_patches": num_patches,
        "row_count": row_count,
        "input_hash": input_hash,
        "split_hash": split_hash,
        "config": config,
        "config_hash": stable_hash(config),
        "complete": True,
        "runtime": runtime_versions(),
    }
    if extra:
        payload.update(extra)
    return payload


def manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        observed = read_json(path)
    except Exception:
        return False
    keys = ("stage", "patch_index", "num_patches", "input_hash", "split_hash", "config_hash")
    return bool(observed.get("complete")) and all(observed.get(key) == expected.get(key) for key in keys)


def validate_patch_manifests(root: Path, stage: str, num_patches: int) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for index in range(num_patches):
        path = patch_directory(root, index, num_patches) / "patch_manifest.json"
        if not path.exists():
            raise RuntimeError(f"Missing patch manifest: {path}")
        manifest = read_json(path)
        if not manifest.get("complete"):
            raise RuntimeError(f"Incomplete patch: {path}")
        if manifest.get("stage") != stage:
            raise RuntimeError(f"Stage mismatch in {path}: {manifest.get('stage')} != {stage}")
        if int(manifest.get("patch_index", -1)) != index:
            raise RuntimeError(f"Patch index mismatch in {path}")
        if int(manifest.get("num_patches", -1)) != num_patches:
            raise RuntimeError(f"Patch-count mismatch in {path}")
        manifests.append(manifest)
    for key in ("input_hash", "split_hash", "config_hash"):
        values = {manifest.get(key) for manifest in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Mixed {key} values across {stage} patches: {sorted(values)}")
    return manifests


# Stage-local progress display.
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable



def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit pair-level Exp1 assumption effects.")
    value.add_argument("--pairs-csv", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp01_results"))
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--bootstrap-draws", type=int, default=1000)
    value.add_argument("--force", action="store_true")
    return value


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Exp1 columns: {missing}")


def run(args: argparse.Namespace) -> None:
    input_hash = file_hash(args.pairs_csv)
    config = {"pairs_csv": str(args.pairs_csv), "seed": args.seed, "bootstrap_draws": args.bootstrap_draws}
    config_hash = stable_hash(config)
    summary_path = args.output_dir / "summary.json"
    config_path = args.output_dir / "config.json"
    if not args.force and summary_path.exists() and config_path.exists():
        observed_summary = read_json(summary_path)
        observed_config = read_json(config_path)
        if observed_summary.get("input_hash") == input_hash and observed_config.get("config_hash") == config_hash:
            return
    frame = pd.read_csv(args.pairs_csv)
    require_columns(
        frame,
        [
            "episode_id",
            "reciprocal_rank_without_assumptions",
            "reciprocal_rank_with_assumptions",
            "top1_without_assumptions",
            "top1_with_assumptions",
        ],
    )
    frame["mrr_delta"] = (
        pd.to_numeric(frame["reciprocal_rank_with_assumptions"], errors="coerce")
        - pd.to_numeric(frame["reciprocal_rank_without_assumptions"], errors="coerce")
    )
    frame["top1_delta"] = (
        pd.to_numeric(frame["top1_with_assumptions"], errors="coerce")
        - pd.to_numeric(frame["top1_without_assumptions"], errors="coerce")
    )
    frame = frame.dropna(subset=["mrr_delta", "top1_delta"]).copy()
    if frame.empty:
        raise RuntimeError("Exp01 has no valid numeric pair rows after filtering")
    episode = frame.groupby("episode_id", as_index=False).agg(
        pair_count=("mrr_delta", "size"),
        mean_mrr_delta=("mrr_delta", "mean"),
        mean_top1_delta=("top1_delta", "mean"),
    )
    rng = np.random.default_rng(args.seed)
    episode_values = episode["mean_mrr_delta"].to_numpy(dtype=float)
    draws = []
    for _ in tqdm(range(args.bootstrap_draws), desc="stage 02 bootstrap", unit="draw", dynamic_ncols=True):
        draws.append(float(np.mean(rng.choice(episode_values, size=len(episode_values), replace=True))))
    group_columns = [
        column for column in ("category", "true_next_turn_move_label", "assumption_count", "negative_source")
        if column in frame.columns
    ]
    metric_rows: list[dict[str, Any]] = [
        {
            "group": "overall",
            "value": "all",
            "n_pairs": len(frame),
            "n_episodes": episode["episode_id"].nunique(),
            "mean_mrr_delta": float(frame["mrr_delta"].mean()),
            "mean_top1_delta": float(frame["top1_delta"].mean()),
            "positive_episode_rate": float((episode["mean_mrr_delta"] > 0).mean()),
        }
    ]
    for column in group_columns:
        for group, rows in frame.groupby(column, dropna=False):
            metric_rows.append(
                {
                    "group": column,
                    "value": str(group),
                    "n_pairs": len(rows),
                    "n_episodes": rows["episode_id"].nunique(),
                    "mean_mrr_delta": float(rows["mrr_delta"].mean()),
                    "mean_top1_delta": float(rows["top1_delta"].mean()),
                    "positive_episode_rate": float("nan"),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    episode.to_csv(args.output_dir / "episode_metrics.csv", index=False)
    write_json(args.output_dir / "config.json", {**config, "config_hash": config_hash})
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp01_existing_result_audit",
            "status": "complete",
            "input_hash": input_hash,
            "pair_count": len(frame),
            "episode_count": len(episode),
            "mean_mrr_delta": float(frame["mrr_delta"].mean()),
            "mean_top1_delta": float(frame["top1_delta"].mean()),
            "positive_episode_rate": float((episode["mean_mrr_delta"] > 0).mean()),
            "mrr_delta_ci95_low": float(np.quantile(draws, 0.025)),
            "mrr_delta_ci95_high": float(np.quantile(draws, 0.975)),
            "runtime": runtime_versions(),
        },
    )


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
