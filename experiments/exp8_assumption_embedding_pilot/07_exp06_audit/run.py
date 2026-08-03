from __future__ import annotations

"""Stage 07: sample, summarize, and report the human audit."""

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



LABELS = ("supported", "plausible", "unsupported", "contradicted", "unclear")
GUIDELINES = """# Exp06 blinded assumption audit

Annotate each row independently. Do not inspect system predictions or retrieval outcomes.

## Labels

- `supported`: directly supported or reasonably implied by the visible dialogue.
- `plausible`: not established, but sensible and not conflicting with the dialogue.
- `unsupported`: introduces unjustified facts, motives, or events.
- `contradicted`: the visible dialogue provides evidence against it.
- `unclear`: the text is too ambiguous or malformed to judge.

Each annotator fills their own label column. Do not edit `item_id` or any source column.
"""


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the Stage 07 human-audit workflow locally.")
    value.add_argument("--mode", choices=("sample", "summarize", "pilot-summary"), default="sample")
    value.add_argument("--root", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot"))
    value.add_argument("--data-dir", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--audit-csv", type=Path)
    value.add_argument("--sample-size", type=int, default=100)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    return value


def assumption_candidates(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in turns:
        for index, statement in enumerate(turn.get("assumptions", [])):
            text = statement.get("text") if isinstance(statement, dict) else str(statement)
            if not str(text).strip():
                continue
            confidence = statement.get("confidence") if isinstance(statement, dict) else None
            try:
                confidence_value = float(confidence) if confidence is not None else float("nan")
            except (TypeError, ValueError):
                confidence_value = float("nan")
            rows.append(
                {
                    "source_key": f"{turn['turn_id']}::assumption::{index}",
                    "category": turn["category"],
                    "show_id": turn["show_id"],
                    "episode_id": turn["episode_id"],
                    "turn_id": turn["turn_id"],
                    "turn_index": turn["turn_idx"],
                    "turn_text": turn["turn_text"],
                    "assumption": str(text).strip(),
                    "confidence": confidence_value,
                    "turn_length": len(str(turn["turn_text"]).split()),
                }
            )
    return rows


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> pd.DataFrame:
    if len(rows) < size:
        raise RuntimeError(f"Only {len(rows)} assumptions are available; cannot sample {size}")
    frame = pd.DataFrame(rows)
    confidence = frame["confidence"].fillna(frame["confidence"].median()).fillna(0.5)
    frame["confidence_bin"] = pd.qcut(confidence.rank(method="first"), q=min(4, len(frame)), labels=False)
    frame["length_bin"] = pd.qcut(frame["turn_length"].rank(method="first"), q=min(3, len(frame)), labels=False)
    frame["stratum"] = frame["category"].astype(str) + "/" + frame["confidence_bin"].astype(str) + "/" + frame["length_bin"].astype(str)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for key, group in frame.groupby("stratum", sort=True):
        values = group.index.to_numpy(copy=True)
        rng.shuffle(values)
        groups[str(key)] = [int(value) for value in values]
    chosen: list[int] = []
    while len(chosen) < size:
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(chosen) < size:
                chosen.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    sample = frame.loc[chosen].copy()
    sample["blind_order"] = rng.permutation(len(sample))
    sample = sample.sort_values("blind_order").reset_index(drop=True)
    sample["item_id"] = [f"audit_{index:04d}" for index in range(len(sample))]
    sample["annotator_1_label"] = ""
    sample["annotator_2_label"] = ""
    sample["notes"] = ""
    return sample


def sample(args: argparse.Namespace) -> None:
    data_dir = args.data_dir or (args.root / "shared_data")
    output_dir = args.output_dir or (args.root / "exp06_results")
    audit_path = output_dir / "audit_sample.csv"
    turns_path = data_dir / "turns.jsonl"
    config: dict[str, Any] = {
        "sample_size": args.sample_size,
        "seed": args.seed,
        "input_hash": file_hash(turns_path),
        "sampling": "round-robin over category x confidence quartile x turn-length tercile",
    }
    config_hash = stable_hash(config)
    if audit_path.exists() and not args.force:
        existing_config = output_dir / "config.json"
        if existing_config.exists() and read_json(existing_config).get("config_hash") == config_hash:
            return
        raise RuntimeError(f"{audit_path} exists with a different input/config hash")
    sampled = stratified_sample(assumption_candidates(list(read_jsonl(turns_path))), args.sample_size, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "item_id", "category", "show_id", "episode_id", "turn_id", "turn_index", "turn_text",
        "assumption", "confidence", "annotator_1_label", "annotator_2_label", "notes",
    ]
    immutable_columns = [column for column in columns if column not in {"annotator_1_label", "annotator_2_label", "notes"}]
    audit_frame = sampled[columns]
    audit_frame.to_csv(audit_path, index=False)
    (output_dir / "annotation_guidelines.md").write_text(GUIDELINES, encoding="utf-8")
    write_json(output_dir / "config.json", {**config, "config_hash": config_hash})
    write_json(output_dir / "agreement.json", {"status": "awaiting_annotation", "sample_size": len(sampled)})
    pd.DataFrame([{"status": "awaiting_annotation", "sample_size": len(sampled)}]).to_csv(output_dir / "metrics.csv", index=False)
    write_json(
        output_dir / "summary.json",
        {
            "experiment": "exp06_human_audit",
            "status": "awaiting_annotation",
            "sample_size": len(sampled),
            "audit_hash": file_hash(audit_path),
            "audit_source_hash": stable_hash(audit_frame[immutable_columns].to_dict("records")),
            "immutable_columns": immutable_columns,
            "instructions": str(output_dir / "annotation_guidelines.md"),
        },
    )


def cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    observed = float((first == second).mean())
    first_distribution = first.value_counts(normalize=True)
    second_distribution = second.value_counts(normalize=True)
    expected = sum(float(first_distribution.get(label, 0.0) * second_distribution.get(label, 0.0)) for label in LABELS)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def summarize(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or (args.root / "exp06_results")
    audit_path = args.audit_csv or (output_dir / "audit_sample.csv")
    frame = pd.read_csv(audit_path, keep_default_na=False)
    required = {"item_id", "annotator_1_label", "annotator_2_label", "category"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Audit CSV is missing columns: {sorted(missing)}")
    for column in ("annotator_1_label", "annotator_2_label"):
        frame[column] = frame[column].astype(str).str.strip().str.lower()
        invalid = sorted(set(frame[column]).difference(LABELS))
        if invalid:
            raise RuntimeError(f"{column} contains missing or invalid labels: {invalid}")
    if frame["item_id"].duplicated().any():
        raise RuntimeError("Audit CSV contains duplicate item_id values")
    previous = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    immutable_columns = previous.get("immutable_columns", [])
    if immutable_columns:
        source_hash = stable_hash(frame[list(immutable_columns)].to_dict("records"))
        if source_hash != previous.get("audit_source_hash"):
            raise RuntimeError("One or more immutable audit source fields changed")
    raw_agreement = float((frame["annotator_1_label"] == frame["annotator_2_label"]).mean())
    kappa = cohen_kappa(frame["annotator_1_label"], frame["annotator_2_label"])
    supported = {"supported", "plausible"}
    metrics = [
        {"metric": "sample_size", "value": float(len(frame))},
        {"metric": "raw_agreement", "value": raw_agreement},
        {"metric": "cohen_kappa", "value": kappa},
        {"metric": "annotator_1_supported_or_plausible", "value": float(frame["annotator_1_label"].isin(supported).mean())},
        {"metric": "annotator_2_supported_or_plausible", "value": float(frame["annotator_2_label"].isin(supported).mean())},
    ]
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    agreement = {
        "sample_size": len(frame),
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "label_counts": {
            column: {label: int(count) for label, count in frame[column].value_counts().items()}
            for column in ("annotator_1_label", "annotator_2_label")
        },
    }
    write_json(output_dir / "agreement.json", agreement)
    write_json(
        output_dir / "summary.json",
        {
            **previous,
            "experiment": "exp06_human_audit",
            "status": "complete",
            "completed_audit_hash": file_hash(audit_path),
            "agreement": agreement,
        },
    )


def pilot_summary(args: argparse.Namespace) -> None:
    experiments: list[dict[str, Any]] = []
    for index in range(1, 7):
        summary_path = args.root / f"exp{index:02d}_results" / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            experiments.append({"experiment": f"exp{index:02d}", "available": True, "status": summary.get("status", "complete"), "summary_path": str(summary_path), "summary": summary})
        else:
            experiments.append({"experiment": f"exp{index:02d}", "available": False, "status": "missing", "summary_path": str(summary_path)})
    report = {
        "pilot": "exp8_assumption_embedding_pilot",
        "complete_experiment_count": sum(item["status"] == "complete" for item in experiments),
        "available_experiment_count": sum(bool(item["available"]) for item in experiments),
        "manual_audit_pending": next(item for item in experiments if item["experiment"] == "exp06")["status"] == "awaiting_annotation",
        "experiments": experiments,
    }
    write_json(args.root / "pilot_summary.json", report)
    lines = [
        "# Exp8 assumption-embedding pilot summary", "",
        f"Available: {report['available_experiment_count']}/6; complete: {report['complete_experiment_count']}/6.", "",
        "| Experiment | Status |", "|---|---|",
    ]
    lines.extend(f"| {item['experiment']} | {item['status']} |" for item in experiments)
    (args.root / "pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    if args.mode == "sample":
        sample(args)
    elif args.mode == "summarize":
        summarize(args)
    else:
        pilot_summary(args)


if __name__ == "__main__":
    main()
