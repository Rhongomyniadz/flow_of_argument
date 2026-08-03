from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..common.utils import file_hash, read_json, stable_hash, write_json

LABELS = ("supported", "plausible", "unsupported", "contradicted", "unclear")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Summarize a completed two-annotator Exp06 audit CSV.")
    value.add_argument("--audit-csv", type=Path)
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp06_results"))
    return value


def cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    observed = float((first == second).mean())
    first_distribution = first.value_counts(normalize=True)
    second_distribution = second.value_counts(normalize=True)
    expected = sum(float(first_distribution.get(label, 0.0) * second_distribution.get(label, 0.0)) for label in LABELS)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def main() -> None:
    args = parser().parse_args()
    audit_path = args.audit_csv or (args.output_dir / "audit_sample.csv")
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
    previous = read_json(args.output_dir / "summary.json") if (args.output_dir / "summary.json").exists() else {}
    immutable_columns = previous.get("immutable_columns", [])
    if immutable_columns:
        missing_immutable = set(immutable_columns).difference(frame.columns)
        if missing_immutable:
            raise RuntimeError(f"Audit CSV is missing immutable columns: {sorted(missing_immutable)}")
        source_hash = stable_hash(frame[list(immutable_columns)].to_dict("records"))
        if source_hash != previous.get("audit_source_hash"):
            raise RuntimeError("One or more immutable Exp06 sample/source fields changed during annotation")
    raw_agreement = float((frame["annotator_1_label"] == frame["annotator_2_label"]).mean())
    kappa = cohen_kappa(frame["annotator_1_label"], frame["annotator_2_label"])
    supported = {"supported", "plausible"}
    first_supported = frame["annotator_1_label"].isin(supported)
    second_supported = frame["annotator_2_label"].isin(supported)
    metrics = [
        {"metric": "sample_size", "value": float(len(frame))},
        {"metric": "raw_agreement", "value": raw_agreement},
        {"metric": "cohen_kappa", "value": kappa},
        {"metric": "annotator_1_supported_or_plausible", "value": float(first_supported.mean())},
        {"metric": "annotator_2_supported_or_plausible", "value": float(second_supported.mean())},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    agreement = {
        "sample_size": len(frame),
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "label_counts": {
            column: {label: int(count) for label, count in frame[column].value_counts().items()}
            for column in ("annotator_1_label", "annotator_2_label")
        },
        "by_category": [
            {
                "category": str(category),
                "n": len(group),
                "raw_agreement": float((group["annotator_1_label"] == group["annotator_2_label"]).mean()),
            }
            for category, group in frame.groupby("category", sort=True)
        ],
    }
    write_json(args.output_dir / "agreement.json", agreement)
    write_json(
        args.output_dir / "summary.json",
        {
            **previous,
            "experiment": "exp06_human_audit",
            "status": "complete",
            "completed_audit_hash": file_hash(audit_path),
            "agreement": agreement,
        },
    )


if __name__ == "__main__":
    main()
