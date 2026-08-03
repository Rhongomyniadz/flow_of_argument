from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.utils import file_hash, read_json, read_jsonl, stable_hash, write_json


GUIDELINES = """# Exp06 blinded assumption audit

Annotate each row independently. Do not inspect system predictions or retrieval outcomes.

## Labels

- `supported`: the assumption is directly supported or is a reasonable implication of the visible current turn and prior dialogue.
- `plausible`: the assumption is not established, but is a sensible possibility and does not conflict with the dialogue.
- `unsupported`: the assumption introduces unjustified facts, motives, or events.
- `contradicted`: the visible dialogue provides evidence against the assumption.
- `unclear`: the text is too ambiguous or malformed to judge.

Each of two annotators fills `annotator_1_label` or `annotator_2_label`. Do not edit `item_id` or any source column after annotation starts. Use `notes` only for short adjudication-relevant comments.
"""


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create the immutable blinded Exp06 audit sample.")
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp06_results"))
    value.add_argument("--sample-size", type=int, default=100)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    return value


def candidates(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def main() -> None:
    args = parser().parse_args()
    audit_path = args.output_dir / "audit_sample.csv"
    turns_path = args.data_dir / "turns.jsonl"
    config: dict[str, Any] = {
        "sample_size": args.sample_size,
        "seed": args.seed,
        "input_hash": file_hash(turns_path),
        "sampling": "round-robin over category x confidence quartile x turn-length tercile",
    }
    config_hash = stable_hash(config)
    if audit_path.exists() and not args.force:
        existing_config = args.output_dir / "config.json"
        if existing_config.exists() and read_json(existing_config).get("config_hash") == config_hash:
            print(f"Reusing immutable Exp06 sample {audit_path}")
            return
        raise RuntimeError(f"{audit_path} exists but its input/config hash differs; refusing to overwrite it")
    sampled = stratified_sample(candidates(list(read_jsonl(turns_path))), args.sample_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "item_id", "category", "show_id", "episode_id", "turn_id", "turn_index", "turn_text",
        "assumption", "confidence", "annotator_1_label", "annotator_2_label", "notes",
    ]
    immutable_columns = [column for column in columns if column not in {"annotator_1_label", "annotator_2_label", "notes"}]
    audit_frame = sampled[columns]
    audit_frame.to_csv(audit_path, index=False)
    (args.output_dir / "annotation_guidelines.md").write_text(GUIDELINES, encoding="utf-8")
    write_json(args.output_dir / "config.json", {**config, "config_hash": config_hash})
    write_json(args.output_dir / "agreement.json", {"status": "awaiting_annotation", "sample_size": len(sampled)})
    pd.DataFrame([{"status": "awaiting_annotation", "sample_size": len(sampled)}]).to_csv(args.output_dir / "metrics.csv", index=False)
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp06_human_audit",
            "status": "awaiting_annotation",
            "sample_size": len(sampled),
            "audit_hash": file_hash(audit_path),
            "audit_source_hash": stable_hash(audit_frame[immutable_columns].to_dict("records")),
            "immutable_columns": immutable_columns,
            "instructions": str(args.output_dir / "annotation_guidelines.md"),
        },
    )


if __name__ == "__main__":
    main()
