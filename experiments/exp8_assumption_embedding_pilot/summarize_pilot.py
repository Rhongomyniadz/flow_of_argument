from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common.utils import read_json, write_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Collect all available Exp8 experiment summaries.")
    value.add_argument("--root", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot"))
    return value


def compact_signal(index: int, summary: dict[str, Any]) -> str:
    if index == 2:
        deltas = summary.get("paired_deltas", [])
        return "; ".join(
            f"{item.get('first')} - {item.get('second')}: ΔMRR={item.get('mean_delta', item.get('delta'))}"
            for item in deltas
        ) or "frozen retrieval complete"
    if index in {3, 5}:
        return f"{len(summary.get('metrics', summary.get('aggregate', [])))} metric rows"
    if index == 6:
        agreement = summary.get("agreement", {})
        return f"status={summary.get('status')}, κ={agreement.get('cohen_kappa', 'pending')}"
    return str(summary.get("status", "available"))


def main() -> None:
    args = parser().parse_args()
    experiments: list[dict[str, Any]] = []
    for index in range(1, 7):
        directory = args.root / f"exp{index:02d}_results"
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            experiments.append(
                {
                    "experiment": f"exp{index:02d}",
                    "available": True,
                    "status": summary.get("status", "complete"),
                    "signal": compact_signal(index, summary),
                    "summary_path": str(summary_path),
                    "summary": summary,
                }
            )
        else:
            experiments.append(
                {
                    "experiment": f"exp{index:02d}",
                    "available": False,
                    "status": "missing",
                    "signal": "not run",
                    "summary_path": str(summary_path),
                }
            )
    complete = sum(item["status"] == "complete" for item in experiments)
    report = {
        "pilot": "exp8_assumption_embedding_pilot",
        "complete_experiment_count": complete,
        "available_experiment_count": sum(bool(item["available"]) for item in experiments),
        "manual_audit_pending": next(item for item in experiments if item["experiment"] == "exp06")["status"] == "awaiting_annotation",
        "experiments": experiments,
    }
    write_json(args.root / "pilot_summary.json", report)
    lines = [
        "# Exp8 assumption-embedding pilot summary",
        "",
        f"Available: {report['available_experiment_count']}/6; complete: {complete}/6.",
        "",
        "| Experiment | Status | Signal |",
        "|---|---|---|",
    ]
    for item in experiments:
        signal = str(item["signal"]).replace("|", "\\|")
        lines.append(f"| {item['experiment']} | {item['status']} | {signal} |")
    lines.extend(["", "The JSON companion retains each experiment's complete summary for programmatic analysis.", ""])
    (args.root / "pilot_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("complete_experiment_count", "available_experiment_count", "manual_audit_pending")}, indent=2))


if __name__ == "__main__":
    main()
