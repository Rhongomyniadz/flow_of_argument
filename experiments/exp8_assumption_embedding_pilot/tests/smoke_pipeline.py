from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


MODULE_ROOT = "experiments.exp8_assumption_embedding_pilot"


def run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", f"{MODULE_ROOT}.{module}", *[str(value) for value in arguments]]
    subprocess.run(command, check=True)


def make_inputs(root: Path) -> tuple[Path, Path]:
    input_dir = root / "input"
    input_dir.mkdir()
    categories = ("news", "science")
    for category in categories:
        category_dir = input_dir / category
        category_dir.mkdir()
        for show_index in range(4):
            episode_id = f"{category}-episode-{show_index}"
            turns = []
            for turn_index in range(6):
                turns.append(
                    {
                        "episode_id": episode_id,
                        "show_id": f"{category}-show-{show_index}",
                        "category": category,
                        "turn_idx": turn_index,
                        "speaker_id": f"speaker-{turn_index % 2}",
                        "text": f"{category} discussion point {turn_index} for show {show_index}",
                        "explicit_propositions": [{"text": f"The visible point is {turn_index}.", "confidence": 0.9}],
                        "assumptions": [
                            {"text": f"The speaker expects consequence {turn_index}.", "confidence": 0.45 + 0.05 * turn_index},
                            {"text": f"The audience knows context {show_index}.", "confidence": 0.6},
                        ],
                    }
                )
            (category_dir / f"{episode_id}.json").write_text(json.dumps(turns), encoding="utf-8")
    pairs = pd.DataFrame(
        [
            {
                "episode_id": f"pair-episode-{index // 4}",
                "category": categories[index % 2],
                "reciprocal_rank_without_assumptions": 0.5,
                "reciprocal_rank_with_assumptions": 1.0 if index % 3 else 0.5,
                "top1_without_assumptions": 0,
                "top1_with_assumptions": 1 if index % 3 else 0,
            }
            for index in range(24)
        ]
    )
    pairs_path = root / "exp1_pairs.csv"
    pairs.to_csv(pairs_path, index=False)
    return input_dir, pairs_path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="exp8_smoke_") as temporary:
        root = Path(temporary)
        input_dir, pairs_path = make_inputs(root)
        data_dir, cache_dir = root / "shared_data", root / "shared_cache"
        results = {index: root / f"exp{index:02d}_results" for index in range(1, 7)}

        run(
            "00_prepare_data.run", "--input-dir", input_dir, "--output-dir", data_dir,
            "--episodes-per-task", 4, "--candidate-count", 8,
            "--development-limit", 100, "--jobs", 2, "--seed", 42,
        )

        embed_patches = 2
        for index in range(embed_patches):
            run("01_cache_embeddings.run", "--mode", "worker", "--data-dir", data_dir, "--output-dir", cache_dir, "--episodes-per-task", 4, "--num-patches", embed_patches, "--patch-index", index, "--backend", "hash", "--hash-dim", 16, "--batch-size", 8)
        run("01_cache_embeddings.run", "--mode", "merge", "--data-dir", data_dir, "--output-dir", cache_dir, "--episodes-per-task", 4, "--num-patches", embed_patches, "--backend", "hash", "--hash-dim", 16, "--batch-size", 8)

        run("02_exp01_audit.run", "--pairs-csv", pairs_path, "--output-dir", results[1], "--bootstrap-draws", 20)
        run("03_exp02_retrieval.run", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[2], "--anchors-per-task", 5, "--bootstrap-draws", 20, "--jobs", 2)
        run("04_exp03_residual.run", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[3], "--feature-dim", 8, "--max-train-anchors", 50, "--jobs", 3)
        run("05_exp04_controls.run", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[4], "--anchors-per-task", 5, "--bootstrap-draws", 20, "--jobs", 3)

        for index in range(9):
            run("06_exp05_fusion.run", "--mode", "worker", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[5], "--num-patches", 9, "--patch-index", index, "--feature-dim", 8, "--max-train-anchors", 50, "--smoke")
        run("06_exp05_fusion.run", "--mode", "merge", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[5], "--num-patches", 9, "--feature-dim", 8, "--max-train-anchors", 50, "--smoke")

        run("07_exp06_audit.run", "--mode", "sample", "--root", root, "--data-dir", data_dir, "--output-dir", results[6], "--sample-size", 12)
        audit_path = results[6] / "audit_sample.csv"
        audit = pd.read_csv(audit_path, keep_default_na=False)
        audit["annotator_1_label"] = ["supported" if index % 3 else "plausible" for index in range(len(audit))]
        audit["annotator_2_label"] = ["supported" if index % 4 else "plausible" for index in range(len(audit))]
        audit.to_csv(audit_path, index=False)
        run("07_exp06_audit.run", "--mode", "summarize", "--root", root, "--output-dir", results[6], "--audit-csv", audit_path)
        run("07_exp06_audit.run", "--mode", "pilot-summary", "--root", root)

        for index, output_dir in results.items():
            for filename in ("config.json", "summary.json", "metrics.csv"):
                path = output_dir / filename
                if not path.exists():
                    raise AssertionError(f"Exp{index:02d} missing {path}")
        summary = json.loads((root / "pilot_summary.json").read_text(encoding="utf-8"))
        if summary["complete_experiment_count"] != 6:
            raise AssertionError(summary)
        print(f"Synthetic Exp8 smoke pipeline passed in {root}")


if __name__ == "__main__":
    main()
