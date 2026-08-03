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

        prep_patches = 2
        for index in range(prep_patches):
            run("prepare.build_pilot_dataset", "--mode", "worker", "--input-dir", input_dir, "--output-dir", data_dir, "--episodes-per-task", 4, "--num-patches", prep_patches, "--patch-index", index, "--candidate-count", 8, "--development-limit", 100, "--seed", 42)
        run("prepare.build_pilot_dataset", "--mode", "merge", "--input-dir", input_dir, "--output-dir", data_dir, "--episodes-per-task", 4, "--num-patches", prep_patches, "--candidate-count", 8, "--development-limit", 100, "--seed", 42)

        embed_patches = 2
        for index in range(embed_patches):
            run("prepare.cache_embeddings", "--mode", "worker", "--data-dir", data_dir, "--output-dir", cache_dir, "--episodes-per-task", 4, "--num-patches", embed_patches, "--patch-index", index, "--backend", "hash", "--hash-dim", 16, "--batch-size", 8)
        run("prepare.cache_embeddings", "--mode", "merge", "--data-dir", data_dir, "--output-dir", cache_dir, "--episodes-per-task", 4, "--num-patches", embed_patches, "--backend", "hash", "--hash-dim", 16, "--batch-size", 8)

        run("exp01_existing_result_audit.run", "--pairs-csv", pairs_path, "--output-dir", results[1], "--bootstrap-draws", 20)
        exp02_patches = 2
        for index in range(exp02_patches):
            run("exp02_frozen_retrieval.run", "--mode", "worker", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[2], "--anchors-per-task", 5, "--num-patches", exp02_patches, "--patch-index", index, "--bootstrap-draws", 20)
        run("exp02_frozen_retrieval.run", "--mode", "merge", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[2], "--anchors-per-task", 5, "--num-patches", exp02_patches, "--bootstrap-draws", 20)

        for index in range(3):
            run("exp03_linear_residual.run", "--mode", "worker", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[3], "--num-patches", 3, "--patch-index", index, "--feature-dim", 8, "--max-train-anchors", 50)
        run("exp03_linear_residual.run", "--mode", "merge", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[3], "--num-patches", 3, "--feature-dim", 8, "--max-train-anchors", 50)

        exp04_patches = 6
        for index in range(exp04_patches):
            run("exp04_counterfactual_controls.run", "--mode", "worker", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[4], "--anchors-per-task", 5, "--num-patches", exp04_patches, "--patch-index", index, "--bootstrap-draws", 20)
        run("exp04_counterfactual_controls.run", "--mode", "merge", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[4], "--anchors-per-task", 5, "--num-patches", exp04_patches, "--bootstrap-draws", 20)

        for index in range(9):
            run("exp05_mini_fusion.run", "--mode", "worker", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[5], "--num-patches", 9, "--patch-index", index, "--feature-dim", 8, "--max-train-anchors", 50, "--smoke")
        run("exp05_mini_fusion.run", "--mode", "merge", "--data-dir", data_dir, "--cache-dir", cache_dir, "--output-dir", results[5], "--num-patches", 9, "--feature-dim", 8, "--max-train-anchors", 50, "--smoke")

        run("exp06_human_audit.sample", "--data-dir", data_dir, "--output-dir", results[6], "--sample-size", 12)
        audit_path = results[6] / "audit_sample.csv"
        audit = pd.read_csv(audit_path, keep_default_na=False)
        audit["annotator_1_label"] = ["supported" if index % 3 else "plausible" for index in range(len(audit))]
        audit["annotator_2_label"] = ["supported" if index % 4 else "plausible" for index in range(len(audit))]
        audit.to_csv(audit_path, index=False)
        run("exp06_human_audit.summarize", "--audit-csv", audit_path, "--output-dir", results[6])
        run("summarize_pilot", "--root", root)

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
