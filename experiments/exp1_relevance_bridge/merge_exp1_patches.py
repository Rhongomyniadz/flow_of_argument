import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp1_relevance_bridge.exp1_relevance_bridge import (
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    build_group_summary,
    build_summary_payload,
    coerce_pair_frame,
    plot_ablation_by_category,
    plot_bridge_lift_by_category,
    plot_legacy_cosine_distribution,
    resolve_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument(
        "--embedding_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_EMBEDDING_DEVICE,
    )
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_patch_dirs(patches_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in patches_dir.glob("patch_*")
        if path.is_dir() and (path / "exp1_summary.json").exists() and (path / "exp1_bridge_pairs.csv").exists()
    )


def load_patch_summaries(patch_dirs: list[Path]) -> list[dict[str, Any]]:
    return [
        json.loads((patch_dir / "exp1_summary.json").read_text(encoding="utf-8"))
        for patch_dir in tqdm(patch_dirs, desc="Loading Exp 1 patch summaries")
    ]


def validate_patch_set(patch_summaries: list[dict[str, Any]]) -> str:
    if not patch_summaries:
        raise RuntimeError("No Exp 1 patch summaries were found.")

    embedding_model_names = {str(summary["embedding_model_name"]) for summary in patch_summaries}
    if len(embedding_model_names) != 1:
        raise RuntimeError(
            "Expected exactly one embedding model across Exp 1 patch summaries, "
            f"got {sorted(embedding_model_names)}."
        )

    num_patches_values = {int(summary.get("num_patches", -1)) for summary in patch_summaries}
    if len(num_patches_values) != 1:
        raise RuntimeError(
            "Expected exactly one num_patches value across Exp 1 patch summaries, "
            f"got {sorted(num_patches_values)}."
        )

    expected_num_patches = next(iter(num_patches_values))
    observed_patch_indices = sorted(int(summary.get("patch_index", -1)) for summary in patch_summaries)
    expected_patch_indices = list(range(expected_num_patches))
    if observed_patch_indices != expected_patch_indices:
        missing = sorted(set(expected_patch_indices) - set(observed_patch_indices))
        print(
            "Warning: Exp 1 patch set is incomplete. "
            f"Missing patch indices: {missing}",
            file=sys.stderr,
        )

    manifest_hashes = {
        str(summary.get("whitening_artifact", {}).get("manifest_hash"))
        for summary in patch_summaries
    }
    if len(manifest_hashes) != 1 or "None" in manifest_hashes:
        raise RuntimeError(
            "Exp 1 patch summaries do not share a single whitening manifest hash. "
            f"Observed hashes: {sorted(manifest_hashes)}"
        )
    return next(iter(manifest_hashes))


def load_patch_pairs(patch_dirs: list[Path]) -> pd.DataFrame:
    pair_frames = [
        coerce_pair_frame(pd.read_csv(patch_dir / "exp1_bridge_pairs.csv"))
        for patch_dir in tqdm(patch_dirs, desc="Loading Exp 1 patch pairs")
    ]
    merged = pd.concat(pair_frames, ignore_index=True)
    duplicate_mask = merged.duplicated(subset=["pair_id"], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        print(
            "Warning: merged Exp 1 patch pairs contain duplicate pair_id rows. "
            f"Deduplicating {duplicate_count} rows.",
            file=sys.stderr,
        )
        merged = merged.drop_duplicates(subset=["pair_id"], keep="first")
    return merged.sort_values(
        by=["category", "episode_id", "turn_a_idx", "turn_b_idx"],
        kind="stable",
    ).reset_index(drop=True)


def build_patch_merge_section(
    model_output_dir: Path,
    patch_dirs: list[Path],
    patch_summaries: list[dict[str, Any]],
    whitening_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "merged_patch_count": int(len(patch_dirs)),
        "num_patches": int(patch_summaries[0]["num_patches"]),
        "episodes_per_patch": patch_summaries[0].get("episodes_per_patch"),
        "patch_dirs": [str(path) for path in patch_dirs],
        "model_output_dir": str(model_output_dir),
        "whitening_manifest_hash": whitening_manifest_hash,
    }


def main() -> None:
    args = parse_args()
    model_output_dir = resolve_output_dir(args.output_dir, args.embedding_model_name)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = model_output_dir / "patches"
    if not patches_dir.exists():
        raise RuntimeError(f"Patches directory does not exist: {patches_dir}")

    patch_dirs = collect_patch_dirs(patches_dir)
    patch_summaries = load_patch_summaries(patch_dirs)
    whitening_manifest_hash = validate_patch_set(patch_summaries)
    merged_df = load_patch_pairs(patch_dirs)

    pair_csv = model_output_dir / "exp1_bridge_pairs.csv"
    category_csv = model_output_dir / "exp1_bridge_by_category.csv"
    move_csv = model_output_dir / "exp1_bridge_by_move.csv"
    main_plot_png = model_output_dir / "exp1_bridge_lift_by_category.png"
    ablation_plot_png = model_output_dir / "exp1_ablation_by_category.png"
    diagnostic_plot_png = model_output_dir / "exp1_legacy_cosine_distribution.png"
    summary_json = model_output_dir / "exp1_summary.json"

    merged_df.to_csv(pair_csv, index=False)

    category_summary = build_group_summary(
        merged_df[(merged_df["analysis_bucket"] == "headline_constructive") & (merged_df["canonical_retained"] == True)].copy(),
        "category",
        args.seed,
    )
    move_summary = build_group_summary(
        merged_df[merged_df["canonical_retained"] == True].copy(),
        "turn_b_move_label",
        args.seed,
    )
    category_summary.to_csv(category_csv, index=False)
    move_summary.to_csv(move_csv, index=False)
    plot_bridge_lift_by_category(category_summary, main_plot_png)
    plot_ablation_by_category(category_summary, ablation_plot_png)
    plot_legacy_cosine_distribution(merged_df, diagnostic_plot_png)

    categories = list(patch_summaries[0].get("categories", sorted(merged_df["category"].astype(str).unique().tolist())))
    selected_episode_file_count = int(sum(int(summary.get("selected_episode_file_count", 0)) for summary in patch_summaries))
    candidate_episode_file_count = int(patch_summaries[0].get("candidate_episode_file_count", 0))
    summary = build_summary_payload(
        args=args,
        output_dir=model_output_dir,
        df=merged_df,
        category_summary=category_summary,
        move_summary=move_summary,
        pair_csv=pair_csv,
        category_csv=category_csv,
        move_csv=move_csv,
        main_plot_png=main_plot_png,
        ablation_plot_png=ablation_plot_png,
        diagnostic_plot_png=diagnostic_plot_png,
        analysis_stage="merged_full_analysis",
        categories=categories,
        selected_files=[],
        category_files=[],
        whitening_manifest={"manifest_hash": whitening_manifest_hash},
        selected_episode_file_count=selected_episode_file_count,
        candidate_episode_file_count=candidate_episode_file_count,
        extra_sections={
            "patch_merge": build_patch_merge_section(
                model_output_dir=model_output_dir,
                patch_dirs=patch_dirs,
                patch_summaries=patch_summaries,
                whitening_manifest_hash=whitening_manifest_hash,
            )
        },
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
