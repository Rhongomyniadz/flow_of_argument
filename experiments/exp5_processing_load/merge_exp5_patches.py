import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp5_processing_load.exp5 import analyze_rows, build_arg_parser, resolve_output_dir


def collect_patch_dirs(patches_dir: Path) -> list[Path]:
    return sorted(path for path in patches_dir.glob("patch_*") if path.is_dir() and (path / "exp5_summary.json").exists())


def load_patch_summaries(patch_dirs: list[Path]) -> list[dict[str, Any]]:
    return [json.loads((patch_dir / "exp5_summary.json").read_text()) for patch_dir in patch_dirs]


def validate_patch_set(patch_summaries: list[dict[str, Any]]) -> None:
    if not patch_summaries:
        raise RuntimeError("No Exp 5 patch summaries found.")
    expected_num_patches = int(patch_summaries[0]["num_patches"])
    observed_patch_indices = sorted(int(summary["patch_index"]) for summary in patch_summaries)
    expected_patch_indices = list(range(expected_num_patches))
    if observed_patch_indices != expected_patch_indices:
        raise RuntimeError(
            "Patch set is incomplete or inconsistent. "
            f"Expected patch indices {expected_patch_indices}, got {observed_patch_indices}"
        )
    values = {float(summary["silence_gap_threshold_sec"]) for summary in patch_summaries}
    if len(values) != 1:
        raise RuntimeError(f"Expected identical silence gap thresholds across patches, got {sorted(values)}")
    embedding_models = {str(summary["embedding_model_name"]) for summary in patch_summaries}
    if len(embedding_models) != 1:
        raise RuntimeError(f"Expected exactly one embedding model across patches, got {sorted(embedding_models)}")


def load_merged_rows(patch_dirs: list[Path]) -> pd.DataFrame:
    rows = pd.concat([pd.read_csv(path / "exp5_turn_level_features.csv") for path in patch_dirs], ignore_index=True)
    duplicate_mask = rows.duplicated(subset=["episode_id", "turn_idx"], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        print(
            "Warning: merged Exp 5 turn-level features contain duplicate episode/turn rows. "
            f"Deduplicating {duplicate_count} rows.",
            file=sys.stderr,
        )
        rows = rows.drop_duplicates(subset=["episode_id", "turn_idx"], keep="first")
    return rows


def build_merge_args(base_args: argparse.Namespace, patch_summary: dict[str, Any]) -> argparse.Namespace:
    merged_args = argparse.Namespace(**vars(base_args))
    merged_args.silence_gap_quantile = float(patch_summary["silence_gap_quantile"])
    merged_args.min_silence_gap = float(patch_summary["min_silence_gap_sec"])
    merged_args.assumption_similarity_threshold = float(
        patch_summary["assumption_sharedness_method"]["similarity_threshold"]
    )
    merged_args.bayes_multinomial_prior_sd = float(patch_summary["bayes_multinomial_prior_sd"])
    merged_args.bayes_multinomial_draws = int(patch_summary["bayes_multinomial_draws"])
    merged_args.bayes_linear_draws = int(patch_summary["bayes_linear_draws"])
    merged_args.bayes_linear_prior_precision_scale = float(
        patch_summary["bayes_linear_prior_precision_scale"]
    )
    merged_args.bayes_linear_prior_a0 = float(patch_summary["bayes_linear_prior_a0"])
    merged_args.bayes_linear_prior_b0 = float(patch_summary["bayes_linear_prior_b0"])
    merged_args.num_patches = int(patch_summary["num_patches"])
    merged_args.patch_index = -1
    merged_args.episodes_per_patch = patch_summary.get("episodes_per_patch")
    return merged_args


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    model_output_dir = resolve_output_dir(args.output_dir, args.embedding_model_name)
    patches_dir = model_output_dir / "patches"
    if not patches_dir.exists():
        raise RuntimeError(f"Patches directory does not exist: {patches_dir}")

    patch_dirs = collect_patch_dirs(patches_dir)
    patch_summaries = load_patch_summaries(patch_dirs)
    validate_patch_set(patch_summaries)

    rows_df = load_merged_rows(patch_dirs)
    merged_args = build_merge_args(args, patch_summaries[0])
    analyze_rows(
        rows=rows_df.to_dict("records"),
        output_dir=model_output_dir,
        input_dir=Path(args.input_dir),
        num_episodes=int(rows_df["episode_id"].nunique()),
        silence_gap=float(patch_summaries[0]["silence_gap_threshold_sec"]),
        args=merged_args,
        show_progress=not args.no_tqdm,
        summary_extra={
            "selected_episode_file_count": int(
                sum(int(summary.get("selected_episode_file_count", 0)) for summary in patch_summaries)
            ),
            "candidate_episode_file_count": int(patch_summaries[0].get("candidate_episode_file_count", 0)),
            "patch_merge": {
                "merged_patch_count": int(len(patch_dirs)),
                "num_patches": int(patch_summaries[0]["num_patches"]),
                "episodes_per_patch": patch_summaries[0].get("episodes_per_patch"),
                "patch_dirs": [str(path) for path in patch_dirs],
                "model_output_dir": str(model_output_dir),
            },
        },
    )


if __name__ == "__main__":
    main()
