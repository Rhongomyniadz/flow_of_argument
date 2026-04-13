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
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
    DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
    bootstrap_mean,
    bootstrap_pointplot,
    build_by_category,
    compute_trajectory_metrics,
    distribution_plot,
    embed_texts,
    pointplot_long_frame,
    pointplot_summary,
    resolve_output_dir,
    write_umap_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_patch_dirs(patches_dir: Path) -> list[Path]:
    return sorted(
        path for path in patches_dir.glob("patch_*") if path.is_dir() and (path / "exp1_summary.json").exists()
    )


def load_patch_summaries(patch_dirs: list[Path]) -> list[dict[str, Any]]:
    return [
        json.loads((patch_dir / "exp1_summary.json").read_text())
        for patch_dir in tqdm(patch_dirs, desc="Loading Exp 1 patch summaries")
    ]


def validate_patch_set(patch_summaries: list[dict[str, Any]]) -> None:
    if not patch_summaries:
        raise RuntimeError("No patch summaries found.")
    expected_num_patches = int(patch_summaries[0]["num_patches"])
    observed_patch_indices = sorted(int(summary["patch_index"]) for summary in patch_summaries)
    expected_patch_indices = list(range(expected_num_patches))
    if observed_patch_indices != expected_patch_indices:
        missing = sorted(set(expected_patch_indices) - set(observed_patch_indices))
        print(
            "Warning: patch set is incomplete. "
            f"Missing indices: {missing}",
            file=sys.stderr,
        )
    embedding_model_names = {str(summary["embedding_model_name"]) for summary in patch_summaries}
    if len(embedding_model_names) != 1:
        raise RuntimeError(f"Expected exactly one embedding model in patch set, got {sorted(embedding_model_names)}")


def load_patch_pairs(patch_dirs: list[Path]) -> pd.DataFrame:
    pair_frames = [
        pd.read_csv(patch_dir / "exp1_bridge_pairs.csv")
        for patch_dir in tqdm(patch_dirs, desc="Loading Exp 1 patch pairs")
    ]
    merged = pd.concat(pair_frames, ignore_index=True)
    duplicate_mask = merged.duplicated(subset=["episode_id", "turn_a_idx", "turn_b_idx"], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        print(
            "Warning: merged patch pairs contain duplicate adjacent-turn rows. "
            f"Deduplicating {duplicate_count} rows.",
            file=sys.stderr,
        )
        merged = merged.drop_duplicates(subset=["episode_id", "turn_a_idx", "turn_b_idx"], keep="first")
    return merged


def add_vector_columns(df: pd.DataFrame, embedding_model_name: str) -> pd.DataFrame:
    texts_to_embed = pd.concat(
        [
            df["turn_b_claim_text"].astype(str),
            df["turn_b_selected_context_text"].astype(str),
        ],
        ignore_index=True,
    ).tolist()
    text_to_vec = embed_texts(
        texts=texts_to_embed,
        batch_size=128,
        use_tqdm=True,
        embedding_model_name=embedding_model_name,
    )
    enriched = df.copy()
    enriched["vec_claim"] = enriched["turn_b_claim_text"].map(text_to_vec)
    enriched["vec_context"] = enriched["turn_b_selected_context_text"].map(text_to_vec)
    return enriched


def build_summary(
    args: argparse.Namespace,
    model_output_dir: Path,
    patch_dirs: list[Path],
    patch_summaries: list[dict[str, Any]],
    df: pd.DataFrame,
    categories: list[str],
    pair_csv: Path,
    category_csv: Path,
    pointplot_csv: Path,
    pointplot_png: Path,
    summary_json: Path,
    dist_png: Path,
    umap_sample_csv: Path,
    umap_png: Path,
    trajectory_metrics: dict[str, Any],
    umap_outputs: dict[str, Any],
) -> dict[str, Any]:
    bridge_boot = bootstrap_mean(df["bridge_delta"], seed=args.seed, draws=DEFAULT_BOOTSTRAP_DRAWS)
    full_bag_boot = bootstrap_mean(df["full_bag_bridge_delta"], seed=args.seed + 1, draws=DEFAULT_BOOTSTRAP_DRAWS)
    return {
        "experiment": "Experiment 1: The Relevance Bridge",
        "analysis_stage": "merged_full_analysis",
        "input_dir": str(args.input_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "selected_episode_file_count": int(
            sum(int(summary.get("selected_episode_file_count", 0)) for summary in patch_summaries)
        ),
        "candidate_episode_file_count": int(patch_summaries[0].get("candidate_episode_file_count", 0)),
        "global_assumption_pool_size": int(patch_summaries[0].get("global_assumption_pool_size", 0)),
        "global_assumption_pool_scope": patch_summaries[0].get("global_assumption_pool_scope"),
        "global_assumption_pool_source": patch_summaries[0].get("global_assumption_pool_source"),
        "global_assumption_pool_path": patch_summaries[0].get("global_assumption_pool_path"),
        "categories": categories,
        "total_pairs": int(len(df)),
        "pairs_with_assumptions_on_turn_b": int(df["turn_b_has_assumptions"].sum()),
        "assumption_pair_rate": float(df["turn_b_has_assumptions"].mean()),
        "selected_assumption_rate": float(df["selected_assumption_rate"].mean()),
        "average_selected_assumption_count": float(df["selected_assumption_count"].mean()),
        "bridge_score_mean_delta": float(df["bridge_delta"].mean()),
        "bridge_score_median_delta": float(df["bridge_delta"].median()),
        "positive_bridge_rate": float((df["bridge_delta"] > 0).mean()),
        "legacy_full_bag_mean_delta": float(df["full_bag_bridge_delta"].mean()),
        "legacy_full_bag_median_delta": float(df["full_bag_bridge_delta"].median()),
        "legacy_full_bag_positive_rate": float((df["full_bag_bridge_delta"] > 0).mean()),
        "best_single_assumption_gain_mean": float(df["best_single_assumption_gain"].mean()),
        "mean_similarity_claim_only": float(df["sim_claim"].mean()),
        "mean_similarity_filtered_context": float(df["sim_context"].mean()),
        "mean_similarity_with_assumptions": float(df["sim_context"].mean()),
        "mean_similarity_full_bag_context": float(df["sim_full_bag_context"].mean()),
        "mean_similarity_same_episode_implicit_sample": float(df["sim_same_episode_sample"].mean()),
        "mean_similarity_any_episode_implicit_sample": float(df["sim_global_sample"].mean()),
        "mean_similarity_gain_percent": float(
            100.0 * (df["sim_context"].mean() - df["sim_claim"].mean()) / max(abs(df["sim_claim"].mean()), 1e-9)
        ),
        "bridge_delta_bootstrap": bridge_boot,
        "legacy_full_bag_bridge_delta_bootstrap": full_bag_boot,
        "trajectory_smoothing": trajectory_metrics,
        "umap": umap_outputs,
        "patch_merge": {
            "merged_patch_count": int(len(patch_dirs)),
            "num_patches": int(patch_summaries[0]["num_patches"]),
            "episodes_per_patch": patch_summaries[0].get("episodes_per_patch"),
            "patch_dirs": [str(path) for path in patch_dirs],
            "model_output_dir": str(model_output_dir),
        },
        "outputs": {
            "pair_csv": str(pair_csv),
            "category_csv": str(category_csv),
            "pointplot_summary_csv": str(pointplot_csv),
            "pointplot_png": str(pointplot_png),
            "summary_json": str(summary_json),
            "distance_distribution_png": str(dist_png),
            "umap_sample_csv": str(umap_sample_csv),
            "umap_trajectory_png": str(umap_png),
        },
        "notes": [
            "This summary is merged from completed Exp 1 patch outputs.",
            "Only turns with turn_type_label == 'Substantive' are included.",
            "Turn A is vectorized from explicit propositions when available, otherwise raw turn text.",
            "Turn B context adds only greedily selected same-turn assumptions whose inclusion improves similarity to Turn A.",
            "Matched implicit baselines sample the same number of assumptions as the filtered bridge context.",
            "Full bag context is retained only as a diagnostic comparison against the filtered bridge context.",
            "Cosine similarity is computed on L2-normalized sentence embeddings from the selected embedding model.",
        ],
    }


def main() -> None:
    args = parse_args()
    model_output_dir = resolve_output_dir(args.output_dir, args.embedding_model_name)
    patches_dir = model_output_dir / "patches"
    if not patches_dir.exists():
        raise RuntimeError(f"Patches directory does not exist: {patches_dir}")

    patch_dirs = collect_patch_dirs(patches_dir)
    patch_summaries = load_patch_summaries(patch_dirs)
    validate_patch_set(patch_summaries)

    df = load_patch_pairs(patch_dirs)
    categories = sorted(df["category"].dropna().astype(str).unique().tolist())

    pair_csv = model_output_dir / "exp1_bridge_pairs.csv"
    category_csv = model_output_dir / "exp1_bridge_by_category.csv"
    pointplot_csv = model_output_dir / "exp1_similarity_pointplot_summary.csv"
    pointplot_png = model_output_dir / "exp1_similarity_pointplot.png"
    summary_json = model_output_dir / "exp1_summary.json"
    dist_png = model_output_dir / "exp1_distance_distribution.png"
    umap_sample_csv = model_output_dir / "exp1_umap_sample.csv"
    umap_png = model_output_dir / "exp1_umap_trajectory.png"

    distribution_plot(df, dist_png)
    df.to_csv(pair_csv, index=False)

    by_category = build_by_category(df)
    by_category.to_csv(category_csv, index=False)

    long_plot_df = pointplot_long_frame(df)
    pointplot_summary_df = pointplot_summary(long_plot_df, seed=args.seed)
    pointplot_summary_df.to_csv(pointplot_csv, index=False)
    bootstrap_pointplot(long_plot_df, pointplot_png, category_order=categories, seed=args.seed)

    trajectory_df = add_vector_columns(df, args.embedding_model_name)
    trajectory_metrics = compute_trajectory_metrics(trajectory_df)
    umap_outputs = write_umap_outputs(
        df=trajectory_df,
        sample_csv_path=umap_sample_csv,
        plot_path=umap_png,
        seed=args.seed,
    )

    summary = build_summary(
        args=args,
        model_output_dir=model_output_dir,
        patch_dirs=patch_dirs,
        patch_summaries=patch_summaries,
        df=df,
        categories=categories,
        pair_csv=pair_csv,
        category_csv=category_csv,
        pointplot_csv=pointplot_csv,
        pointplot_png=pointplot_png,
        summary_json=summary_json,
        dist_png=dist_png,
        umap_sample_csv=umap_sample_csv,
        umap_png=umap_png,
        trajectory_metrics=trajectory_metrics,
        umap_outputs=umap_outputs,
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
