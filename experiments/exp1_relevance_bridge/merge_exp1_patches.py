import argparse
import json
from pathlib import Path
import sys

import pandas as pd

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
    distribution_plot,
    pointplot_long_frame,
    pointplot_summary,
    resolve_output_dir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_patch_dirs(patches_dir: Path):
    return sorted(
        path for path in patches_dir.glob("patch_*") if path.is_dir() and (path / "exp1_summary.json").exists()
    )


def load_patch_summaries(patch_dirs):
    return [json.loads((patch_dir / "exp1_summary.json").read_text()) for patch_dir in patch_dirs]


def validate_patch_set(patch_summaries):
    if not patch_summaries:
        raise RuntimeError("No patch summaries found.")
    expected_num_patches = int(patch_summaries[0]["num_patches"])
    observed_patch_indices = sorted(int(summary["patch_index"]) for summary in patch_summaries)
    expected_patch_indices = list(range(expected_num_patches))
    if observed_patch_indices != expected_patch_indices:
        raise RuntimeError(
            "Patch set is incomplete or inconsistent. "
            f"Expected patch indices {expected_patch_indices}, got {observed_patch_indices}"
        )
    embedding_model_names = {str(summary["embedding_model_name"]) for summary in patch_summaries}
    if len(embedding_model_names) != 1:
        raise RuntimeError(f"Expected exactly one embedding model in patch set, got {sorted(embedding_model_names)}")


def load_patch_pairs(patch_dirs):
    pair_frames = [pd.read_csv(patch_dir / "exp1_bridge_pairs.csv") for patch_dir in patch_dirs]
    merged = pd.concat(pair_frames, ignore_index=True)
    duplicate_mask = merged.duplicated(subset=["episode_id", "turn_a_idx", "turn_b_idx"], keep=False)
    if duplicate_mask.any():
        raise RuntimeError(
            "Merged patch pairs contain duplicate adjacent-turn rows. "
            f"Duplicate row count: {int(duplicate_mask.sum())}"
        )
    return merged


def build_by_category(df: pd.DataFrame):
    return (
        df.groupby("category", as_index=False)
        .agg(
            pair_count=("bridge_delta", "size"),
            mean_sim_claim=("sim_claim", "mean"),
            mean_sim_context=("sim_context", "mean"),
            mean_sim_same_episode_sample=("sim_same_episode_sample", "mean"),
            mean_sim_global_sample=("sim_global_sample", "mean"),
            mean_bridge_delta=("bridge_delta", "mean"),
            positive_bridge_rate=("bridge_delta", lambda x: float((x > 0).mean())),
            assumption_pair_rate=("turn_b_has_assumptions", "mean"),
        )
        .sort_values("mean_bridge_delta", ascending=False)
    )


def build_summary(
    args,
    model_output_dir: Path,
    patch_dirs,
    patch_summaries,
    df: pd.DataFrame,
    categories,
    pair_csv: Path,
    category_csv: Path,
    pointplot_csv: Path,
    pointplot_png: Path,
    summary_json: Path,
    dist_png: Path,
):
    boot = bootstrap_mean(df["bridge_delta"], seed=args.seed, draws=DEFAULT_BOOTSTRAP_DRAWS)
    positive_rate = float((df["bridge_delta"] > 0).mean())
    return {
        "experiment": "Experiment 1: The Relevance Bridge",
        "input_dir": str(args.input_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "categories": categories,
        "total_pairs": int(len(df)),
        "pairs_with_assumptions_on_turn_b": int(df["turn_b_has_assumptions"].sum()),
        "assumption_pair_rate": float(df["turn_b_has_assumptions"].mean()),
        "bridge_score_mean_delta": float(df["bridge_delta"].mean()),
        "bridge_score_median_delta": float(df["bridge_delta"].median()),
        "positive_bridge_rate": positive_rate,
        "mean_similarity_claim_only": float(df["sim_claim"].mean()),
        "mean_similarity_with_assumptions": float(df["sim_context"].mean()),
        "mean_similarity_same_episode_implicit_sample": float(df["sim_same_episode_sample"].mean()),
        "mean_similarity_any_episode_implicit_sample": float(df["sim_global_sample"].mean()),
        "mean_similarity_gain_percent": float(
            100.0 * (df["sim_context"].mean() - df["sim_claim"].mean()) / max(abs(df["sim_claim"].mean()), 1e-9)
        ),
        "bridge_delta_bootstrap": boot,
        "patch_merge": {
            "merged_patch_count": int(len(patch_dirs)),
            "num_patches": int(patch_summaries[0]["num_patches"]),
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
        },
        "notes": [
            "This summary is merged from completed Exp 1 patch outputs.",
            "Only turns with turn_type_label == 'Substantive' are included.",
            "Turn A is vectorized from explicit propositions when available, otherwise raw turn text.",
            "Turn B claims come from explicit_propositions; context adds assumptions from the same turn.",
            "Matched implicit baselines sample the same number of assumptions as Turn B from the same episode or from the corpus-wide pool.",
            "Cosine similarity is computed on L2-normalized sentence embeddings from the selected embedding model.",
        ],
    }


def main():
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

    distribution_plot(df, dist_png)
    df.to_csv(pair_csv, index=False)

    by_category = build_by_category(df)
    by_category.to_csv(category_csv, index=False)

    long_plot_df = pointplot_long_frame(df)
    pointplot_summary_df = pointplot_summary(long_plot_df, seed=args.seed)
    pointplot_summary_df.to_csv(pointplot_csv, index=False)
    bootstrap_pointplot(long_plot_df, pointplot_png, category_order=categories, seed=args.seed)

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
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
