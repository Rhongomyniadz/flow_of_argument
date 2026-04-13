import re
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/conversation_moves_labeled")
DEFAULT_OUTPUT_DIR = Path("experiments/exp1_relevance_bridge/results")
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QWEN_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEFAULT_TARGET_EMBEDDING_MAX_LENGTH = 1024
DEFAULT_BOOTSTRAP_DRAWS = 2000
PAIR_EXPORT_COLUMNS = [
    "category",
    "episode_id",
    "turn_a_idx",
    "turn_b_idx",
    "turn_b_has_assumptions",
    "sim_claim",
    "sim_context",
    "sim_same_episode_sample",
    "sim_global_sample",
    "bridge_delta",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--max_episodes_per_category", type=int, default=None)
    ap.add_argument("--num_patches", type=int, default=1)
    ap.add_argument("--patch_index", type=int, default=0)
    ap.add_argument("--episodes_per_patch", type=int, default=None)
    ap.add_argument("--global_assumption_pool_path", type=Path, default=None)
    ap.add_argument("--embedding_batch_size", type=int, default=128)
    ap.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_tqdm", action="store_true")
    return ap.parse_args()


def resolve_output_dir(base_output_dir: Path, embedding_model_name: str):
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embedding_model_name.replace("/", "__").strip())
    if not model_slug:
        raise ValueError("embedding_model_name must not be empty.")
    return base_output_dir / model_slug


def validate_patch_args(num_patches: int, patch_index: int, episodes_per_patch):
    if num_patches < 1:
        raise ValueError(f"num_patches must be >= 1, got {num_patches}")
    if patch_index < 0 or patch_index >= num_patches:
        raise ValueError(f"patch_index must be in [0, {num_patches - 1}], got {patch_index}")
    if episodes_per_patch is not None and episodes_per_patch < 1:
        raise ValueError(f"episodes_per_patch must be >= 1, got {episodes_per_patch}")


def resolve_patch_output_dir(base_output_dir: Path, num_patches: int, patch_index: int):
    if num_patches == 1:
        return base_output_dir
    return base_output_dir / "patches" / f"patch_{patch_index:04d}_of_{num_patches:04d}"


def normalize_categories(input_dir: Path, requested):
    available = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    if not requested or any(str(x).lower() == "all" for x in requested):
        return available
    lookup = {name.lower(): name for name in available}
    chosen = []
    for raw in requested:
        match = lookup.get(str(raw).lower())
        if match is None:
            raise ValueError(f"Unknown category: {raw}. Available: {', '.join(available)}")
        if match not in chosen:
            chosen.append(match)
    return chosen


def collect_category_files(input_dir: Path, categories, max_episodes_per_category):
    files = []
    for category in categories:
        category_files = sorted((input_dir / category).glob("*.json"))
        if max_episodes_per_category:
            category_files = category_files[:max_episodes_per_category]
        files.extend((category, path) for path in category_files)
    return files


def select_patch_files(category_files, num_patches: int, patch_index: int, episodes_per_patch):
    if episodes_per_patch is not None:
        start = patch_index * episodes_per_patch
        end = min(start + episodes_per_patch, len(category_files))
        return category_files[start:end]
    return [item for idx, item in enumerate(category_files) if idx % num_patches == patch_index]


def load_turns(path: Path):
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("turns", [])


def turn_time(turn):
    raw = turn.get("start_time", turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))))
    return float(raw if raw is not None else 0.0)


def item_texts(items):
    vals = []
    for item in items or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            vals.append(text)
    return vals


def item_text(items):
    return " ".join(item_texts(items)).strip()


def substantive_pairs(path: Path):
    turns = [t for t in load_turns(path) if str(t.get("turn_type_label") or "").strip() == "Substantive"]
    turns.sort(key=turn_time)
    rows = []
    for a, b in zip(turns, turns[1:]):
        episode_id = str(b.get("episode_id") or path.stem)
        turn_a_idx = int(a.get("turn_idx", -1))
        turn_b_idx = int(b.get("turn_idx", -1))
        a_text = item_text(a.get("explicit_propositions")) or str(a.get("turn_text") or "").strip()
        b_claim = item_text(b.get("explicit_propositions"))
        if not a_text or not b_claim:
            continue
        b_assumption_texts = item_texts(b.get("assumptions"))
        b_assumptions = " ".join(b_assumption_texts).strip()
        rows.append(
            {
                "episode_id": episode_id,
                "turn_a_idx": turn_a_idx,
                "turn_b_idx": turn_b_idx,
                "turn_a_text": a_text,
                "turn_b_claim_text": b_claim,
                "turn_b_context_text": f"{b_claim} {b_assumptions}".strip(),
                "turn_b_assumption_count": len(b_assumption_texts),
                "turn_b_assumption_texts": b_assumption_texts,
                "turn_b_assumption_ids": [
                    f"{episode_id}:{turn_b_idx}:{assumption_idx}"
                    for assumption_idx, _ in enumerate(b_assumption_texts)
                ],
                "turn_b_has_assumptions": int(bool(b_assumptions)),
            }
        )
    return rows


def collect_global_assumption_pool(category_files):
    global_pool = []
    for _, path in category_files:
        for row in substantive_pairs(path):
            for assumption_id, assumption_text in zip(
                row["turn_b_assumption_ids"],
                row["turn_b_assumption_texts"],
            ):
                global_pool.append((str(assumption_id), str(assumption_text)))
    return global_pool


def load_global_assumption_pool(path: Path):
    global_pool = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            global_pool.append((str(payload["assumption_id"]), str(payload["assumption_text"])))
    return global_pool


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def resolve_embedding_max_length(tokenizer):
    raw_model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(raw_model_max_length, int):
        return DEFAULT_TARGET_EMBEDDING_MAX_LENGTH
    if raw_model_max_length <= 0:
        return DEFAULT_TARGET_EMBEDDING_MAX_LENGTH
    if raw_model_max_length > 100000:
        return DEFAULT_TARGET_EMBEDDING_MAX_LENGTH
    return min(DEFAULT_TARGET_EMBEDDING_MAX_LENGTH, raw_model_max_length)


def embed_texts(texts, batch_size, use_tqdm, embedding_model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True).to(device).eval()
    embedding_max_length = resolve_embedding_max_length(tokenizer)
    unique = list(dict.fromkeys(texts))
    vectors = {}
    iterator = range(0, len(unique), batch_size)
    iterator = tqdm(iterator, desc="Embedding texts", disable=not use_tqdm)
    with torch.inference_mode():
        for start in iterator:
            batch = unique[start:start + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=embedding_max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            for text, vec in zip(batch, pooled):
                vectors[text] = vec.astype(np.float32, copy=False)
    return vectors


def bootstrap_mean(values, seed, draws=2000):
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(draws, len(vals)))
    means = vals[idx].mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def build_assumption_pools(df: pd.DataFrame):
    episode_pools = {}
    global_pool = []
    for episode_id, assumption_ids, assumption_texts in zip(
        df["episode_id"],
        df["turn_b_assumption_ids"],
        df["turn_b_assumption_texts"],
    ):
        if not isinstance(assumption_ids, list) or not isinstance(assumption_texts, list):
            continue
        for assumption_id, assumption_text in zip(assumption_ids, assumption_texts):
            record = (str(assumption_id), str(assumption_text))
            episode_pools.setdefault(str(episode_id), []).append(record)
            global_pool.append(record)
    return episode_pools, global_pool


def sample_assumption_texts(pool_records, excluded_ids, sample_count, rng):
    if sample_count <= 0:
        return []
    candidate_records = [record for record in pool_records if record[0] not in excluded_ids]
    if not candidate_records:
        candidate_records = list(pool_records)
    if not candidate_records:
        return []
    replace = len(candidate_records) < sample_count
    sample_indices = rng.choice(len(candidate_records), size=sample_count, replace=replace)
    return [candidate_records[int(idx)][1] for idx in np.asarray(sample_indices).tolist()]


def add_sampled_contexts(df: pd.DataFrame, seed: int, global_pool_records=None):
    episode_pools, local_global_pool = build_assumption_pools(df)
    global_pool = list(global_pool_records) if global_pool_records is not None else local_global_pool
    rng = np.random.default_rng(seed)

    same_episode_contexts = []
    global_contexts = []
    for row in df.itertuples(index=False):
        sample_count = int(row.turn_b_assumption_count)
        excluded_ids = set(row.turn_b_assumption_ids)

        same_episode_sample = sample_assumption_texts(
            episode_pools.get(str(row.episode_id), []),
            excluded_ids,
            sample_count,
            rng,
        )
        global_sample = sample_assumption_texts(
            global_pool,
            excluded_ids,
            sample_count,
            rng,
        )
        same_episode_contexts.append(f"{row.turn_b_claim_text} {' '.join(same_episode_sample).strip()}".strip())
        global_contexts.append(f"{row.turn_b_claim_text} {' '.join(global_sample).strip()}".strip())

    df = df.copy()
    df["turn_b_same_episode_context_text"] = same_episode_contexts
    df["turn_b_global_context_text"] = global_contexts
    return df


def pointplot_long_frame(df: pd.DataFrame):
    value_columns = [
        ("Claim Only", "sim_claim"),
        ("With Assumptions", "sim_context"),
        ("Matched Implicit Sample (Same Episode)", "sim_same_episode_sample"),
        ("Matched Implicit Sample (Any Episode)", "sim_global_sample"),
    ]
    frames = []
    for condition, column_name in value_columns:
        part = df[["category", column_name]].rename(columns={column_name: "cosine_similarity"})
        part["comparison"] = condition
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def pointplot_summary(long_df: pd.DataFrame, seed: int):
    rows = []
    for summary_idx, ((category, comparison), group_df) in enumerate(
        long_df.groupby(["category", "comparison"], sort=False, observed=False)
    ):
        boot = bootstrap_mean(group_df["cosine_similarity"], seed=seed + summary_idx, draws=DEFAULT_BOOTSTRAP_DRAWS)
        rows.append(
            {
                "category": category,
                "comparison": comparison,
                "pair_count": int(len(group_df)),
                "mean_cosine_similarity": boot["mean"],
                "ci95_low": boot["ci95_low"],
                "ci95_high": boot["ci95_high"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_pointplot(long_df: pd.DataFrame, path: Path, category_order, seed: int):
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    hue_order = [
        "Claim Only",
        "With Assumptions",
        "Matched Implicit Sample (Same Episode)",
        "Matched Implicit Sample (Any Episode)",
    ]
    palette = {
        "Claim Only": "#d95f02",
        "With Assumptions": "#1b9e77",
        "Matched Implicit Sample (Same Episode)": "#7570b3",
        "Matched Implicit Sample (Any Episode)": "#e7298a",
    }
    sns.pointplot(
        data=long_df,
        x="cosine_similarity",
        y="category",
        hue="comparison",
        order=list(category_order),
        hue_order=hue_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        n_boot=DEFAULT_BOOTSTRAP_DRAWS,
        seed=seed,
        dodge=0.6,
        linestyles="none",
        palette=palette,
        markers=["o", "s", "D", "^"],
        ax=ax,
    )
    ax.set_title("Relevance Bridge With Matched Implicit Baselines", fontsize=15)
    ax.set_xlabel("Cosine Similarity With Previous Substantive Turn")
    ax.set_ylabel("Category")
    ax.legend(title=None, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def distribution_plot(df: pd.DataFrame, path: Path):
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    bins = np.linspace(0, 1, 60)
    ax.hist(df["sim_claim"], bins=bins, density=True, alpha=0.5, color="#d95f02", label="Claims Only")
    ax.hist(df["sim_context"], bins=bins, density=True, alpha=0.5, color="#1b9e77", label="With Assumptions")
    ax.axvline(df["sim_claim"].mean(), color="#d95f02", linestyle="--", linewidth=2)
    ax.axvline(df["sim_context"].mean(), color="#1b9e77", linestyle="--", linewidth=2)
    ax.set_title("Relevance Bridge Distribution", fontsize=15)
    ax.set_xlabel("Cosine Similarity With Previous Substantive Turn")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_pair_csv(df: pd.DataFrame, path: Path):
    df[PAIR_EXPORT_COLUMNS].to_csv(path, index=False)


def build_patch_summary(
    args,
    output_dir: Path,
    categories,
    selected_files,
    category_files,
    global_pool_records,
    global_pool_source: str,
    df: pd.DataFrame,
    pair_csv: Path,
    summary_json: Path,
):
    return {
        "experiment": "Experiment 1: The Relevance Bridge",
        "analysis_stage": "patch_pair_extraction_only",
        "input_dir": str(args.input_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "num_patches": int(args.num_patches),
        "patch_index": int(args.patch_index),
        "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
        "selected_episode_file_count": int(len(selected_files)),
        "candidate_episode_file_count": int(len(category_files)),
        "global_assumption_pool_size": int(len(global_pool_records)),
        "global_assumption_pool_scope": "full_selected_corpus",
        "global_assumption_pool_source": global_pool_source,
        "global_assumption_pool_path": (
            str(args.global_assumption_pool_path)
            if args.global_assumption_pool_path is not None
            else None
        ),
        "categories": categories,
        "total_pairs": int(len(df)),
        "pairs_with_assumptions_on_turn_b": int(df["turn_b_has_assumptions"].sum()),
        "assumption_pair_rate": float(df["turn_b_has_assumptions"].mean()),
        "outputs": {
            "pair_csv": str(pair_csv),
            "summary_json": str(summary_json),
        },
        "deferred_outputs": [
            "exp1_bridge_by_category.csv",
            "exp1_similarity_pointplot_summary.csv",
            "exp1_similarity_pointplot.png",
            "exp1_distance_distribution.png",
        ],
        "notes": [
            "Patch mode writes adjacent-pair similarity rows only.",
            "Run merge_exp1_patches.py after all patches finish to build the category summaries and plots.",
        ],
    }


def main():
    args = parse_args()
    validate_patch_args(args.num_patches, args.patch_index, args.episodes_per_patch)
    model_output_dir = resolve_output_dir(args.output_dir, args.embedding_model_name)
    output_dir = resolve_patch_output_dir(model_output_dir, args.num_patches, args.patch_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = normalize_categories(args.input_dir, args.categories)
    use_tqdm = not args.no_tqdm

    category_files = collect_category_files(args.input_dir, categories, args.max_episodes_per_category)
    selected_files = select_patch_files(
        category_files,
        args.num_patches,
        args.patch_index,
        args.episodes_per_patch,
    )
    if not selected_files:
        raise RuntimeError(
            f"No episode files selected for patch {args.patch_index} out of {args.num_patches}. "
            f"Candidate file count: {len(category_files)}"
        )
    logger.info(
        "Selected %d episode files for patch %d/%d.",
        len(selected_files),
        args.patch_index + 1,
        args.num_patches,
    )
    rows = []
    for category in categories:
        files = [path for file_category, path in selected_files if file_category == category]
        iterator = tqdm(files, desc=f"{category}: pairs", disable=not use_tqdm)
        for path in iterator:
            for row in substantive_pairs(path):
                row["category"] = category
                rows.append(row)

    if not rows:
        raise RuntimeError("No valid substantive adjacent pairs found.")

    df = pd.DataFrame(rows)
    if args.global_assumption_pool_path is not None:
        global_pool_records = load_global_assumption_pool(args.global_assumption_pool_path)
        global_pool_source = "precomputed_file"
    else:
        global_pool_records = collect_global_assumption_pool(category_files)
        global_pool_source = "computed_in_process"
    df = add_sampled_contexts(df, seed=args.seed, global_pool_records=global_pool_records)
    logger.info("Collected %d adjacent substantive pairs across %d categories.", len(df), len(categories))

    text_to_vec = embed_texts(
        pd.concat(
            [
                df["turn_a_text"],
                df["turn_b_claim_text"],
                df["turn_b_context_text"],
                df["turn_b_same_episode_context_text"],
                df["turn_b_global_context_text"],
            ],
            ignore_index=True,
        ).tolist(),
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=args.embedding_model_name,
    )
    df["vec_a"] = df["turn_a_text"].map(text_to_vec)
    df["vec_claim"] = df["turn_b_claim_text"].map(text_to_vec)
    df["vec_context"] = df["turn_b_context_text"].map(text_to_vec)
    df["vec_same_episode_sample"] = df["turn_b_same_episode_context_text"].map(text_to_vec)
    df["vec_global_sample"] = df["turn_b_global_context_text"].map(text_to_vec)
    df["sim_claim"] = [float(np.dot(a, b)) for a, b in zip(df["vec_a"], df["vec_claim"])]
    df["sim_context"] = [float(np.dot(a, b)) for a, b in zip(df["vec_a"], df["vec_context"])]
    df["sim_same_episode_sample"] = [
        float(np.dot(a, b)) for a, b in zip(df["vec_a"], df["vec_same_episode_sample"])
    ]
    df["sim_global_sample"] = [
        float(np.dot(a, b)) for a, b in zip(df["vec_a"], df["vec_global_sample"])
    ]
    df["bridge_delta"] = df["sim_context"] - df["sim_claim"]

    pair_csv = output_dir / "exp1_bridge_pairs.csv"
    summary_json = output_dir / "exp1_summary.json"
    write_pair_csv(df, pair_csv)

    if args.num_patches > 1:
        patch_summary = build_patch_summary(
            args=args,
            output_dir=output_dir,
            categories=categories,
            selected_files=selected_files,
            category_files=category_files,
            global_pool_records=global_pool_records,
            global_pool_source=global_pool_source,
            df=df,
            pair_csv=pair_csv,
            summary_json=summary_json,
        )
        summary_json.write_text(json.dumps(patch_summary, indent=2))
        logger.info("Done. Wrote patch features to %s", output_dir)
        return

    category_csv = output_dir / "exp1_bridge_by_category.csv"
    pointplot_csv = output_dir / "exp1_similarity_pointplot_summary.csv"
    pointplot_png = output_dir / "exp1_similarity_pointplot.png"
    dist_png = output_dir / "exp1_distance_distribution.png"

    distribution_plot(df, dist_png)

    by_category = (
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
    by_category.to_csv(category_csv, index=False)

    long_plot_df = pointplot_long_frame(df)
    pointplot_summary_df = pointplot_summary(long_plot_df, seed=args.seed)
    pointplot_summary_df.to_csv(pointplot_csv, index=False)
    bootstrap_pointplot(long_plot_df, pointplot_png, category_order=categories, seed=args.seed)

    boot = bootstrap_mean(df["bridge_delta"], seed=args.seed)
    positive_rate = float((df["bridge_delta"] > 0).mean())
    summary = {
        "experiment": "Experiment 1: The Relevance Bridge",
        "analysis_stage": "full_analysis",
        "input_dir": str(args.input_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "num_patches": int(args.num_patches),
        "patch_index": int(args.patch_index),
        "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
        "selected_episode_file_count": int(len(selected_files)),
        "candidate_episode_file_count": int(len(category_files)),
        "global_assumption_pool_size": int(len(global_pool_records)),
        "global_assumption_pool_scope": "full_selected_corpus",
        "global_assumption_pool_source": global_pool_source,
        "global_assumption_pool_path": (
            str(args.global_assumption_pool_path)
            if args.global_assumption_pool_path is not None
            else None
        ),
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
        "outputs": {
            "pair_csv": str(pair_csv),
            "category_csv": str(category_csv),
            "pointplot_summary_csv": str(pointplot_csv),
            "pointplot_png": str(pointplot_png),
            "summary_json": str(summary_json),
            "distance_distribution_png": str(dist_png),
        },
        "notes": [
            "Only turns with turn_type_label == 'Substantive' are included.",
            "Turn A is vectorized from explicit propositions when available, otherwise raw turn text.",
            "Turn B claims come from explicit_propositions; context adds assumptions from the same turn.",
            "Matched implicit baselines sample the same number of assumptions as Turn B from the same episode or from the corpus-wide pool.",
            "Cosine similarity is computed on L2-normalized sentence embeddings from the selected embedding model.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    logger.info("Done. Wrote results to %s", output_dir)


if __name__ == "__main__":
    main()
