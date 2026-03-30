import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
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
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--max_episodes_per_category", type=int, default=None)
    ap.add_argument("--embedding_batch_size", type=int, default=128)
    ap.add_argument("--umap_pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_tqdm", action="store_true")
    return ap.parse_args()


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


def load_turns(path: Path):
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("turns", [])


def turn_time(turn):
    raw = turn.get("start_time", turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))))
    return float(raw if raw is not None else 0.0)


def item_text(items):
    vals = []
    for item in items or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            vals.append(text)
    return " ".join(vals).strip()


def substantive_pairs(path: Path):
    turns = [t for t in load_turns(path) if str(t.get("turn_type_label") or "").strip() == "Substantive"]
    turns.sort(key=turn_time)
    rows = []
    for a, b in zip(turns, turns[1:]):
        a_text = item_text(a.get("explicit_propositions")) or str(a.get("turn_text") or "").strip()
        b_claim = item_text(b.get("explicit_propositions"))
        if not a_text or not b_claim:
            continue
        b_assumptions = item_text(b.get("assumptions"))
        rows.append(
            {
                "episode_id": str(b.get("episode_id") or path.stem),
                "turn_a_idx": int(a.get("turn_idx", -1)),
                "turn_b_idx": int(b.get("turn_idx", -1)),
                "turn_a_text": a_text,
                "turn_b_claim_text": b_claim,
                "turn_b_context_text": f"{b_claim} {b_assumptions}".strip(),
                "turn_b_has_assumptions": int(bool(b_assumptions)),
            }
        )
    return rows


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts, batch_size, use_tqdm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
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
                max_length=256,
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


def distribution_plot(df: pd.DataFrame, path: Path):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 60)
    ax.hist(df["sim_claim"], bins=bins, density=True, alpha=0.5, color="#d95f02", label="Claims Only")
    ax.hist(df["sim_context"], bins=bins, density=True, alpha=0.5, color="#1b9e77", label="With Assumptions")
    ax.axvline(df["sim_claim"].mean(), color="#d95f02", linestyle="--", linewidth=2)
    ax.axvline(df["sim_context"].mean(), color="#1b9e77", linestyle="--", linewidth=2)
    ax.set_title("Experiment 1: Relevance Bridge Distribution")
    ax.set_xlabel("Cosine Similarity With Previous Substantive Turn")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def umap_plot(df: pd.DataFrame, path: Path, pair_limit: int, seed: int):
    if df.empty:
        return None
    sample = df.sample(n=min(pair_limit, len(df)), random_state=seed).reset_index(drop=True)
    stack = np.vstack(
        [
            np.stack(sample["vec_a"].to_list()),
            np.stack(sample["vec_claim"].to_list()),
            np.stack(sample["vec_context"].to_list()),
        ]
    )
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(stack)
    n = len(sample)
    a_xy = coords[:n]
    c_xy = coords[n:2 * n]
    x_xy = coords[2 * n:]
    claim_len = np.linalg.norm(a_xy - c_xy, axis=1)
    context_len = np.linalg.norm(a_xy - x_xy, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, target_xy, color, title, mean_len in [
        (axes[0], c_xy, "#d95f02", "A -> Claims", claim_len.mean()),
        (axes[1], x_xy, "#1b9e77", "A -> Claims + Assumptions", context_len.mean()),
    ]:
        for p0, p1 in zip(a_xy, target_xy):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, alpha=0.05, linewidth=0.8)
        ax.scatter(a_xy[:, 0], a_xy[:, 1], s=8, color="#4c4c4c", alpha=0.20, label="Turn A")
        ax.scatter(target_xy[:, 0], target_xy[:, 1], s=8, color=color, alpha=0.20, label=title.split(" -> ", 1)[1])
        ax.set_title(f"{title}\nMean UMAP Step = {mean_len:.3f}")
        ax.set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, loc="best")
    fig.suptitle("Experiment 1: Conversation Trajectory Smoothing", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    sample[["umap_a_x", "umap_a_y"]] = a_xy
    sample[["umap_claim_x", "umap_claim_y"]] = c_xy
    sample[["umap_context_x", "umap_context_y"]] = x_xy
    sample["umap_step_claim"] = claim_len
    sample["umap_step_context"] = context_len
    return {
        "sample_size": int(n),
        "mean_umap_step_claim": float(claim_len.mean()),
        "mean_umap_step_context": float(context_len.mean()),
        "mean_umap_step_delta": float((claim_len - context_len).mean()),
        "sample_frame": sample.drop(columns=["vec_a", "vec_claim", "vec_context"]),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = normalize_categories(args.input_dir, args.categories)
    use_tqdm = not args.no_tqdm

    rows = []
    for category in categories:
        files = sorted((args.input_dir / category).glob("*.json"))
        if args.max_episodes_per_category:
            files = files[:args.max_episodes_per_category]
        iterator = tqdm(files, desc=f"{category}: pairs", disable=not use_tqdm)
        for path in iterator:
            for row in substantive_pairs(path):
                row["category"] = category
                rows.append(row)

    if not rows:
        raise RuntimeError("No valid substantive adjacent pairs found.")

    df = pd.DataFrame(rows)
    logger.info("Collected %d adjacent substantive pairs across %d categories.", len(df), len(categories))

    text_to_vec = embed_texts(
        pd.concat(
            [df["turn_a_text"], df["turn_b_claim_text"], df["turn_b_context_text"]],
            ignore_index=True,
        ).tolist(),
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
    )
    df["vec_a"] = df["turn_a_text"].map(text_to_vec)
    df["vec_claim"] = df["turn_b_claim_text"].map(text_to_vec)
    df["vec_context"] = df["turn_b_context_text"].map(text_to_vec)
    df["sim_claim"] = [float(np.dot(a, b)) for a, b in zip(df["vec_a"], df["vec_claim"])]
    df["sim_context"] = [float(np.dot(a, b)) for a, b in zip(df["vec_a"], df["vec_context"])]
    df["bridge_delta"] = df["sim_context"] - df["sim_claim"]

    pair_csv = args.output_dir / "exp1_bridge_pairs.csv"
    category_csv = args.output_dir / "exp1_bridge_by_category.csv"
    summary_json = args.output_dir / "exp1_summary.json"
    dist_png = args.output_dir / "exp1_distance_distribution.png"
    umap_png = args.output_dir / "exp1_umap_trajectory.png"
    umap_csv = args.output_dir / "exp1_umap_sample.csv"

    distribution_plot(df, dist_png)
    umap_info = umap_plot(df, umap_png, args.umap_pairs, args.seed)
    if umap_info is not None:
        umap_info["sample_frame"].to_csv(umap_csv, index=False)

    export_cols = [
        "category",
        "episode_id",
        "turn_a_idx",
        "turn_b_idx",
        "turn_b_has_assumptions",
        "sim_claim",
        "sim_context",
        "bridge_delta",
    ]
    df[export_cols].to_csv(pair_csv, index=False)

    by_category = (
        df.groupby("category", as_index=False)
        .agg(
            pair_count=("bridge_delta", "size"),
            mean_sim_claim=("sim_claim", "mean"),
            mean_sim_context=("sim_context", "mean"),
            mean_bridge_delta=("bridge_delta", "mean"),
            positive_bridge_rate=("bridge_delta", lambda x: float((x > 0).mean())),
            assumption_pair_rate=("turn_b_has_assumptions", "mean"),
        )
        .sort_values("mean_bridge_delta", ascending=False)
    )
    by_category.to_csv(category_csv, index=False)

    boot = bootstrap_mean(df["bridge_delta"], seed=args.seed)
    positive_rate = float((df["bridge_delta"] > 0).mean())
    summary = {
        "experiment": "Experiment 1: The Relevance Bridge",
        "input_dir": str(args.input_dir),
        "categories": categories,
        "total_pairs": int(len(df)),
        "pairs_with_assumptions_on_turn_b": int(df["turn_b_has_assumptions"].sum()),
        "assumption_pair_rate": float(df["turn_b_has_assumptions"].mean()),
        "bridge_score_mean_delta": float(df["bridge_delta"].mean()),
        "bridge_score_median_delta": float(df["bridge_delta"].median()),
        "positive_bridge_rate": positive_rate,
        "mean_similarity_claim_only": float(df["sim_claim"].mean()),
        "mean_similarity_with_assumptions": float(df["sim_context"].mean()),
        "mean_similarity_gain_percent": float(
            100.0 * (df["sim_context"].mean() - df["sim_claim"].mean()) / max(abs(df["sim_claim"].mean()), 1e-9)
        ),
        "bridge_delta_bootstrap": boot,
        "umap_topology": None if umap_info is None else {
            "sample_size": umap_info["sample_size"],
            "mean_umap_step_claim": umap_info["mean_umap_step_claim"],
            "mean_umap_step_context": umap_info["mean_umap_step_context"],
            "mean_umap_step_delta": umap_info["mean_umap_step_delta"],
        },
        "outputs": {
            "pair_csv": str(pair_csv),
            "category_csv": str(category_csv),
            "summary_json": str(summary_json),
            "distance_distribution_png": str(dist_png),
            "umap_png": str(umap_png),
            "umap_sample_csv": str(umap_csv) if umap_info is not None else None,
        },
        "notes": [
            "Only turns with turn_type_label == 'Substantive' are included.",
            "Turn A is vectorized from explicit propositions when available, otherwise raw turn text.",
            "Turn B claims come from explicit_propositions; context adds assumptions from the same turn.",
            "Cosine similarity is computed on L2-normalized MiniLM embeddings.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    logger.info("Done. Wrote results to %s", args.output_dir)


if __name__ == "__main__":
    main()
