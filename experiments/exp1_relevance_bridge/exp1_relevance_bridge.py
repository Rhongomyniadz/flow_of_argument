import argparse
import hashlib
import json
import logging
import re
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
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_TARGET_EMBEDDING_MAX_LENGTH = 1024
DEFAULT_BOOTSTRAP_DRAWS = 2000
UNRELATED_SENTENCES_PATH = Path(__file__).with_name("unrelated_sentences.json")
UNRELATED_SENTENCE_POOL_SIZE = 100
BASELINE_SENTENCE_SAMPLE_SIZE = 10
UMAP_MAX_EPISODES = 12
UMAP_MAX_POINTS_PER_EPISODE = 30
UMAP_NEIGHBORS = 18
UMAP_MIN_DIST = 0.18
PAIR_EXPORT_COLUMNS = [
    "category",
    "episode_id",
    "turn_a_idx",
    "turn_b_idx",
    "turn_b_has_assumptions",
    "candidate_assumption_count",
    "sim_claim",
    "sim_context",
    "sim_unrelated_sentences_only",
    "bridge_delta",
    "turn_a_text",
    "turn_b_claim_text",
    "turn_b_context_text",
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
    ap.add_argument("--embedding_batch_size", type=int, default=128)
    ap.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    ap.add_argument(
        "--embedding_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_EMBEDDING_DEVICE,
    )
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
    available = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    if not requested or any(str(item).lower() == "all" for item in requested):
        return available
    lookup = {name.lower(): name for name in available}
    chosen = []
    for raw_name in requested:
        match = lookup.get(str(raw_name).lower())
        if match is None:
            raise ValueError(f"Unknown category: {raw_name}. Available: {', '.join(available)}")
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


def load_unrelated_sentences(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing unrelated sentences file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("unrelated_sentences.json must contain a top-level JSON array of strings.")
    sentences: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise TypeError(
                f"unrelated_sentences.json item {index} must be a string, got {type(item).__name__}."
            )
        sentence = item.strip()
        if not sentence:
            raise ValueError(f"unrelated_sentences.json item {index} is empty after trimming whitespace.")
        if sentence in seen:
            raise ValueError(f"unrelated_sentences.json contains a duplicate sentence at item {index}.")
        seen.add(sentence)
        sentences.append(sentence)
    if len(sentences) != UNRELATED_SENTENCE_POOL_SIZE:
        raise ValueError(
            "unrelated_sentences.json must contain exactly "
            f"{UNRELATED_SENTENCE_POOL_SIZE} unique sentences, got {len(sentences)}."
        )
    return sentences


def turn_time(turn):
    raw = turn.get("start_time", turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))))
    return float(raw if raw is not None else 0.0)


def item_texts(items):
    texts = []
    for item in items or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            texts.append(text)
    return texts


def item_text(items):
    return " ".join(item_texts(items)).strip()


def substantive_pairs(path: Path):
    turns = [turn for turn in load_turns(path) if str(turn.get("turn_type_label") or "").strip() == "Substantive"]
    turns.sort(key=turn_time)
    rows = []
    for first_turn, second_turn in zip(turns, turns[1:]):
        episode_id = str(second_turn.get("episode_id") or path.stem)
        turn_a_idx = int(first_turn.get("turn_idx", -1))
        turn_b_idx = int(second_turn.get("turn_idx", -1))
        turn_a_text = item_text(first_turn.get("explicit_propositions")) or str(first_turn.get("turn_text") or "").strip()
        turn_b_claim_text = item_text(second_turn.get("explicit_propositions"))
        if not turn_a_text or not turn_b_claim_text:
            continue
        turn_b_assumption_texts = item_texts(second_turn.get("assumptions"))
        rows.append(
            {
                "episode_id": episode_id,
                "turn_a_idx": turn_a_idx,
                "turn_b_idx": turn_b_idx,
                "turn_a_text": turn_a_text,
                "turn_b_claim_text": turn_b_claim_text,
                "turn_b_assumption_texts": turn_b_assumption_texts,
                "turn_b_assumption_ids": [
                    f"{episode_id}:{turn_b_idx}:{assumption_idx}"
                    for assumption_idx, _ in enumerate(turn_b_assumption_texts)
                ],
                "turn_b_context_text": " ".join([turn_b_claim_text, *turn_b_assumption_texts]).strip(),
                "candidate_assumption_count": len(turn_b_assumption_texts),
                "turn_b_has_assumptions": int(bool(turn_b_assumption_texts)),
            }
        )
    return rows


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


def resolve_embedding_device(requested_device: str) -> torch.device:
    normalized_device = requested_device.strip().lower()
    if normalized_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_device == "cpu":
        return torch.device("cpu")
    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("embedding_device='cuda' was requested, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported embedding device: {requested_device}")


def embed_texts(texts, batch_size, use_tqdm, embedding_model_name, embedding_device):
    device = resolve_embedding_device(embedding_device)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True).to(device).eval()
    except torch.OutOfMemoryError as error:
        if device.type != "cuda":
            raise
        raise RuntimeError(
            "CUDA ran out of memory while loading the embedding model. "
            f"model={embedding_model_name}, embedding_device={device.type}. "
            "Free GPU memory or rerun with --embedding_device cpu."
        ) from error
    embedding_max_length = resolve_embedding_max_length(tokenizer)
    unique_texts = list(dict.fromkeys(texts))
    vectors = {}
    logger.info(
        "Embedding %d unique texts with model=%s on device=%s and batch_size=%d.",
        len(unique_texts),
        embedding_model_name,
        device.type,
        batch_size,
    )
    iterator = tqdm(
        range(0, len(unique_texts), batch_size),
        desc="Embedding texts",
        disable=not use_tqdm,
    )
    with torch.inference_mode():
        for start in iterator:
            batch = unique_texts[start:start + batch_size]
            try:
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=embedding_max_length,
                    return_tensors="pt",
                )
                tokens = {name: value.to(device) for name, value in tokens.items()}
                output = model(**tokens)
            except torch.OutOfMemoryError as error:
                if device.type != "cuda":
                    raise
                raise RuntimeError(
                    "CUDA ran out of memory while embedding texts. "
                    f"model={embedding_model_name}, embedding_device={device.type}, batch_size={batch_size}. "
                    "Reduce --embedding_batch_size, free GPU memory, or rerun with --embedding_device cpu."
                ) from error
            pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            for text, vector in zip(batch, pooled):
                vectors[text] = vector.astype(np.float32, copy=False)
    return vectors


def bootstrap_mean(values, seed, draws=2000):
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(vals), size=(draws, len(vals)))
    means = vals[sample_indices].mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def compose_normalized_mean(vectors):
    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    composed = matrix.mean(axis=0)
    norm = float(np.linalg.norm(composed))
    if norm <= 1e-9:
        return matrix[0].copy()
    return composed / norm


def cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b))


def build_row_rng(seed: int, episode_id: str, turn_a_idx: int, turn_b_idx: int) -> np.random.Generator:
    seed_material = f"{seed}:{episode_id}:{turn_a_idx}:{turn_b_idx}".encode("utf-8")
    row_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
    return np.random.default_rng(row_seed)


def sample_unrelated_sentences(
    unrelated_sentences: list[str],
    seed: int,
    episode_id: str,
    turn_a_idx: int,
    turn_b_idx: int,
) -> list[str]:
    if len(unrelated_sentences) < BASELINE_SENTENCE_SAMPLE_SIZE:
        raise ValueError(
            "Unrelated sentence pool is too small for the baseline sample size. "
            f"pool_size={len(unrelated_sentences)}, sample_size={BASELINE_SENTENCE_SAMPLE_SIZE}"
        )
    row_rng = build_row_rng(seed, episode_id, turn_a_idx, turn_b_idx)
    sampled_indices = row_rng.choice(
        len(unrelated_sentences),
        size=BASELINE_SENTENCE_SAMPLE_SIZE,
        replace=False,
    )
    return [unrelated_sentences[int(idx)] for idx in np.asarray(sampled_indices).tolist()]


def select_bridge_context(vec_a, vec_claim, assumption_records):
    sim_claim = cosine_similarity(vec_a, vec_claim)
    if not assumption_records:
        return {
            "sim_claim": sim_claim,
            "sim_context": sim_claim,
            "context_assumption_count": 0,
        }

    context_vec = compose_normalized_mean([vec_claim] + [record["vec"] for record in assumption_records])

    return {
        "sim_claim": sim_claim,
        "sim_context": cosine_similarity(vec_a, context_vec),
        "context_assumption_count": len(assumption_records),
    }


def add_bridge_metrics(df: pd.DataFrame, text_to_vec, unrelated_sentences: list[str], seed: int, use_tqdm: bool):
    sim_claim_values = []
    sim_context_values = []
    sim_unrelated_sentences_only_values = []
    bridge_deltas = []

    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Scoring relevance bridges", disable=not use_tqdm)
    for row in iterator:
        vec_a = text_to_vec[row.turn_a_text]
        vec_claim = text_to_vec[row.turn_b_claim_text]
        assumption_records = [
            {
                "id": assumption_id,
                "text": assumption_text,
                "vec": text_to_vec[assumption_text],
            }
            for assumption_id, assumption_text in zip(row.turn_b_assumption_ids, row.turn_b_assumption_texts)
            if assumption_text in text_to_vec
        ]
        bridge = select_bridge_context(vec_a, vec_claim, assumption_records)
        baseline_texts = sample_unrelated_sentences(
            unrelated_sentences=unrelated_sentences,
            seed=seed,
            episode_id=str(row.episode_id),
            turn_a_idx=int(row.turn_a_idx),
            turn_b_idx=int(row.turn_b_idx),
        )
        baseline_vec = compose_normalized_mean([text_to_vec[text] for text in baseline_texts])

        sim_claim_values.append(bridge["sim_claim"])
        sim_context_values.append(bridge["sim_context"])
        sim_unrelated_sentences_only_values.append(cosine_similarity(vec_a, baseline_vec))
        bridge_deltas.append(bridge["sim_context"] - bridge["sim_claim"])

    df = df.copy()
    df["sim_claim"] = sim_claim_values
    df["sim_context"] = sim_context_values
    df["sim_unrelated_sentences_only"] = sim_unrelated_sentences_only_values
    df["bridge_delta"] = bridge_deltas
    return df


def add_representation_vectors(
    df: pd.DataFrame,
    embedding_model_name: str,
    batch_size: int,
    use_tqdm: bool,
    embedding_device: str,
):
    texts_to_embed = pd.concat(
        [
            df["turn_b_claim_text"].astype(str),
            df["turn_b_context_text"].astype(str),
        ],
        ignore_index=True,
    ).tolist()
    text_to_vec = embed_texts(
        texts=texts_to_embed,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
    )
    enriched = df.copy()
    enriched["vec_claim"] = enriched["turn_b_claim_text"].map(text_to_vec)
    enriched["vec_context"] = enriched["turn_b_context_text"].map(text_to_vec)
    return enriched


def pointplot_long_frame(df: pd.DataFrame):
    value_columns = [
        ("Claim Only", "sim_claim"),
        ("Assumption Context", "sim_context"),
        ("Baseline", "sim_unrelated_sentences_only"),
    ]
    frames = []
    for comparison, column_name in value_columns:
        part = df[["category", column_name]].rename(columns={column_name: "cosine_similarity"})
        part["comparison"] = comparison
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
        "Assumption Context",
        "Baseline",
    ]
    palette = {
        "Claim Only": "#d95f02",
        "Assumption Context": "#1b9e77",
        "Baseline": "#4c78a8",
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
        markers=["o", "s", "D"],
        ax=ax,
    )
    ax.set_title("Relevance Bridge With Full Assumption Context", fontsize=15)
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
    ax.hist(df["sim_claim"], bins=bins, density=True, alpha=0.42, color="#d95f02", label="Claim Only")
    ax.hist(df["sim_context"], bins=bins, density=True, alpha=0.48, color="#1b9e77", label="Assumption Context")
    ax.hist(
        df["sim_unrelated_sentences_only"],
        bins=bins,
        density=True,
        alpha=0.42,
        color="#4c78a8",
        label="Baseline",
    )
    ax.axvline(df["sim_claim"].mean(), color="#d95f02", linestyle="--", linewidth=1.8)
    ax.axvline(df["sim_context"].mean(), color="#1b9e77", linestyle="--", linewidth=1.8)
    ax.axvline(df["sim_unrelated_sentences_only"].mean(), color="#4c78a8", linestyle="--", linewidth=1.8)
    ax.set_title("Relevance Bridge Distribution", fontsize=15)
    ax.set_xlabel("Cosine Similarity With Previous Substantive Turn")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compute_trajectory_metrics(df: pd.DataFrame):
    claim_steps = []
    context_steps = []
    total_claim_path_length = 0.0
    total_context_path_length = 0.0
    episode_count = 0

    ordered = df.sort_values(["episode_id", "turn_b_idx"]).reset_index(drop=True)
    for _, episode_df in ordered.groupby("episode_id", sort=False, observed=False):
        if len(episode_df) < 2:
            continue
        episode_count += 1
        claim_vectors = episode_df["vec_claim"].tolist()
        context_vectors = episode_df["vec_context"].tolist()
        for idx in range(1, len(episode_df)):
            claim_step = 1.0 - cosine_similarity(claim_vectors[idx - 1], claim_vectors[idx])
            context_step = 1.0 - cosine_similarity(context_vectors[idx - 1], context_vectors[idx])
            claim_steps.append(claim_step)
            context_steps.append(context_step)
            total_claim_path_length += claim_step
            total_context_path_length += context_step

    return {
        "claim_only": {
            "mean_adjacent_step_distance": float(np.mean(claim_steps)) if claim_steps else 0.0,
            "total_path_length": float(total_claim_path_length),
            "episode_count": int(episode_count),
            "transition_count": int(len(claim_steps)),
        },
        "assumption_context": {
            "mean_adjacent_step_distance": float(np.mean(context_steps)) if context_steps else 0.0,
            "total_path_length": float(total_context_path_length),
            "episode_count": int(episode_count),
            "transition_count": int(len(context_steps)),
        },
    }


def select_umap_rows(df: pd.DataFrame):
    sampled_parts = []
    episode_sizes = df.groupby("episode_id", observed=False).size().sort_values(ascending=False)
    for episode_id in episode_sizes.head(UMAP_MAX_EPISODES).index.tolist():
        episode_df = df[df["episode_id"] == episode_id].sort_values("turn_b_idx")
        if len(episode_df) > UMAP_MAX_POINTS_PER_EPISODE:
            sample_indices = np.linspace(0, len(episode_df) - 1, UMAP_MAX_POINTS_PER_EPISODE, dtype=int)
            episode_df = episode_df.iloc[sample_indices]
        sampled_parts.append(episode_df)
    if not sampled_parts:
        return df.iloc[0:0].copy()
    return pd.concat(sampled_parts, ignore_index=True)


def write_umap_outputs(df: pd.DataFrame, sample_csv_path: Path, plot_path: Path, seed: int):
    import umap

    sampled_df = select_umap_rows(df)
    if sampled_df.empty:
        empty_frame = pd.DataFrame(
            columns=["episode_id", "category", "turn_b_idx", "representation", "umap_x", "umap_y"]
        )
        empty_frame.to_csv(sample_csv_path, index=False)
        return {
            "sample_csv": str(sample_csv_path),
            "plot_png": str(plot_path),
            "sampled_episode_count": 0,
            "sampled_row_count": 0,
        }

    claim_matrix = np.vstack(sampled_df["vec_claim"].to_list())
    context_matrix = np.vstack(sampled_df["vec_context"].to_list())
    stacked_matrix = np.vstack([claim_matrix, context_matrix])
    umap_model = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(UMAP_NEIGHBORS, max(2, len(sampled_df) - 1)),
        min_dist=UMAP_MIN_DIST,
        random_state=seed,
    )
    embedding = umap_model.fit_transform(stacked_matrix)
    split_index = len(sampled_df)
    claim_embedding = embedding[:split_index]
    context_embedding = embedding[split_index:]

    sample_rows = []
    for row_idx, row in enumerate(sampled_df.itertuples(index=False)):
        sample_rows.append(
            {
                "episode_id": row.episode_id,
                "category": row.category,
                "turn_b_idx": int(row.turn_b_idx),
                "representation": "Claim Only",
                "umap_x": float(claim_embedding[row_idx, 0]),
                "umap_y": float(claim_embedding[row_idx, 1]),
            }
        )
        sample_rows.append(
            {
                "episode_id": row.episode_id,
                "category": row.category,
                "turn_b_idx": int(row.turn_b_idx),
                "representation": "Assumption Context",
                "umap_x": float(context_embedding[row_idx, 0]),
                "umap_y": float(context_embedding[row_idx, 1]),
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(sample_csv_path, index=False)

    palette = sns.color_palette("tab20", sampled_df["episode_id"].nunique())
    episode_colors = {episode_id: palette[idx % len(palette)] for idx, episode_id in enumerate(sampled_df["episode_id"].unique())}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    for ax, representation in zip(axes, ["Claim Only", "Assumption Context"]):
        part = sample_df[sample_df["representation"] == representation]
        for episode_id, episode_df in part.groupby("episode_id", sort=False, observed=False):
            episode_df = episode_df.sort_values("turn_b_idx")
            color = episode_colors[episode_id]
            ax.plot(
                episode_df["umap_x"],
                episode_df["umap_y"],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                alpha=0.82,
                color=color,
            )
        ax.set_title(representation)
        ax.set_xlabel("UMAP-1")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("UMAP-2")
    fig.suptitle("Exp 1 Conversation Trajectory Smoothing", fontsize=15)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    return {
        "sample_csv": str(sample_csv_path),
        "plot_png": str(plot_path),
        "sampled_episode_count": int(sampled_df["episode_id"].nunique()),
        "sampled_row_count": int(len(sampled_df)),
    }


def build_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category", as_index=False)
        .agg(
            pair_count=("bridge_delta", "size"),
            mean_sim_claim=("sim_claim", "mean"),
            mean_sim_context=("sim_context", "mean"),
            mean_sim_unrelated_sentences_only=("sim_unrelated_sentences_only", "mean"),
            mean_bridge_delta=("bridge_delta", "mean"),
            positive_bridge_rate=("bridge_delta", lambda values: float((values > 0).mean())),
            average_assumption_count=("candidate_assumption_count", "mean"),
            assumption_pair_rate=("turn_b_has_assumptions", "mean"),
        )
        .sort_values("mean_bridge_delta", ascending=False)
    )


def write_pair_csv(df: pd.DataFrame, path: Path):
    df[PAIR_EXPORT_COLUMNS].to_csv(path, index=False)


def build_patch_summary(
    args,
    categories,
    selected_files,
    category_files,
    df: pd.DataFrame,
    pair_csv: Path,
    summary_json: Path,
):
    return {
        "experiment": "Experiment 1: The Relevance Bridge",
        "analysis_stage": "patch_pair_extraction_only",
        "input_dir": str(args.input_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "embedding_batch_size": int(args.embedding_batch_size),
        "embedding_device": str(resolve_embedding_device(args.embedding_device)),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "default_embedding_device": DEFAULT_EMBEDDING_DEVICE,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "num_patches": int(args.num_patches),
        "patch_index": int(args.patch_index),
        "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
        "selected_episode_file_count": int(len(selected_files)),
        "candidate_episode_file_count": int(len(category_files)),
        "baseline_sentence_pool_path": str(UNRELATED_SENTENCES_PATH),
        "baseline_sentence_pool_size": int(UNRELATED_SENTENCE_POOL_SIZE),
        "baseline_sentence_sample_size": int(BASELINE_SENTENCE_SAMPLE_SIZE),
        "categories": categories,
        "total_pairs": int(len(df)),
        "pairs_with_assumptions_on_turn_b": int(df["turn_b_has_assumptions"].sum()),
        "assumption_pair_rate": float(df["turn_b_has_assumptions"].mean()),
        "average_assumption_count": float(df["candidate_assumption_count"].mean()),
        "bridge_score_mean_delta": float(df["bridge_delta"].mean()),
        "outputs": {
            "pair_csv": str(pair_csv),
            "summary_json": str(summary_json),
        },
        "deferred_outputs": [
            "exp1_bridge_by_category.csv",
            "exp1_similarity_pointplot_summary.csv",
            "exp1_similarity_pointplot.png",
            "exp1_distance_distribution.png",
            "exp1_umap_sample.csv",
            "exp1_umap_trajectory.png",
        ],
        "notes": [
            "Patch mode writes adjacent-pair bridge diagnostics only.",
            "Turn B context is the full same-turn assumption bag appended to the claim text.",
            "Baseline samples 10 unrelated sentences from the fixed Exp 1 sentence pool without including the Turn B claim.",
            "Run merge_exp1_patches.py after all patches finish to build the merged summaries, plots, and UMAP outputs.",
        ],
    }


def build_full_summary(
    args,
    output_dir: Path,
    categories,
    selected_files,
    category_files,
    df: pd.DataFrame,
    pair_csv: Path,
    category_csv: Path,
    pointplot_csv: Path,
    pointplot_png: Path,
    dist_png: Path,
    umap_sample_csv: Path,
    umap_png: Path,
    summary_json: Path,
    trajectory_metrics,
    umap_outputs,
):
    boot = bootstrap_mean(df["bridge_delta"], seed=args.seed)
    return {
        "experiment": "Experiment 1: The Relevance Bridge",
        "analysis_stage": "full_analysis",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "embedding_batch_size": int(args.embedding_batch_size),
        "embedding_device": str(resolve_embedding_device(args.embedding_device)),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "default_embedding_device": DEFAULT_EMBEDDING_DEVICE,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "num_patches": int(args.num_patches),
        "patch_index": int(args.patch_index),
        "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
        "selected_episode_file_count": int(len(selected_files)),
        "candidate_episode_file_count": int(len(category_files)),
        "baseline_sentence_pool_path": str(UNRELATED_SENTENCES_PATH),
        "baseline_sentence_pool_size": int(UNRELATED_SENTENCE_POOL_SIZE),
        "baseline_sentence_sample_size": int(BASELINE_SENTENCE_SAMPLE_SIZE),
        "categories": categories,
        "total_pairs": int(len(df)),
        "pairs_with_assumptions_on_turn_b": int(df["turn_b_has_assumptions"].sum()),
        "assumption_pair_rate": float(df["turn_b_has_assumptions"].mean()),
        "average_assumption_count": float(df["candidate_assumption_count"].mean()),
        "bridge_score_mean_delta": float(df["bridge_delta"].mean()),
        "bridge_score_median_delta": float(df["bridge_delta"].median()),
        "positive_bridge_rate": float((df["bridge_delta"] > 0).mean()),
        "mean_similarity_claim_only": float(df["sim_claim"].mean()),
        "mean_similarity_assumption_context": float(df["sim_context"].mean()),
        "mean_similarity_with_assumptions": float(df["sim_context"].mean()),
        "mean_similarity_unrelated_sentences_only": float(df["sim_unrelated_sentences_only"].mean()),
        "mean_similarity_gain_percent": float(
            100.0 * (df["sim_context"].mean() - df["sim_claim"].mean()) / max(abs(df["sim_claim"].mean()), 1e-9)
        ),
        "bridge_delta_bootstrap": boot,
        "trajectory_smoothing": trajectory_metrics,
        "umap": umap_outputs,
        "outputs": {
            "pair_csv": str(pair_csv),
            "category_csv": str(category_csv),
            "pointplot_summary_csv": str(pointplot_csv),
            "pointplot_png": str(pointplot_png),
            "distance_distribution_png": str(dist_png),
            "umap_sample_csv": str(umap_sample_csv),
            "umap_trajectory_png": str(umap_png),
            "summary_json": str(summary_json),
        },
        "notes": [
            "Only turns with turn_type_label == 'Substantive' are included.",
            "Turn A is vectorized from explicit propositions when available, otherwise raw turn text.",
            "Turn B claim text comes from explicit_propositions.",
            "Turn B context uses the full same-turn assumption bag without any greedy filtering.",
            "Baseline samples 10 unrelated sentences from the fixed Exp 1 sentence pool without including the Turn B claim.",
            "Cosine similarity is computed on L2-normalized sentence embeddings from the selected embedding model.",
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
    unrelated_sentences = load_unrelated_sentences(UNRELATED_SENTENCES_PATH)

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
    texts_to_embed = pd.concat(
        [
            df["turn_a_text"],
            df["turn_b_claim_text"],
            pd.Series(
                [text for texts in df["turn_b_assumption_texts"] for text in texts],
                dtype=object,
            ),
            pd.Series(unrelated_sentences, dtype=object),
        ],
        ignore_index=True,
    ).tolist()
    text_to_vec = embed_texts(
        texts=texts_to_embed,
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
    )
    df = add_bridge_metrics(
        df=df,
        text_to_vec=text_to_vec,
        unrelated_sentences=unrelated_sentences,
        seed=args.seed,
        use_tqdm=use_tqdm,
    )
    logger.info("Collected %d adjacent substantive pairs across %d categories.", len(df), len(categories))

    pair_csv = output_dir / "exp1_bridge_pairs.csv"
    summary_json = output_dir / "exp1_summary.json"
    write_pair_csv(df, pair_csv)

    if args.num_patches > 1:
        patch_summary = build_patch_summary(
            args=args,
            categories=categories,
            selected_files=selected_files,
            category_files=category_files,
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
    umap_sample_csv = output_dir / "exp1_umap_sample.csv"
    umap_png = output_dir / "exp1_umap_trajectory.png"

    distribution_plot(df, dist_png)

    by_category = build_by_category(df)
    by_category.to_csv(category_csv, index=False)

    long_plot_df = pointplot_long_frame(df)
    pointplot_summary_df = pointplot_summary(long_plot_df, seed=args.seed)
    pointplot_summary_df.to_csv(pointplot_csv, index=False)
    bootstrap_pointplot(long_plot_df, pointplot_png, category_order=categories, seed=args.seed)

    trajectory_df = add_representation_vectors(
        df=df,
        embedding_model_name=args.embedding_model_name,
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
        embedding_device=args.embedding_device,
    )
    trajectory_metrics = compute_trajectory_metrics(trajectory_df)
    umap_outputs = write_umap_outputs(
        df=trajectory_df,
        sample_csv_path=umap_sample_csv,
        plot_path=umap_png,
        seed=args.seed,
    )

    summary = build_full_summary(
        args=args,
        output_dir=output_dir,
        categories=categories,
        selected_files=selected_files,
        category_files=category_files,
        df=df,
        pair_csv=pair_csv,
        category_csv=category_csv,
        pointplot_csv=pointplot_csv,
        pointplot_png=pointplot_png,
        dist_png=dist_png,
        umap_sample_csv=umap_sample_csv,
        umap_png=umap_png,
        summary_json=summary_json,
        trajectory_metrics=trajectory_metrics,
        umap_outputs=umap_outputs,
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    logger.info("Done. Wrote results to %s", output_dir)


if __name__ == "__main__":
    main()
