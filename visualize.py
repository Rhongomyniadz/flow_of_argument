import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, util

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)


# -----------------------------
# Embedding helpers (caching)
# -----------------------------
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, torch.Tensor] = {}

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of strings with caching. Returns a 2D tensor [n, d]."""
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty(0, dim)
        to_compute = [t for t in texts if t not in self.cache]
        if to_compute:
            embs = self.model.encode(
                to_compute,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            if embs.ndim == 1:
                embs = embs.unsqueeze(0)
            for t, e in zip(to_compute, embs):
                self.cache[t] = e.detach()
        stacked = torch.stack([self.cache[t] for t in texts], dim=0)
        return stacked


# -----------------------------
# Text / metric helpers
# -----------------------------
def texts_from_items(items: List[dict]) -> List[str]:
    out = []
    for x in items or []:
        if isinstance(x, dict) and "text" in x and isinstance(x["text"], str):
            s = x["text"].strip()
            if s:
                out.append(s)
    return out


def ttr(texts: List[str]) -> float:
    """Type-token ratio over whitespace tokens; 0 if empty."""
    if not texts:
        return 0.0
    toks = []
    for t in texts:
        toks.extend(t.lower().split())
    if not toks:
        return 0.0
    return float(len(set(toks))) / float(len(toks))


def mean_pairwise_cosine(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean of all pairwise cosine similarities between rows of A and rows of B."""
    if A.numel() == 0 or B.numel() == 0:
        return 0.0
    sim = util.cos_sim(A, B)  # [|A|, |B|]
    return float(sim.mean().item())


def mean_self_cosine(X: torch.Tensor) -> float:
    """Mean pairwise cosine similarity within a set (upper triangle, excl. diagonal)."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    S = util.cos_sim(X, X)  # includes diagonal 1s
    iu = torch.triu_indices(n, n, offset=1)
    vals = S[iu[0], iu[1]]
    return float(vals.mean().item()) if vals.numel() > 0 else 0.0


# -----------------------------
# Loading & tabulation
# -----------------------------
def load_prompt_dirs(base_dir: str = "results/prompt_camprison") -> List[Path]:
    base = Path(base_dir)
    prompt_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("prompt")],
        key=lambda p: p.name,
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt directories found under {base_dir}")
    return prompt_dirs


def load_turn_rows_for_prompt_dir(prompt_dir: Path) -> List[Dict]:
    """
    Returns per-turn rows with:
      prompt, episode_key, turn_num, exp_texts, imp_texts
    """
    rows: List[Dict] = []
    files = sorted(prompt_dir.glob("*.json"))
    print(f"🔍 Evaluating {prompt_dir.name} ({len(files)} files)...")

    for fpath in tqdm(files, desc=prompt_dir.name):
        try:
            data = json.load(open(fpath, "r", encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        # derive episode_key from filename (minus extension)
        episode_key = fpath.stem

        for idx, item in enumerate(items, start=1):
            exp = texts_from_items(item.get("explicit_propositions", []))
            # treating assumptions as "implicit"
            imp = texts_from_items(item.get("assumptions", []))
            rows.append({
                "prompt": prompt_dir.name,
                "episode_key": episode_key,
                "turn_num": idx,
                "exp_texts": exp,
                "imp_texts": imp,
            })
    return rows


def evaluate_prompt_dir(base_dir: str = "results/prompt_camprison") -> pd.DataFrame:
    """
    Builds a long dataframe with per-turn metrics and counts.
    """
    prompt_dirs = load_prompt_dirs(base_dir)
    embedder = Embedder("all-MiniLM-L6-v2")

    all_rows: List[Dict] = []
    # For random-baseline construction later
    global_imp_pool: List[str] = []

    for pdir in prompt_dirs:
        rows = load_turn_rows_for_prompt_dir(pdir)
        for r in rows:
            global_imp_pool.extend(r["imp_texts"])

        # Compute per-turn metrics
        for r in rows:
            exp_txts = r["exp_texts"]
            imp_txts = r["imp_texts"]

            exp_emb = embedder.encode_texts(exp_txts)
            imp_emb = embedder.encode_texts(imp_txts)

            # EP-IM similarity
            ep_im_sim = mean_pairwise_cosine(exp_emb, imp_emb)

            # Lexical diversity (separate)
            div_exp = ttr(exp_txts)
            div_imp = ttr(imp_txts)

            # Redundancy within implicit (assumptions)
            red_imp = mean_self_cosine(imp_emb)

            # Counts
            all_rows.append({
                "prompt": r["prompt"],
                "episode_key": r["episode_key"],
                "turn_num": r["turn_num"],
                "exp_count": len(exp_txts),
                "imp_count": len(imp_txts),
                "ep_im_similarity": ep_im_sim,
                "diversity_exp": div_exp,
                "diversity_imp": div_imp,
                "redundancy_imp": red_imp,
            })

    df = pd.DataFrame(all_rows)
    # Build random-baseline redundancy by shuffling assumptions across turns, preserving per-turn sizes.
    if not df.empty:
        baseline_vals = build_redundancy_baseline(df, global_imp_pool, embedder, n_runs=5, seed=42)
        # Attach overall baseline (mean of runs) for convenience
        df.attrs["redundancy_baseline_mean"] = float(np.mean(baseline_vals)) if baseline_vals else 0.0
        df.attrs["redundancy_baseline_std"] = float(np.std(baseline_vals)) if baseline_vals else 0.0

    return df


def build_redundancy_baseline(df: pd.DataFrame,
                              pool_texts: List[str],
                              embedder: Embedder,
                              n_runs: int = 5,
                              seed: int = 42) -> List[float]:
    """
    Random baseline for redundancy: shuffle all implicit (assumption) texts across turns,
    preserving per-turn implicit counts; compute mean within-turn redundancy across dataset.
    Repeat n_runs and return the list of run means.
    """
    pool = [t for t in pool_texts if isinstance(t, str) and t.strip()]
    if not pool:
        return []

    rng = random.Random(seed)
    run_means: List[float] = []

    # Build per-turn sizes in order
    turn_sizes: List[int] = df["imp_count"].tolist()

    for _ in range(n_runs):
        pool_copy = pool[:]  # fresh copy to shuffle
        rng.shuffle(pool_copy)
        pos = 0
        redundancies = []

        for k in turn_sizes:
            if k <= 1:
                redundancies.append(0.0)
                continue
            # if we run out (shouldn't), wrap around
            if pos + k > len(pool_copy):
                # recycle by reshuffling
                rng.shuffle(pool_copy)
                pos = 0
            sample = pool_copy[pos:pos + k]
            pos += k
            emb = embedder.encode_texts(sample)
            redundancies.append(mean_self_cosine(emb))

        run_means.append(float(np.mean(redundancies)) if redundancies else 0.0)

    return run_means


# -----------------------------
# Visualization
# -----------------------------
def safe_name(name: str) -> str:
    """Safe filename from prompt name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def visualize_metrics(df: pd.DataFrame, outdir: str = "results/analysis_charts"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("⚠️ No data to visualize.")
        return

    prompt_order = sorted(df["prompt"].unique())
    vis_order = ["explicit", "implicit"]

    # --- 1) Lexical diversity by visibility (explicit vs implicit) ---
    div_long = pd.concat([
        df[["prompt", "diversity_exp"]]
        .rename(columns={"diversity_exp": "diversity"})
        .assign(visibility="explicit"),
        df[["prompt", "diversity_imp"]]
        .rename(columns={"diversity_imp": "diversity"})
        .assign(visibility="implicit"),
    ], ignore_index=True)

    plt.figure(figsize=(9, 5))
    sns.pointplot(
        x="prompt",
        y="diversity",
        hue="visibility",
        data=div_long,
        order=prompt_order,
        hue_order=vis_order,
        dodge=0.3,
        errorbar="se",
    )
    plt.title("Lexical Diversity (TTR) by Prompt and Visibility")
    plt.tight_layout()
    plt.savefig(out / "lexical_diversity_pointplot.png")
    plt.close()

    # --- 2) EP–IM similarity per prompt ---
    plt.figure(figsize=(9, 5))
    sns.pointplot(
        x="prompt",
        y="ep_im_similarity",
        data=df,
        order=prompt_order,
        errorbar="se",
    )
    plt.title("Explicit ↔ Implicit (Assumptions) Similarity by Prompt")
    plt.ylabel("Mean Pairwise Cosine")
    plt.tight_layout()
    plt.savefig(out / "ep_im_similarity_pointplot.png")
    plt.close()

    # --- 3) Assumption redundancy per prompt with random baseline ---
    baseline_mean = df.attrs.get("redundancy_baseline_mean", 0.0)
    baseline_std = df.attrs.get("redundancy_baseline_std", 0.0)

    plt.figure(figsize=(9, 5))
    sns.pointplot(
        x="prompt",
        y="redundancy_imp",
        data=df,
        order=prompt_order,
        errorbar="se",
    )
    plt.axhline(baseline_mean, color="red", linestyle="--",
                label=f"Random baseline (mean={baseline_mean:.3f})")
    if baseline_std > 0:
        plt.fill_between(
            [-0.5, len(prompt_order) - 0.5],
            [baseline_mean - baseline_std] * 2,
            [baseline_mean + baseline_std] * 2,
            color="red", alpha=0.1, label="Baseline ±1σ"
        )
    plt.title("Assumption Redundancy (within-turn) by Prompt\n(with Random Shuffle Baseline)")
    plt.ylabel("Mean Pairwise Cosine (upper triangle)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "redundancy_with_baseline.png")
    plt.close()

    # --- 4) Average number of explicit & implicit per turn (per prompt) ---
    cnt_long = pd.concat([
        df[["prompt", "exp_count"]]
        .rename(columns={"exp_count": "num_statements"})
        .assign(visibility="explicit"),
        df[["prompt", "imp_count"]]
        .rename(columns={"imp_count": "num_statements"})
        .assign(visibility="implicit"),
    ], ignore_index=True)

    plt.figure(figsize=(9, 5))
    sns.pointplot(
        x="prompt",
        y="num_statements",
        hue="visibility",
        data=cnt_long,
        order=prompt_order,
        hue_order=vis_order,
        dodge=0.3,
        errorbar="se",
    )
    plt.title("Average Number of Explicit vs. Implicit per Turn (per Prompt)")
    plt.tight_layout()
    plt.savefig(out / "counts_per_prompt_pointplot.png")
    plt.close()

    # --- 5) Time series: average number across all prompts per turn index ---
    # First, build long-form with prompt included
    cnt_ts = pd.concat([
        df[["prompt", "turn_num", "exp_count"]]
        .rename(columns={"exp_count": "num_statements"})
        .assign(visibility="explicit"),
        df[["prompt", "turn_num", "imp_count"]]
        .rename(columns={"imp_count": "num_statements"})
        .assign(visibility="implicit"),
    ], ignore_index=True)

    # Overall aggregated time series across prompts
    plt.figure(figsize=(10, 5))
    ax = sns.pointplot(
        x="turn_num",
        y="num_statements",
        hue="visibility",
        data=cnt_ts,
        hue_order=vis_order,
        errorbar="se",
    )
    plt.title("Average Number of Explicit vs. Implicit per Turn Index (All Prompts)")
    plt.xlabel("Turn Index within Episode")

    # >>> thin x-axis labels so they don't overlap
    uniq_turns = sorted(cnt_ts["turn_num"].unique())
    max_labels = 20
    if len(uniq_turns) > max_labels:
        step = math.ceil(len(uniq_turns) / max_labels)
        ticks = uniq_turns[::step]
    else:
        ticks = uniq_turns
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

    plt.tight_layout()
    plt.savefig(out / "counts_time_series_pointplot.png")
    plt.close()

    # --- 6) Time series per prompt: one plot per prompt in a subfolder ---
    ts_dir = out / "time_series_by_prompt"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for p in prompt_order:
        sub_df = cnt_ts[cnt_ts["prompt"] == p].copy()
        if sub_df.empty:
            continue

        plt.figure(figsize=(10, 5))
        ax = sns.pointplot(
            x="turn_num",
            y="num_statements",
            hue="visibility",
            data=sub_df,
            hue_order=vis_order,
            errorbar="se",
        )
        plt.title(f"Average Explicit vs. Implicit per Turn Index\n({p})")
        plt.xlabel("Turn Index within Episode")

        # >>> thin labels for this prompt only
        uniq_turns_p = sorted(sub_df["turn_num"].unique())
        max_labels_p = 20
        if len(uniq_turns_p) > max_labels_p:
            step_p = math.ceil(len(uniq_turns_p) / max_labels_p)
            ticks_p = uniq_turns_p[::step_p]
        else:
            ticks_p = uniq_turns_p
        ax.set_xticks(ticks_p)
        ax.set_xticklabels([str(t) for t in ticks_p])

        plt.tight_layout()
        fname = ts_dir / f"{safe_name(p)}_counts_time_series_pointplot.png"
        plt.savefig(fname)
        plt.close()

    # --- Save data tables for further analysis ---
    df.to_csv(out / "turn_level_metrics.csv", index=False)
    div_long.to_csv(out / "diversity_long.csv", index=False)
    cnt_long.to_csv(out / "counts_per_prompt_long.csv", index=False)
    cnt_ts.to_csv(out / "counts_time_series_long.csv", index=False)
    print(f"📊 Charts & tables saved to {out}/")
    print(f"📊 Per-prompt time series saved to {ts_dir}/")


# -----------------------------
# Main
# -----------------------------
def main():
    # Pick the latest prompt directory under results (e.g., results/prompt_camprison)
    candidates = [p for p in Path("results").iterdir()
                  if p.is_dir() and p.name.startswith("prompt")]
    if not candidates:
        raise FileNotFoundError("No prompt* directory found under ./results")
    latest_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"📂 Using latest prompt directory: {latest_dir}")

    df = evaluate_prompt_dir(str(latest_dir))
    print(f"✅ Parsed {len(df)} turn rows.")

    visualize_metrics(df)


if __name__ == "__main__":
    main()
