#!/usr/bin/env python3
"""
cross_speaker_match_and_plot.py

One-file pipeline:
  1) Load parsed prompt outputs under results/prompt_camprison/prompt*/Episode*.json
  2) Match explicit propositions of speaker A to assumptions of speaker B (and vice versa)
  3) Save CSVs under results/cross_speaker_matching/
  4) Plot seaborn dashboards under results/cross_speaker_matching/plots/

Expected per-turn structure (parsed outputs):
{
  "turn_text": "...",
  "speaker_id": "SPEAKER_01",
  ...
  "explicit_propositions": [{"text": "...", "confidence": 0.9}, ...],
  "assumptions": [{"text": "...", "confidence": 0.8}, ...]
}

Notes:
- We treat "assumptions" as the implicit side.
- Matching is within +/- window turns in the same episode.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, util


# ============================================================
# Loading helpers
# ============================================================
def load_prompt_dirs(base_dir: str) -> List[Path]:
    base = Path(base_dir)
    prompt_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("prompt")],
        key=lambda p: p.name,
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt directories under {base_dir}")
    return prompt_dirs


def texts_from_items(items: Any) -> List[str]:
    """Extract list of text strings from [{text, confidence}, ...]"""
    out = []
    if not isinstance(items, list):
        return out
    for x in items:
        if isinstance(x, dict) and isinstance(x.get("text"), str):
            s = x["text"].strip()
            if s:
                out.append(s)
        elif isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
    return out


def load_episode_items(fpath: Path) -> List[Dict[str, Any]]:
    try:
        data = json.load(open(fpath, "r", encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


# ============================================================
# Embedding with caching
# ============================================================
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, torch.Tensor] = {}
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.dim)

        to_compute = [t for t in texts if t not in self.cache]
        if to_compute:
            embs = self.model.encode(
                to_compute, convert_to_tensor=True, show_progress_bar=False
            )
            if embs.ndim == 1:
                embs = embs.unsqueeze(0)
            for t, e in zip(to_compute, embs):
                self.cache[t] = e.detach()

        return torch.stack([self.cache[t] for t in texts], dim=0)


# ============================================================
# Matching logic
# ============================================================
def collect_statements_for_episode(items: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
      explicits:  [{turn_num, speaker_id, text}, ...]
      assumptions:[{turn_num, speaker_id, text}, ...]
    """
    explicits = []
    assumptions = []
    for idx, item in enumerate(items, start=1):
        spk = item.get("speaker_id")

        exp_txts = texts_from_items(item.get("explicit_propositions", []))
        asm_txts = texts_from_items(item.get("assumptions", []))

        for t in exp_txts:
            explicits.append({"turn_num": idx, "speaker_id": spk, "text": t})

        for t in asm_txts:
            assumptions.append({"turn_num": idx, "speaker_id": spk, "text": t})

    return explicits, assumptions


def match_episode(explicits: List[Dict], assumptions: List[Dict],
                  embedder: Embedder, window: int, topk: int) -> List[Dict]:
    """
    For each explicit proposition, find top-k most similar assumptions
    from the OTHER speaker within +/- window turns.
    """
    if not explicits or not assumptions:
        return []

    # Pre-embed all assumption texts once
    asm_texts = [a["text"] for a in assumptions]
    asm_embs = embedder.encode(asm_texts)

    rows = []
    for e in explicits:
        e_spk = e["speaker_id"]
        e_turn = e["turn_num"]
        e_text = e["text"]

        # Candidate assumptions: other speaker + within window
        cand_idx = [
            j for j, a in enumerate(assumptions)
            if a["speaker_id"] is not None
            and e_spk is not None
            and a["speaker_id"] != e_spk
            and abs(a["turn_num"] - e_turn) <= window
        ]
        if not cand_idx:
            continue

        cand_embs = asm_embs[cand_idx]
        e_emb = embedder.encode([e_text])  # [1, d]

        sims = util.cos_sim(e_emb, cand_embs).squeeze(0)  # [num_cands]
        # topk indices within cand_idx
        k = min(topk, sims.numel())
        top_vals, top_pos = torch.topk(sims, k=k, largest=True)

        for v, pos in zip(top_vals.tolist(), top_pos.tolist()):
            a = assumptions[cand_idx[pos]]
            rows.append({
                "explicit_turn_num": e_turn,
                "explicit_speaker_id": e_spk,
                "explicit_text": e_text,
                "assumption_turn_num": a["turn_num"],
                "assumption_speaker_id": a["speaker_id"],
                "assumption_text": a["text"],
                "similarity": float(v),
                "window": window,
            })

    return rows


def run_matching(base_dir: str, out_root: str, window: int, topk: int,
                 emb_model: str) -> Path:
    """
    Runs matching for all prompts and episodes.
    Saves:
      - per-prompt CSV
      - combined CSV
    Returns path to combined CSV.
    """
    prompt_dirs = load_prompt_dirs(base_dir)
    embedder = Embedder(emb_model)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for pdir in prompt_dirs:
        prompt_name = pdir.name
        files = sorted(pdir.glob("*.json"))
        print(f"🔍 Matching {prompt_name} ({len(files)} files)...")

        prompt_rows = []
        for fpath in tqdm(files, desc=prompt_name):
            episode_key = fpath.stem
            items = load_episode_items(fpath)
            if not items:
                continue

            explicits, assumptions = collect_statements_for_episode(items)
            rows = match_episode(explicits, assumptions, embedder, window=window, topk=topk)
            for r in rows:
                r["episode_key"] = episode_key
                r["prompt"] = prompt_name
            prompt_rows.extend(rows)

        if prompt_rows:
            df_p = pd.DataFrame(prompt_rows)
            df_p.to_csv(out_root / f"{prompt_name}_matches.csv", index=False)
            prompt_rows = df_p.to_dict(orient="records")

        all_rows.extend(prompt_rows)

    df_all = pd.DataFrame(all_rows)
    combined_csv = out_root / "cross_speaker_matches.csv"
    df_all.to_csv(combined_csv, index=False)
    print(f"✅ Saved combined matches to {combined_csv}")
    return combined_csv


# ============================================================
# Plotting
# ============================================================
def thin_xticklabels(ax, max_labels: int = 25):
    """
    Reduce x tick label density to avoid overlap, without rotating.
    Keeps at most max_labels labels.
    """
    labels = ax.get_xticklabels()
    n = len(labels)
    if n <= max_labels:
        return
    step = int(np.ceil(n / max_labels))
    new_labels = []
    for i, lab in enumerate(labels):
        if i % step == 0:
            new_labels.append(lab.get_text())
        else:
            new_labels.append("")
    ax.set_xticklabels(new_labels)


def plot_all(csv_path: str, out_dir: str, top_k: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No matches to plot.")
        return

    df["turn_distance"] = (df["assumption_turn_num"] - df["explicit_turn_num"]).abs()
    df["speaker_pair"] = df["explicit_speaker_id"].astype(str) + " → " + df["assumption_speaker_id"].astype(str)
    prompt_order = sorted(df["prompt"].unique())

    # 1) Similarity distribution per prompt
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        data=df, x="prompt", y="similarity",
        order=prompt_order, inner=None, cut=0
    )
    sns.stripplot(
        data=df, x="prompt", y="similarity",
        order=prompt_order, color="k", alpha=0.25, size=2, jitter=0.25
    )
    plt.title("Cross-speaker similarity distribution per prompt")
    ax = plt.gca()
    thin_xticklabels(ax, max_labels=10)
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_violin_per_prompt.png", dpi=200)
    plt.close()

    # 2) Mean similarity per prompt
    plt.figure(figsize=(9, 5))
    sns.pointplot(
        data=df, x="prompt", y="similarity",
        order=prompt_order, errorbar="se"
    )
    plt.title("Mean cross-speaker similarity per prompt")
    plt.ylabel("Mean cosine similarity")
    ax = plt.gca()
    thin_xticklabels(ax, max_labels=10)
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_mean_per_prompt.png", dpi=200)
    plt.close()

    # 3) Heatmap overall speaker->speaker mean similarity
    heat_overall = (
        df.groupby(["explicit_speaker_id","assumption_speaker_id"])["similarity"]
          .mean()
          .reset_index()
          .pivot(index="explicit_speaker_id", columns="assumption_speaker_id", values="similarity")
          .fillna(0.0)
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(heat_overall, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mean similarity: explicit speaker → assumption speaker (overall)")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_speaker_pair_overall.png", dpi=220)
    plt.close()

    # Per-prompt heatmaps
    heat_dir = out_dir / "heatmaps_by_prompt"
    heat_dir.mkdir(exist_ok=True)
    for p in prompt_order:
        sub = df[df["prompt"] == p]
        if sub.empty:
            continue
        heat_p = (
            sub.groupby(["explicit_speaker_id","assumption_speaker_id"])["similarity"]
               .mean()
               .reset_index()
               .pivot(index="explicit_speaker_id", columns="assumption_speaker_id", values="similarity")
               .fillna(0.0)
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(heat_p, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Mean similarity by speaker pair ({p})")
        plt.tight_layout()
        plt.savefig(heat_dir / f"{p}_heatmap_speaker_pair.png", dpi=220)
        plt.close()

    # 4) Similarity vs. turn distance (binned)
    max_d = int(df["turn_distance"].max())
    bins = list(range(0, max_d + 2, 1))
    df["dist_bin"] = pd.cut(df["turn_distance"], bins=bins, right=False, labels=bins[:-1])

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df, x="dist_bin", y="similarity",
        hue="prompt", hue_order=prompt_order,
        errorbar="se", marker="o"
    )
    plt.title("Similarity vs. turn distance (binned)")
    plt.xlabel("Turn distance |assumption_turn - explicit_turn|")
    plt.ylabel("Mean cosine similarity")
    plt.legend(title="prompt", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax = plt.gca()
    thin_xticklabels(ax, max_labels=25)
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_vs_turn_distance.png", dpi=220)
    plt.close()

    # 5) Top-K matches table
    topk = df.sort_values("similarity", ascending=False).head(top_k)
    topk.to_csv(out_dir / f"top_{top_k}_matches.csv", index=False)

    # 6) Mean similarity by speaker pair overall
    pair_mean = (
        df.groupby("speaker_pair")["similarity"].mean()
          .sort_values(ascending=False)
          .reset_index()
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=pair_mean, x="speaker_pair", y="similarity")
    plt.title("Mean similarity by speaker pair (overall)")
    plt.xlabel("")
    plt.ylabel("Mean cosine similarity")
    ax = plt.gca()
    thin_xticklabels(ax, max_labels=8)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_speaker_pair_overall.png", dpi=220)
    plt.close()

    print(f"📊 Plots saved to {out_dir}")
    print(f"📊 Per-prompt heatmaps saved to {heat_dir}")
    print(f"📊 Top-{top_k} matches saved to {out_dir / f'top_{top_k}_matches.csv'}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="results/prompt_camprison",
                    help="Directory containing prompt1..prompt5 folders")
    ap.add_argument("--out_root", type=str, default="results/cross_speaker_matching",
                    help="Where to write CSVs and plots")
    ap.add_argument("--window", type=int, default=10,
                    help="+/- turn window for cross-speaker matching")
    ap.add_argument("--topk", type=int, default=3,
                    help="Top-k assumptions per explicit proposition")
    ap.add_argument("--emb_model", type=str, default="all-MiniLM-L6-v2",
                    help="SentenceTransformer model name")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip seaborn plotting")
    ap.add_argument("--plot_top_k", type=int, default=50,
                    help="Top-K rows to export for strongest matches")
    args = ap.parse_args()

    combined_csv = run_matching(
        base_dir=args.base_dir,
        out_root=args.out_root,
        window=args.window,
        topk=args.topk,
        emb_model=args.emb_model,
    )

    if not args.no_plots:
        plot_all(
            csv_path=str(combined_csv),
            out_dir=str(Path(args.out_root) / "plots"),
            top_k=args.plot_top_k,
        )


if __name__ == "__main__":
    main()
