#!/usr/bin/env python3
"""
carry_over_timelines_directed_randomspeakers_keepnames.py

For each prompt* dir and each episode JSON:
- Randomly (but reproducibly per episode) map first 2 speakers to roles A/B
- Plot A explicit -> B implicit(assumptions)
- Plot B explicit -> A implicit(assumptions)

IMPORTANT: plots keep the ORIGINAL speaker IDs/names.
Saves into two subfolders per prompt.
"""

import json
import re
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)

# -----------------------------
# Config
# -----------------------------
BASE_RESULTS = Path("results")
PROMPT_PARENT_GLOB = "prompt_camprison"  # your folder name
PROMPT_PREFIX = "prompt"                # prompt1..prompt5
OUTDIR = BASE_RESULTS / "carry_over_timelines"

SIM_THRESH = 0.50
WINDOW = 10
MODEL_NAME = "all-MiniLM-L6-v2"

# y-bases for 2 speakers
YBASE_TWO_SPEAKERS = (0.3, 0.7)

# stacking for multiple statements
STACK_STEP = 0.015

# carry line style (bold)
CARRY_LW = 2.6
CARRY_ALPHA = 0.85


# -----------------------------
# Embedding cache
# -----------------------------
class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, torch.Tensor] = {}

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty(0, dim)
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


# -----------------------------
# Helpers
# -----------------------------
def texts_from_items(items: List[dict]) -> List[str]:
    out = []
    for x in items or []:
        if isinstance(x, dict) and isinstance(x.get("text"), str):
            s = x["text"].strip()
            if s:
                out.append(s)
    return out


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def find_latest_prompt_parent() -> Path:
    candidates = [p for p in BASE_RESULTS.iterdir()
                  if p.is_dir() and p.name.startswith("prompt")]
    parent = BASE_RESULTS / PROMPT_PARENT_GLOB
    if parent.exists():
        return parent
    if not candidates:
        raise FileNotFoundError("No prompt* folder found under ./results")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_prompt_dirs(parent: Path) -> List[Path]:
    prompt_dirs = sorted(
        [p for p in parent.iterdir()
         if p.is_dir() and p.name.startswith(PROMPT_PREFIX)],
        key=lambda p: p.name
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No {PROMPT_PREFIX}* dirs under {parent}")
    return prompt_dirs


# -----------------------------
# Carry-over computation (DIRECTED)
# -----------------------------
def compute_carry_edges(
    speaker_turns: List[Dict],
    embedder: Embedder,
    src_field: str,
    tgt_field: str,
    src_speaker: str,
    tgt_speaker: str,
    sim_thresh: float = SIM_THRESH,
    window: int = WINDOW,
) -> List[Tuple[int, str, int, str, float]]:
    """
    Directed edges from src_speaker/src_field (earlier turn)
    to tgt_speaker/tgt_field (later turn).
    """
    edges = []

    speakers_per_turn = []
    src_texts_per_turn = []
    tgt_texts_per_turn = []

    for t in speaker_turns:
        speakers_per_turn.append(t["speaker_id"])
        src_texts_per_turn.append(texts_from_items(t.get(src_field, [])))
        tgt_texts_per_turn.append(texts_from_items(t.get(tgt_field, [])))

    src_embeds_per_turn = [embedder.encode(txts) for txts in src_texts_per_turn]
    tgt_embeds_per_turn = [embedder.encode(txts) for txts in tgt_texts_per_turn]

    for i in range(len(speaker_turns)):
        if speakers_per_turn[i] != tgt_speaker:
            continue

        cur_txts = tgt_texts_per_turn[i]
        cur_emb = tgt_embeds_per_turn[i]
        if len(cur_txts) == 0:
            continue

        start_j = max(0, i - window)
        for j in range(start_j, i):
            if speakers_per_turn[j] != src_speaker:
                continue

            prev_txts = src_texts_per_turn[j]
            prev_emb = src_embeds_per_turn[j]
            if len(prev_txts) == 0:
                continue

            sims = util.cos_sim(prev_emb, cur_emb)
            sims_np = sims.cpu().numpy()

            for k_cur, cur_text in enumerate(cur_txts):
                best_prev_idx = int(np.argmax(sims_np[:, k_cur]))
                best_sim = float(sims_np[best_prev_idx, k_cur])
                if best_sim >= sim_thresh:
                    edges.append((j, prev_txts[best_prev_idx], i, cur_text, best_sim))

    return edges


# -----------------------------
# Plot one episode
# -----------------------------
def plot_episode_timeline(
    prompt_name: str,
    episode_key: str,
    speaker_turns: List[Dict],
    edges: List[Tuple[int, str, int, str, float]],
    outdir: Path,
    src_field: str,
    tgt_field: str,
    src_speaker: str,
    tgt_speaker: str,
    title_suffix: str,
    save_suffix: str,
):
    speakers = [src_speaker, tgt_speaker]
    y_bases = {
        src_speaker: YBASE_TWO_SPEAKERS[0],
        tgt_speaker: YBASE_TWO_SPEAKERS[1],
    }

    # Collect points with stacking
    point_rows = []
    for turn_idx, t in enumerate(speaker_turns):
        spk = t["speaker_id"] or "NO_SPEAKER"
        if spk not in speakers:
            continue

        texts = texts_from_items(t.get(src_field if spk == src_speaker else tgt_field, []))
        m = len(texts)
        if m == 0:
            continue

        base = y_bases[spk]
        offsets = [(k - (m - 1) / 2.0) * STACK_STEP for k in range(m)]
        for k, txt in enumerate(texts):
            point_rows.append({
                "turn_num": turn_idx + 1,
                "speaker": spk,
                "y": base + offsets[k],
                "text": txt
            })

    df_pts = pd.DataFrame(point_rows)

    plt.figure(figsize=(13, 4.5))
    ax = plt.gca()

    if not df_pts.empty:
        sns.scatterplot(
            data=df_pts,
            x="turn_num",
            y="y",
            hue="speaker",
            palette="deep",
            s=30,
            linewidth=0,
            ax=ax,
            legend=False,
        )

    # Speaker guide lines and y labels (KEEP ORIGINAL SPEAKER NAMES)
    for spk in speakers:
        y = y_bases[spk]
        ax.axhline(y, color="gray", lw=1, alpha=0.4)

    ax.set_yticks([y_bases[spk] for spk in speakers])
    ax.set_yticklabels(speakers)

    pts_by_turn = {}
    for _, r in df_pts.iterrows():
        pts_by_turn.setdefault(r["turn_num"], []).append(r)

    # carry-over lines
    for (j, src_text, i, tgt_text, sim) in edges:
        src_turn = j + 1
        tgt_turn = i + 1

        src_y = y_bases[src_speaker]
        tgt_y = y_bases[tgt_speaker]

        if src_turn in pts_by_turn:
            cand = [r for r in pts_by_turn[src_turn]
                    if r["text"] == src_text and r["speaker"] == src_speaker]
            if cand:
                src_y = float(cand[0]["y"])

        if tgt_turn in pts_by_turn:
            cand = [r for r in pts_by_turn[tgt_turn]
                    if r["text"] == tgt_text and r["speaker"] == tgt_speaker]
            if cand:
                tgt_y = float(cand[0]["y"])

        ax.plot([src_turn, tgt_turn], [src_y, tgt_y],
                lw=CARRY_LW, alpha=CARRY_ALPHA)

    # X ticks (no rotation)
    max_turn = len(speaker_turns)
    if max_turn <= 25:
        step = 1
    elif max_turn <= 60:
        step = 2
    else:
        step = 5
    xticks = list(range(1, max_turn + 1, step))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_turn + 0.5)

    ax.set_xlabel("Turn number")
    ax.set_ylabel("Speaker")
    ax.set_title(
        f"{prompt_name} — {episode_key}\n"
        f"{title_suffix} (carry-over lines: sim≥{SIM_THRESH}, window={WINDOW})"
    )

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{safe_name(prompt_name)}__{safe_name(episode_key)}__{save_suffix}.png"
    plt.savefig(outdir / fname, dpi=160)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parent = find_latest_prompt_parent()
    prompt_dirs = load_prompt_dirs(parent)
    print(f"📂 Using prompt parent: {parent}")

    embedder = Embedder(MODEL_NAME)

    for pdir in prompt_dirs:
        prompt_name = pdir.name
        files = sorted(pdir.glob("*.json"))
        if not files:
            continue

        out_A_to_B = OUTDIR / prompt_name / "A_exp_to_B_implicit"
        out_B_to_A = OUTDIR / prompt_name / "B_exp_to_A_implicit"
        out_A_to_B.mkdir(parents=True, exist_ok=True)
        out_B_to_A.mkdir(parents=True, exist_ok=True)

        print(f"\n🔍 {prompt_name}: {len(files)} episode files")
        for fpath in tqdm(files, desc=prompt_name):
            try:
                data = json.load(open(fpath, "r", encoding="utf-8"))
            except Exception:
                continue

            turns = data if isinstance(data, list) else [data]
            if not turns:
                continue

            speaker_turns = []
            ordered_speakers = []
            seen = set()

            for t in turns:
                spk = t.get("speaker_id", "NO_SPEAKER") or "NO_SPEAKER"
                speaker_turns.append({
                    "speaker_id": spk,
                    "assumptions": t.get("assumptions", []),
                    "explicit_propositions": t.get("explicit_propositions", []),
                    "turn_text": t.get("turn_text", ""),
                })
                if spk != "NO_SPEAKER" and spk not in seen:
                    seen.add(spk)
                    ordered_speakers.append(spk)

            episode_key = fpath.stem
            if len(ordered_speakers) < 2:
                continue

            # Random-but-stable pick of two speakers per episode
            two = ordered_speakers[:2]
            seed_int = int(hashlib.sha1(episode_key.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(seed_int)
            rng.shuffle(two)
            A, B = two[0], two[1]

            # A explicit -> B implicit(assumptions)
            edges_A_B = compute_carry_edges(
                speaker_turns=speaker_turns,
                embedder=embedder,
                src_field="explicit_propositions",
                tgt_field="assumptions",
                src_speaker=A,
                tgt_speaker=B,
                sim_thresh=SIM_THRESH,
                window=WINDOW,
            )
            plot_episode_timeline(
                prompt_name=prompt_name,
                episode_key=episode_key,
                speaker_turns=speaker_turns,
                edges=edges_A_B,
                outdir=out_A_to_B,
                src_field="explicit_propositions",
                tgt_field="assumptions",
                src_speaker=A,
                tgt_speaker=B,
                title_suffix=f"{A} explicit → {B} implicit(assumptions)",
                save_suffix="Aexp_to_Bimpl",
            )

            # B explicit -> A implicit(assumptions)
            edges_B_A = compute_carry_edges(
                speaker_turns=speaker_turns,
                embedder=embedder,
                src_field="explicit_propositions",
                tgt_field="assumptions",
                src_speaker=B,
                tgt_speaker=A,
                sim_thresh=SIM_THRESH,
                window=WINDOW,
            )
            plot_episode_timeline(
                prompt_name=prompt_name,
                episode_key=episode_key,
                speaker_turns=speaker_turns,
                edges=edges_B_A,
                outdir=out_B_to_A,
                src_field="explicit_propositions",
                tgt_field="assumptions",
                src_speaker=B,
                tgt_speaker=A,
                title_suffix=f"{B} explicit → {A} implicit(assumptions)",
                save_suffix="Bexp_to_Aimpl",
            )

    print(f"\n✅ Saved timelines to: {OUTDIR}")


if __name__ == "__main__":
    main()
