import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Seaborn style (no deprecated .set)
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)

# -----------------------------
# Config
# -----------------------------
BASE_RESULTS = Path("results")
PROMPT_PARENT_GLOB = "prompt_camprison"  # your folder name
PROMPT_PREFIX = "prompt"                # prompt1..prompt5
OUTDIR = BASE_RESULTS / "carry_over_timelines"

SIM_THRESH = 0.60
WINDOW = 10
MODEL_NAME = "all-MiniLM-L6-v2"

# y-bases for 2 speakers must be (0.3, 0.7)
YBASE_TWO_SPEAKERS = (0.3, 0.7)

# stack step: make dots of one speaker closer
STACK_STEP = 0.015  # smaller = tighter stacking around base y

# carry line style (thicker)
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
                to_compute,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            if embs.ndim == 1:
                embs = embs.unsqueeze(0)
            for t, e in zip(to_compute, embs):
                self.cache[t] = e.detach()
        return torch.stack([self.cache[t] for t in texts], dim=0)


# -----------------------------
# Small helpers (inline-only style)
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


# -----------------------------
# Load prompt dirs
# -----------------------------
def find_latest_prompt_parent() -> Path:
    candidates = [p for p in BASE_RESULTS.iterdir()
                  if p.is_dir() and p.name.startswith("prompt")]
    # if you have multiple prompt* dirs, pick most recent
    # otherwise fall back to prompt_camprison
    parent = BASE_RESULTS / PROMPT_PARENT_GLOB
    if parent.exists():
        return parent
    if not candidates:
        raise FileNotFoundError("No prompt* folder found under ./results")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_prompt_dirs(parent: Path) -> List[Path]:
    prompt_dirs = sorted(
        [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(PROMPT_PREFIX)],
        key=lambda p: p.name
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No {PROMPT_PREFIX}* dirs under {parent}")
    return prompt_dirs


# -----------------------------
# Carry-over computation
# -----------------------------
def compute_carry_edges(
    speaker_turns: List[Dict],
    embedder: Embedder,
    sim_thresh: float = SIM_THRESH,
    window: int = WINDOW,
) -> List[Tuple[int, str, int, str, float]]:
    """
    Returns list of edges:
        (src_turn_idx, src_text, tgt_turn_idx, tgt_text, similarity)
    where src is earlier assumption by other speaker, tgt is current assumption.
    """
    edges = []

    # Pre-collect assumptions per turn
    assm_texts_per_turn = []
    speakers_per_turn = []
    for t in speaker_turns:
        speakers_per_turn.append(t["speaker_id"])
        assm_texts_per_turn.append(texts_from_items(t.get("assumptions", [])))

    # Pre-embed all assumptions per turn
    assm_embeds_per_turn = [embedder.encode(txts) for txts in assm_texts_per_turn]

    for i in range(len(speaker_turns)):
        cur_spk = speakers_per_turn[i]
        cur_txts = assm_texts_per_turn[i]
        cur_emb = assm_embeds_per_turn[i]
        if len(cur_txts) == 0:
            continue

        # look back window
        start_j = max(0, i - window)
        for j in range(start_j, i):
            prev_spk = speakers_per_turn[j]
            if prev_spk == cur_spk:
                continue

            prev_txts = assm_texts_per_turn[j]
            prev_emb = assm_embeds_per_turn[j]
            if len(prev_txts) == 0:
                continue

            sims = util.cos_sim(prev_emb, cur_emb)  # [prev, cur]
            sims_np = sims.cpu().numpy()

            # For each current assumption, connect to best previous (other-speaker) if >= thresh
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
):
    # Unique speakers in episode (preserve order of appearance)
    speakers = []
    for t in speaker_turns:
        spk = t["speaker_id"] or "NO_SPEAKER"
        if spk not in speakers:
            speakers.append(spk)

    n_spk = len(speakers)

    # y-bases
    if n_spk == 2:
        y_bases = {speakers[0]: YBASE_TWO_SPEAKERS[0], speakers[1]: YBASE_TWO_SPEAKERS[1]}
    else:
        # fallback spread (still centered-ish)
        ys = np.linspace(0.2, 0.8, n_spk)
        y_bases = {spk: float(y) for spk, y in zip(speakers, ys)}

    # Collect points with stacking
    point_rows = []
    for turn_idx, t in enumerate(speaker_turns):
        spk = t["speaker_id"] or "NO_SPEAKER"
        assm_txts = texts_from_items(t.get("assumptions", []))
        m = len(assm_txts)
        if m == 0:
            continue

        base = y_bases[spk]

        # tighter stack centered on base
        offsets = [(k - (m - 1) / 2.0) * STACK_STEP for k in range(m)]
        for k, txt in enumerate(assm_txts):
            point_rows.append({
                "turn_num": turn_idx + 1,
                "speaker": spk,
                "y": base + offsets[k],
                "assumption_text": txt
            })

    df_pts = pd.DataFrame(point_rows)

    # Build plot
    plt.figure(figsize=(13, 4.5))
    ax = plt.gca()

    # Scatter (points)
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

    # Speaker horizontal guide lines + labels (placed at y-bases)
    for spk in speakers:
        y = y_bases[spk]
        ax.axhline(y, color="gray", lw=1, alpha=0.4)

    ax.set_yticks([y_bases[spk] for spk in speakers])
    ax.set_yticklabels(speakers)

    # Carry-over lines (thicker)
    # We draw line between *centers* of the matched assumptions (use their y positions if found).
    # Find y for a given (turn_idx, spk, assumption_text) by choosing closest point.
    # If missing, fall back to base.
    pts_by_turn = {}
    for _, r in df_pts.iterrows():
        pts_by_turn.setdefault(r["turn_num"], []).append(r)

    for (j, src_text, i, tgt_text, sim) in edges:
        src_turn = j + 1
        tgt_turn = i + 1
        src_spk = speaker_turns[j]["speaker_id"] or "NO_SPEAKER"
        tgt_spk = speaker_turns[i]["speaker_id"] or "NO_SPEAKER"

        # try to locate y by text match in that turn
        src_y = y_bases[src_spk]
        tgt_y = y_bases[tgt_spk]

        if src_turn in pts_by_turn:
            cand = [r for r in pts_by_turn[src_turn] if r["assumption_text"] == src_text and r["speaker"] == src_spk]
            if cand:
                src_y = float(cand[0]["y"])
        if tgt_turn in pts_by_turn:
            cand = [r for r in pts_by_turn[tgt_turn] if r["assumption_text"] == tgt_text and r["speaker"] == tgt_spk]
            if cand:
                tgt_y = float(cand[0]["y"])

        ax.plot(
            [src_turn, tgt_turn],
            [src_y, tgt_y],
            lw=CARRY_LW,
            alpha=CARRY_ALPHA,
        )

    # X ticks: avoid overlap WITHOUT rotation
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
        f"Assumptions over turns (carry-over lines: sim≥{SIM_THRESH}, window={WINDOW})"
    )

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{safe_name(prompt_name)}__{safe_name(episode_key)}.png", dpi=160)
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

        prompt_out = OUTDIR / prompt_name
        prompt_out.mkdir(parents=True, exist_ok=True)

        print(f"\n🔍 {prompt_name}: {len(files)} episode files")
        for fpath in tqdm(files, desc=prompt_name):
            try:
                data = json.load(open(fpath, "r", encoding="utf-8"))
            except Exception:
                continue
            turns = data if isinstance(data, list) else [data]
            if not turns:
                continue

            # normalize speaker_id missing
            speaker_turns = []
            for t in turns:
                speaker_turns.append({
                    "speaker_id": t.get("speaker_id", "NO_SPEAKER"),
                    "assumptions": t.get("assumptions", []),
                    "explicit_propositions": t.get("explicit_propositions", []),
                    "turn_text": t.get("turn_text", ""),
                })

            # compute edges and plot
            edges = compute_carry_edges(speaker_turns, embedder, SIM_THRESH, WINDOW)
            plot_episode_timeline(
                prompt_name=prompt_name,
                episode_key=fpath.stem,
                speaker_turns=speaker_turns,
                edges=edges,
                outdir=prompt_out
            )

    print(f"\n✅ Saved timelines to: {OUTDIR}")


if __name__ == "__main__":
    main()
