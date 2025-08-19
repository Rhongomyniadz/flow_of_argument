import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Optional SBERT (falls back to TF-IDF)
_HAS_SBERT = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _HAS_SBERT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-level-visualizer")


# --------------------------- Utilities ---------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    return sk_normalize(X, norm="l2", copy=False)

def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------- Data Loading ---------------------------

def load_turns(input_dir: Path) -> pd.DataFrame:
    """
    Load all JSON files from input_dir. Each file is an episode.
    Returns rows with: episode_file, turn_idx, speaker_id, speaker_key, inferred_speaker_name,
                       assumptions (list[str]), doc_text (joined assumptions)
    """
    rows = []
    for fp in sorted(input_dir.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                log.warning(f"Skipping {fp.name}: not a list")
                continue
            for i, rec in enumerate(data):
                assumptions = rec.get("assumptions") or []
                assumptions = [a for a in assumptions if isinstance(a, str) and a.strip()]
                if not assumptions:
                    continue
                doc_text = " ; ".join(assumptions)

                # Normalize speaker_id
                sid_raw = rec.get("speaker_id", None)
                sid_norm: Optional[str] = None
                if isinstance(sid_raw, list):
                    sid_norm = "-".join(str(x) for x in sid_raw) if sid_raw else None
                elif isinstance(sid_raw, dict):
                    sid_norm = sid_raw.get("id") or sid_raw.get("speaker_id")
                    sid_norm = str(sid_norm) if sid_norm is not None else None
                elif sid_raw is not None:
                    sid_norm = str(sid_raw)

                name = (rec.get("inferred_speaker_name") or "").strip()
                # Prefer id for grouping; fallback to name; final fallback
                speaker_key = sid_norm or (name if name and not name.startswith("NO_INFERRED") else None) or "UNKNOWN_SPEAKER"

                rows.append({
                    "episode_file": fp.name,
                    "turn_idx": i,
                    "speaker_id": sid_norm if sid_norm is not None else "",
                    "speaker_key": speaker_key,
                    "inferred_speaker_name": name if name else "NO_INFERRED_SPEAKER",
                    "assumptions": assumptions,
                    "doc_text": doc_text,
                })
        except Exception as e:
            log.warning(f"Failed to read {fp.name}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No usable turns found (empty assumptions?).")
    else:
        log.info(f"Loaded {len(df)} turns from {df['episode_file'].nunique()} episodes")
    return df


def build_assumption_table(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Expand per-turn assumptions into per-assumption rows.
    Returns columns:
      episode_file, turn_idx, speaker_id (or ''), speaker_key, inferred_speaker_name, assumption_text
    """
    rows = []
    for _, r in turns.iterrows():
        for j, a in enumerate(r["assumptions"]):
            rows.append({
                "episode_file": r["episode_file"],
                "turn_idx": int(r["turn_idx"]),
                "speaker_id": r["speaker_id"],
                "speaker_key": r["speaker_key"],
                "inferred_speaker_name": r["inferred_speaker_name"],
                "assumption_text": a.strip(),
                "assumption_idx": j,  # index within the turn
            })
    A = pd.DataFrame(rows)
    if A.empty:
        log.warning("Assumption table is empty.")
    else:
        log.info(f"Assumption table has {len(A)} rows (individual assumptions)")
    return A


# --------------------------- Embeddings ---------------------------

class Embedder:
    def __init__(self, backend: str, sbert_model: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        backend = backend.lower()
        if backend not in {"sbert", "tfidf"}:
            raise ValueError("backend must be 'sbert' or 'tfidf'")
        if backend == "sbert" and not _HAS_SBERT:
            log.warning("sentence-transformers not available; falling back to TF-IDF")
            backend = "tfidf"

        self.backend = backend
        self.model: Optional["SentenceTransformer"] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.sbert_model_name = sbert_model
        self.device = device

    def fit(self, texts: List[str]) -> None:
        if self.backend == "sbert":
            self.model = SentenceTransformer(self.sbert_model_name)
            if self.device:
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    pass
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
            self.vectorizer.fit(texts)

    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        if self.backend == "sbert":
            assert self.model is not None, "Call fit() first"
            X = self.model.encode(
                texts, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=len(texts) > 1024
            )
            return np.asarray(X, dtype=np.float32)
        else:
            assert self.vectorizer is not None, "Call fit() first"
            X = self.vectorizer.transform(texts)
            X = X.astype(np.float32)
            X = X.toarray()
            X = l2_normalize(X)
            return X


# --------------------------- Visualizations ---------------------------

def per_episode_assumption_timeline(assump_df: pd.DataFrame,
                                    X_assump: np.ndarray,
                                    out_dir: Path,
                                    sim_threshold: float = 0.60,
                                    max_plots: int = 9999,
                                    seed: int = 0) -> None:
    """
    Two-lane timeline by top-2 speakers (speaker_id first, fallback speaker_key).
    - Plot every assumption as a dot at x = turn_idx (+ jitter), y = lane (1 for A, 0 for B)
    - For each assumption by A, link to its nearest neighbor in B if cosine >= sim_threshold

    IMPORTANT: This function assumes X_assump rows align 1:1 with assump_df rows (global index).
    We first slice X_assump down to the episode's rows, THEN apply boolean masks.
    """
    rng = np.random.default_rng(seed)
    ensure_out_dir(out_dir)

    episodes = assump_df["episode_file"].unique().tolist()
    plotted = 0

    for ep in episodes:
        if plotted >= max_plots:
            break

        df = assump_df[assump_df["episode_file"] == ep].copy()
        if df.empty:
            continue

        # Align embeddings with THIS episode slice up front (FIX)
        idx_global = df.index.to_numpy()
        X_df = X_assump[idx_global, :]

        # Use speaker_id if present; else fallback to speaker_key
        df["sid_used"] = df["speaker_id"].replace("", np.nan).fillna(df["speaker_key"])

        # Pick the top-2 speakers by assumption count
        counts = df["sid_used"].value_counts()
        if len(counts) < 2:
            continue
        sidA, sidB = counts.index[:2].tolist()

        A_mask = (df["sid_used"] == sidA).values
        B_mask = (df["sid_used"] == sidB).values
        A_ix = np.where(A_mask)[0]  # indices within df
        B_ix = np.where(B_mask)[0]

        if len(A_ix) == 0 or len(B_ix) == 0:
            continue

        # Embeddings for A/B assumptions (now safely aligned)
        X_A = X_df[A_mask]
        X_B = X_df[B_mask]

        # Nearest neighbor across speakers (A -> B)
        nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_B)
        dist, idx = nbrs.kneighbors(X_A)

        # Pairs above threshold
        edges = []
        for i, (d, j) in enumerate(zip(dist[:, 0], idx[:, 0])):
            sim = 1.0 - float(d)
            if sim >= sim_threshold:
                ai = A_ix[i]
                bi = B_ix[j]
                edges.append((ai, bi, sim))

        # Jitter so dots don’t overlap on identical turns
        def jitter(n): return rng.normal(0, 0.05, size=n)

        x_A = df.iloc[A_ix]["turn_idx"].to_numpy(dtype=float) + jitter(len(A_ix))
        y_A = np.ones(len(A_ix))
        x_B = df.iloc[B_ix]["turn_idx"].to_numpy(dtype=float) + jitter(len(B_ix))
        y_B = np.zeros(len(B_ix))

        # Plot
        plt.figure(figsize=(10, 4))
        plt.scatter(x_A, y_A, s=20, label=str(sidA), alpha=0.85)
        plt.scatter(x_B, y_B, s=20, label=str(sidB), alpha=0.85, marker="s")

        # Cross-speaker links with width by similarity
        for ai, bi, sim in edges:
            xa = float(df.iloc[ai]["turn_idx"]) + float(rng.normal(0, 0.02))
            xb = float(df.iloc[bi]["turn_idx"]) + float(rng.normal(0, 0.02))
            plt.plot([xa, xb], [1.0, 0.0], linewidth=1.0 + 2.0 * sim, alpha=0.25)

        plt.yticks([0, 1], [str(sidB), str(sidA)])
        plt.ylim(-0.5, 1.5)
        plt.xlabel("Turn index")
        plt.title(f"Assumption Timeline with Cross-Speaker Links\n{ep}\n(threshold={sim_threshold:.2f})")
        plt.legend(loc="upper right", fontsize=8)
        out_path = out_dir / f"{Path(ep).stem}_assumption_timeline.png"
        save_fig(out_path)
        plotted += 1
        log.info(f"Wrote assumption timeline: {out_path}")


def per_episode_assumption_tsne(assump_df: pd.DataFrame,
                                X_assump: np.ndarray,
                                out_dir: Path,
                                highlight_pairs: bool = True,
                                sim_threshold: float = 0.60,
                                max_plots: int = 9999,
                                seed: int = 0) -> None:
    """
    t-SNE map of all assumptions in an episode.
    Color by speaker_id (or speaker_key if id missing). Optionally highlight
    assumptions that form strong cross-speaker pairs (sim >= threshold).
    """
    ensure_out_dir(out_dir)
    episodes = assump_df["episode_file"].unique().tolist()
    plotted = 0

    for ep in episodes:
        if plotted >= max_plots:
            break

        df = assump_df[assump_df["episode_file"] == ep].copy()
        if df.empty:
            continue

        df["sid_used"] = df["speaker_id"].replace("", np.nan).fillna(df["speaker_key"])
        sids = df["sid_used"].unique().tolist()
        if len(sids) < 1:
            continue

        # Slice embeddings using the episode's row indices (already fixed alignment)
        X = X_assump[df.index.to_numpy(), :]

        N = X.shape[0]
        if N < 3:
            continue

        # t-SNE params
        perplexity = max(5, min(30, N // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="random", learning_rate="auto")
        Y = tsne.fit_transform(X)

        # Identify strong cross-speaker pairs (optional)
        strong_mask = np.zeros(N, dtype=bool)
        edges = []
        if highlight_pairs and len(sids) >= 2:
            top2 = df["sid_used"].value_counts().index[:2].tolist()
            if len(top2) == 2:
                A_mask = (df["sid_used"] == top2[0]).values
                B_mask = (df["sid_used"] == top2[1]).values
                A_ix = np.where(A_mask)[0]
                B_ix = np.where(B_mask)[0]
                if len(A_ix) and len(B_ix):
                    X_A = X[A_mask]
                    X_B = X[B_mask]
                    nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_B)
                    dist, idx = nbrs.kneighbors(X_A)
                    for i, (d, j) in enumerate(zip(dist[:, 0], idx[:, 0])):
                        sim = 1.0 - float(d)
                        if sim >= sim_threshold:
                            ai = A_ix[i]
                            bi = B_ix[j]
                            edges.append((ai, bi, sim))
                            strong_mask[ai] = True
                            strong_mask[bi] = True

        # Color map by speaker
        sid_list = df["sid_used"].tolist()
        unique_sids = sorted(df["sid_used"].unique().tolist())
        color_map = {sid: plt.get_cmap("tab20")(i % 20) for i, sid in enumerate(unique_sids)}
        colors = [color_map[sid] for sid in sid_list]

        plt.figure(figsize=(6.5, 5.5))
        # base points
        plt.scatter(Y[:, 0], Y[:, 1], s=18, c=colors, alpha=0.70, edgecolors="none")
        # highlight strong-pair points
        if edges:
            plt.scatter(Y[strong_mask, 0], Y[strong_mask, 1], s=30, facecolors="none", edgecolors="k", linewidths=0.6)
            for ai, bi, sim in edges:
                plt.plot([Y[ai, 0], Y[bi, 0]], [Y[ai, 1], Y[bi, 1]], alpha=0.25, linewidth=0.5)

        # Legend (top few speakers)
        for i, sid in enumerate(unique_sids[:10]):
            plt.scatter([], [], c=[color_map[sid]], label=str(sid))
        plt.legend(title="speaker", loc="best", fontsize=8)

        plt.title(f"Per-Assumption t-SNE Map\n{ep}  (N={N}, perplexity={perplexity})")
        plt.axis("off")
        out_path = out_dir / f"{Path(ep).stem}_assumption_tsne.png"
        save_fig(out_path)
        plotted += 1
        log.info(f"Wrote t-SNE map: {out_path}")


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-assumption visualizations: timeline links + t-SNE maps.")
    ap.add_argument("--input_dir", type=str, default="results/covid",
                    help="Directory of per-episode JSON files (default: results/covid).")
    ap.add_argument("--out_dir", type=str, default="viz_out", help="Output directory for plots.")
    ap.add_argument("--embed_backend", type=str, default="sbert", choices=["sbert", "tfidf"], help="Embedding backend.")
    ap.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2", help="Sentence-Transformer model name.")
    ap.add_argument("--device", type=str, default=None, help="Device hint for SBERT (e.g., 'cuda' or 'cpu').")

    # Timeline & pairing controls
    ap.add_argument("--assump_sim_threshold", type=float, default=0.60, help="Cross-speaker similarity threshold.")
    ap.add_argument("--max_assump_plots", type=int, default=9999, help="Max episodes to plot for per-assumption views.")

    # t-SNE controls
    ap.add_argument("--seed", type=int, default=0, help="Random seed for jitter/t-SNE.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        log.error("Input dir not found: %s", input_dir)
        return
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    # 1) Load and expand to per-assumption rows
    turns = load_turns(input_dir)
    if turns.empty:
        log.error("No data loaded. Exiting.")
        return
    A = build_assumption_table(turns)
    if A.empty:
        log.error("No individual assumptions to visualize. Exiting.")
        return

    # 2) Embed individual assumptions (X_assump row order aligns with A row order)
    texts = A["assumption_text"].tolist()
    embedder = Embedder(backend=args.embed_backend, sbert_model=args.sbert_model, device=args.device)
    log.info(f"Embedding {len(texts)} assumptions with backend='{embedder.backend}'")
    embedder.fit(texts)
    X_assump = embedder.encode(texts)
    log.info(f"Assumption embeddings shape: {X_assump.shape}")

    # 3) Per-episode assumption timeline with cross-speaker links (FIXED indexing)
    per_episode_assumption_timeline(
        assump_df=A,                     # keep original index to align with X_assump
        X_assump=X_assump,
        out_dir=out_dir / "assumptions_timeline",
        sim_threshold=args.assump_sim_threshold,
        max_plots=args.max_assump_plots,
        seed=args.seed,
    )

    # 4) Per-episode assumption t-SNE maps (already uses df.index to slice X)
    per_episode_assumption_tsne(
        assump_df=A,                     # keep original index to align with X_assump
        X_assump=X_assump,
        out_dir=out_dir / "assumptions_tsne",
        highlight_pairs=True,
        sim_threshold=args.assump_sim_threshold,
        max_plots=args.max_assump_plots,
        seed=args.seed,
    )

    log.info("All done. Check output images in: %s", out_dir)


if __name__ == "__main__":
    main()
