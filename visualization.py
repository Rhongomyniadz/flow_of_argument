import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import SentenceTransformer; fall back to TF-IDF if unavailable
_HAS_SBERT = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _HAS_SBERT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-visualizer")


# --------------------------- Utilities ---------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    return sk_normalize(X, norm="l2", copy=False)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def pairwise_cosine_distances(vectors: List[np.ndarray]) -> List[float]:
    """Return 1 - cosine similarities for all unique pairs."""
    dists = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - cosine(vectors[i], vectors[j])
            dists.append(d)
    return dists


# --------------------------- Data Loading ---------------------------

def load_turns(input_dir: Path) -> pd.DataFrame:
    """
    Load all JSON files from input_dir. Each file is considered one episode.
    Returns a DataFrame with columns:
      episode_file, turn_idx, speaker_id, speaker_key, inferred_speaker_name,
      inferred_speaker_role, role_raw, assumptions (list[str]), doc_text (joined)
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
                doc_text = " ; ".join([a for a in assumptions if isinstance(a, str) and a.strip()])
                if not doc_text.strip():
                    continue

                # --- normalize speaker_id ---
                sid_raw = rec.get("speaker_id", None)
                sid_norm: Optional[str] = None
                if isinstance(sid_raw, list):
                    # common case: ["SPEAKER_01"]
                    sid_norm = "-".join(str(x) for x in sid_raw) if sid_raw else None
                elif isinstance(sid_raw, dict):
                    sid_norm = sid_raw.get("id") or sid_raw.get("speaker_id")
                    sid_norm = str(sid_norm) if sid_norm is not None else None
                elif sid_raw is not None:
                    sid_norm = str(sid_raw)

                # --- names/roles (may be NO_INFERRED_*) ---
                name = (rec.get("inferred_speaker_name") or "").strip()
                role = (rec.get("inferred_speaker_role") or "").strip().lower()

                # Prefer id for grouping; fallback to name; final fallback to UNKNOWN_SPEAKER
                speaker_key = sid_norm or (name if name and not name.startswith("NO_INFERRED") else None) or "UNKNOWN_SPEAKER"

                rows.append({
                    "episode_file": fp.name,
                    "turn_idx": i,
                    "speaker_id": sid_norm,
                    "speaker_key": speaker_key,
                    "inferred_speaker_name": name if name else "NO_INFERRED_SPEAKER",
                    "inferred_speaker_role": role if role else "unknown",
                    "role_raw": role,  # may be 'host','guest','unknown','no_inferred_role', etc.
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


def load_metadata(meta_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load optional metadata CSV: episode_file,podcast_id,podcast_category,topic"""
    if meta_csv is None:
        return None
    try:
        meta = pd.read_csv(meta_csv)
        required = {"episode_file", "podcast_id", "podcast_category"}
        if not required.issubset(set(meta.columns)):
            log.warning(f"Metadata CSV missing columns: {required - set(meta.columns)}")
            return None
        # Normalize strings
        meta["episode_file"] = meta["episode_file"].astype(str)
        meta["podcast_id"] = meta["podcast_id"].astype(str)
        meta["podcast_category"] = meta["podcast_category"].astype(str)
        log.info(f"Loaded metadata for {len(meta)} episodes")
        return meta
    except Exception as e:
        log.warning(f"Failed to read metadata CSV: {e}")
        return None


# --------------------------- Role Imputation ---------------------------

def impute_roles(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing/unknown roles by taking the per-(episode, speaker_key) majority role
    among observed roles in that episode. Falls back to 'unknown'.
    """
    df = turns.copy()

    def norm_role(r: str) -> Optional[str]:
        r = (r or "").lower()
        if r in ("", "unknown", "no_inferred_role", "no_inferred", "none", "n/a"):
            return None
        return r

    df["role_clean"] = df["inferred_speaker_role"].apply(norm_role)

    # Majority role per (episode, speaker_key) among known roles
    maj = (
        df.dropna(subset=["role_clean"])
          .groupby(["episode_file", "speaker_key"])["role_clean"]
          .agg(lambda s: s.value_counts().idxmax())
    )

    df = df.join(maj.rename("role_majority"), on=["episode_file", "speaker_key"])
    df["role_used"] = df["role_clean"].fillna(df["role_majority"]).fillna("unknown")
    return df


# --------------------------- Embeddings ---------------------------

class Embedder:
    def __init__(self, backend: str, sbert_model: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        backend: 'sbert' or 'tfidf'
        """
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
            X = X.toarray()  # dense; OK for moderate sizes
            X = l2_normalize(X)
            return X


# --------------------------- Aggregations ---------------------------

def aggregate_mean(vectors: np.ndarray, indices: List[int]) -> np.ndarray:
    """Return the mean vector over rows[indices]."""
    if len(indices) == 0:
        return np.zeros((vectors.shape[1],), dtype=np.float32)
    return np.mean(vectors[indices, :], axis=0)


# --------------------------- Plotting helpers ---------------------------

def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def speaker_variability_boxplot(speaker_episode_vecs: Dict[Tuple[str, str], np.ndarray],
                                out_path: Path,
                                min_episodes: int = 2) -> None:
    """Compute within-speaker pairwise distances across episodes and plot distributions."""
    # Group by speaker -> list of episode vectors
    groups: Dict[str, List[np.ndarray]] = {}
    for (speaker, episode), vec in speaker_episode_vecs.items():
        groups.setdefault(speaker, []).append(vec)

    labels, box_data = [], []
    for speaker, vecs in groups.items():
        if len(vecs) < min_episodes:
            continue
        dists = pairwise_cosine_distances(vecs)
        if dists:
            labels.append(speaker)
            box_data.append(dists)

    if not box_data:
        log.warning("No speakers with >=2 episodes to plot variability.")
        return

    plt.figure(figsize=(max(8, len(labels) * 0.5), 6))
    plt.boxplot(box_data, vert=True, showfliers=False)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=45, ha="right")
    plt.ylabel("Pairwise distance (1 - cosine)")
    plt.title("Speaker Variability Across Episodes (smaller = more consistent)")
    save_fig(out_path)
    log.info(f"Wrote speaker variability boxplot: {out_path}")


def speaker_similarity_graph(speaker_global_vec: Dict[str, np.ndarray],
                             episodes_per_speaker: Dict[str, int],
                             label_map: Optional[Dict[str, str]],
                             out_path: Path,
                             sim_threshold: float = 0.70,
                             seed: int = 0,
                             knn_k: int = 8) -> None:
    """
    Speaker-similarity graph using cosine; falls back to k-NN if threshold yields no edges.
    Nodes are speaker_id strings; roles are ignored.
    """
    speakers = list(speaker_global_vec.keys())
    if len(speakers) < 2:
        log.warning("Not enough speakers for a graph.")
        return

    X = np.vstack([speaker_global_vec[s] for s in speakers])
    S = cosine_similarity(X)

    G = nx.Graph()
    for s in speakers:
        G.add_node(s, episodes=episodes_per_speaker.get(s, 1))

    # First try thresholded edges
    n = len(speakers)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(S[i, j])
            if w >= sim_threshold:
                G.add_edge(speakers[i], speakers[j], weight=w)

    # Fallback: k-NN if nothing passed threshold
    if G.number_of_edges() == 0:
        log.info("No edges above threshold (%.2f). Falling back to k-NN (k=%d).", sim_threshold, knn_k)
        nbrs = NearestNeighbors(n_neighbors=min(knn_k, len(speakers)), metric="cosine").fit(X)
        dist, idx = nbrs.kneighbors(X)
        for i, sid in enumerate(speakers):
            for j, neighbor_ix in enumerate(idx[i][1:], start=1):  # skip self
                sim = 1.0 - float(dist[i][j])
                G.add_edge(sid, speakers[neighbor_ix], weight=sim)

    sizes = [5 + 45 * math.log1p(G.nodes[s]["episodes"]) for s in G.nodes]
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=1.0 / math.sqrt(max(1, G.number_of_nodes())))
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=[2 * G.edges[e]["weight"] for e in G.edges])
    nx.draw_networkx_nodes(G, pos, node_size=sizes)
    # Label top-degree nodes
    deg = dict(G.degree())
    label_nodes = {n for n, d in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:25]}
    if label_map:
        labels = {n: label_map.get(n, n) for n in label_nodes}
    else:
        labels = {n: n for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title("Speaker–Speaker Assumption Similarity Graph (by speaker_id)")
    plt.axis("off")
    save_fig(out_path)
    log.info(f"Wrote speaker similarity graph: {out_path}")


def podcast_similarity_graph(podcast_vecs: Dict[str, np.ndarray],
                             podcast_category: Dict[str, str],
                             out_path: Path,
                             knn_k: int = 8,
                             sim_threshold: float = 0.65,
                             seed: int = 42) -> None:
    """Build k-NN graph among podcasts; color by category."""
    if not podcast_vecs:
        log.warning("No podcast vectors available for podcast graph.")
        return
    pods = list(podcast_vecs.keys())
    X = np.vstack([podcast_vecs[p] for p in pods])

    # Build kNN on cosine distance (1 - cosine sim)
    nbrs = NearestNeighbors(n_neighbors=min(knn_k, len(pods)), metric="cosine", algorithm="auto").fit(X)
    dist, idx = nbrs.kneighbors(X)

    G = nx.Graph()
    for i, pid in enumerate(pods):
        G.add_node(pid, category=podcast_category.get(pid, "unknown"))
        # connect to neighbors except self
        for j, neighbor_ix in enumerate(idx[i][1:], start=1):
            sim = 1.0 - float(dist[i][j])
            if sim >= sim_threshold:
                G.add_edge(pid, pods[neighbor_ix], weight=sim)

    # Color by category
    cats = sorted({G.nodes[n]["category"] for n in G.nodes})
    cmap = plt.get_cmap("tab20")
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
    node_colors = [color_map[G.nodes[n]["category"]] for n in G.nodes]

    pos = nx.spring_layout(G, seed=seed, weight="weight")
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=[2 * G.edges[e]["weight"] for e in G.edges])
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)
    # Label top-degree nodes
    deg = dict(G.degree())
    label_nodes = {n for n, d in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:25]}
    nx.draw_networkx_labels(G, pos, labels={n: n for n in label_nodes}, font_size=8)
    # Legend
    for cat, color in color_map.items():
        plt.scatter([], [], c=[color], label=cat)
    plt.legend(title="Podcast Category", loc="best", fontsize=8)
    plt.title("Podcast Similarity Graph (assumptions; edges = high cosine similarity)")
    plt.axis("off")
    save_fig(out_path)
    log.info(f"Wrote podcast similarity graph: {out_path}")


def host_guest_similarity_over_time(ep_df: pd.DataFrame,
                                    turn_vectors: Dict[Tuple[str, int], np.ndarray],
                                    out_dir: Path,
                                    window: int = 5,
                                    max_plots: int = 9999) -> None:
    """
    For each episode, compute rolling host vs guest similarity over turn index.
    Saves one line plot per episode (up to max_plots).
    """
    ensure_out_dir(out_dir)
    episodes = ep_df["episode_file"].unique().tolist()
    plotted = 0
    for ep in episodes:
        if plotted >= max_plots:
            break
        df_e = ep_df[ep_df["episode_file"] == ep].sort_values("turn_idx").reset_index(drop=True)
        if df_e.empty:
            continue

        sim_series = []
        xs = []

        # Maintain rolling lists of role-specific vectors
        H_roll: List[np.ndarray] = []
        G_roll: List[np.ndarray] = []

        def avg(V: List[np.ndarray]) -> Optional[np.ndarray]:
            if not V:
                return None
            return np.mean(np.stack(V, axis=0), axis=0)

        for _, row in df_e.iterrows():
            key = (row["episode_file"], int(row["turn_idx"]))
            v = turn_vectors.get(key, None)
            if v is None:
                continue
            role = (row["role_used"] or "unknown").lower()
            if "host" in role:
                H_roll.append(v)
                if len(H_roll) > window:
                    H_roll.pop(0)
            elif "guest" in role:
                G_roll.append(v)
                if len(G_roll) > window:
                    G_roll.pop(0)
            else:
                # other roles don't change rolling windows
                pass

            h_avg = avg(H_roll)
            g_avg = avg(G_roll)
            if h_avg is not None and g_avg is not None:
                sim = cosine(h_avg, g_avg)
                sim_series.append(sim)
            else:
                sim_series.append(np.nan)
            xs.append(row["turn_idx"])

        if len(xs) < 3:
            continue

        plt.figure(figsize=(9, 4))
        plt.plot(xs, sim_series)
        plt.ylim(0, 1)
        plt.xlabel("Turn index")
        plt.ylabel("Host–Guest similarity (cosine)")
        plt.title(f"Host vs Guest Assumption Alignment Over Time\n{ep}")
        out_path = out_dir / f"{Path(ep).stem}_host_guest_similarity.png"
        save_fig(out_path)
        plotted += 1
        log.info(f"Wrote host/guest time plot: {out_path}")


# --------------------------- Main Pipeline ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize assumption similarity using graphs and timelines (speaker_id–strict).")
    ap.add_argument("--input_dir", type=str, default="results/covid",
                    help="Directory of per-episode JSON files (default: results/covid).")
    ap.add_argument("--meta_csv", type=str, default=None,
                    help="Optional metadata CSV (episode_file,podcast_id,podcast_category,topic).")
    ap.add_argument("--out_dir", type=str, default="viz_out", help="Output directory for plots.")
    ap.add_argument("--embed_backend", type=str, default="sbert", choices=["sbert", "tfidf"], help="Embedding backend.")
    ap.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2", help="Sentence-Transformer model name.")
    ap.add_argument("--device", type=str, default=None, help="Device hint for SBERT (e.g., 'cuda' or 'cpu').")
    ap.add_argument("--sim_threshold", type=float, default=0.70, help="Similarity threshold for graphs.")
    ap.add_argument("--knn_k", type=int, default=8, help="k for kNN fallback/graphs.")
    ap.add_argument("--rolling_w", type=int, default=5, help="Window for host/guest rolling similarity.")
    ap.add_argument("--max_timeplots", type=int, default=9999, help="Max episodes to plot for time series.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        log.error("Input dir not found: %s", input_dir)
        return
    log.info("Using input dir: %s", input_dir)

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    # 1) Load turns
    turns = load_turns(input_dir)
    if turns.empty:
        log.error("No data loaded. Exiting.")
        return

    # 2) Optional metadata (for podcast-level graph)
    meta = load_metadata(Path(args.meta_csv)) if args.meta_csv else None
    if meta is not None:
        turns = turns.merge(meta[["episode_file", "podcast_id", "podcast_category"]], on="episode_file", how="left")
    else:
        # placeholders
        turns["podcast_id"] = turns["episode_file"]  # fallback to episode-level
        turns["podcast_category"] = "unknown"

    # 3) Impute roles so host/guest timelines are robust
    turns = impute_roles(turns)

    # 4) Fit embeddings on per-turn doc_text, then encode all turns
    texts = turns["doc_text"].tolist()
    embedder = Embedder(backend=args.embed_backend, sbert_model=args.sbert_model, device=args.device)
    log.info(f"Fitting embedder on {len(texts)} turn documents using backend='{embedder.backend}'")
    embedder.fit(texts)
    X_turns = embedder.encode(texts)
    log.info(f"Turn embeddings shape: {X_turns.shape}")

    # Key lookup for per-turn vectors
    turn_key = list(zip(turns["episode_file"], turns["turn_idx"]))
    turn_vectors: Dict[Tuple[str, int], np.ndarray] = {k: v for k, v in zip(turn_key, X_turns)}

    # 5) -------- Speaker aggregates STRICTLY by speaker_id (ignore roles) --------
    turns["_row_ix"] = np.arange(len(turns))

    turns_id = turns[turns["speaker_id"].notna() & (turns["speaker_id"] != "")].copy()
    if turns_id.empty:
        log.warning("No turns have a speaker_id — skipping speaker graphs.")
        speaker_episode_vecs = {}
        speaker_global_vec = {}
        episodes_per_speaker = {}
        label_map = {}
    else:
        # (speaker_id, episode) vectors
        se_groups = turns_id.groupby(["speaker_id", "episode_file"])["_row_ix"].apply(list)
        speaker_episode_vecs: Dict[Tuple[str, str], np.ndarray] = {
            (sid, ep): aggregate_mean(X_turns, idxs) for (sid, ep), idxs in se_groups.items()
        }

        # Global per speaker_id vectors
        s_groups = turns_id.groupby("speaker_id")["_row_ix"].apply(list)
        speaker_global_vec: Dict[str, np.ndarray] = {
            sid: aggregate_mean(X_turns, idxs) for sid, idxs in s_groups.items()
        }
        episodes_per_speaker: Dict[str, int] = turns_id.groupby("speaker_id")["episode_file"].nunique().to_dict()

        # Labels: "SPEAKER_ID | Most-common non-NO name"
        name_by_id = (
            turns_id.loc[turns_id["inferred_speaker_name"] != "NO_INFERRED_SPEAKER"]
                    .groupby("speaker_id")["inferred_speaker_name"]
                    .agg(lambda s: s.value_counts().idxmax())
        )
        label_map: Dict[str, str] = {sid: f"{sid} | {name_by_id.get(sid, '')}".strip(" |") for sid in speaker_global_vec}

    # 6) -------- Podcast aggregates --------
    p_groups = turns.groupby("podcast_id")["_row_ix"].apply(list)
    podcast_vecs: Dict[str, np.ndarray] = {pid: aggregate_mean(X_turns, idxs) for pid, idxs in p_groups.items()}
    podcast_category: Dict[str, str] = turns.drop_duplicates("podcast_id").set_index("podcast_id")["podcast_category"].to_dict()

    # 7) -------- Plots --------

    # 7a) Speaker variability across episodes (by speaker_id)
    if speaker_episode_vecs:
        speaker_variability_boxplot(
            speaker_episode_vecs=speaker_episode_vecs,
            out_path=out_dir / "speaker_variability_across_episodes.png",
            min_episodes=2,
        )
    else:
        log.warning("Skipping speaker variability plot (no speaker_id episode vectors).")

    # 7b) Speaker–Speaker similarity graph (speaker_id-based, with k-NN fallback)
    if speaker_global_vec:
        speaker_similarity_graph(
            speaker_global_vec=speaker_global_vec,
            episodes_per_speaker=episodes_per_speaker,
            label_map=label_map,
            out_path=out_dir / "speaker_similarity_graph.png",
            sim_threshold=args.sim_threshold,
            seed=0,
            knn_k=args.knn_k,
        )
    else:
        log.warning("Skipping speaker similarity graph (no speaker_id vectors).")

    # 7c) Podcast similarity graph (kNN; colored by category)
    podcast_similarity_graph(
        podcast_vecs=podcast_vecs,
        podcast_category=podcast_category,
        out_path=out_dir / "podcast_similarity_graph.png",
        knn_k=args.knn_k,
        sim_threshold=args.sim_threshold,
        seed=42,
    )

    # 7d) Host vs Guest similarity over time (one file per episode, up to max)
    host_guest_similarity_over_time(
        ep_df=turns,
        turn_vectors=turn_vectors,
        out_dir=out_dir / "host_guest_over_time",
        window=args.rolling_w,
        max_plots=args.max_timeplots,
    )

    log.info("All done. Check output images in: %s", out_dir)


if __name__ == "__main__":
    main()
