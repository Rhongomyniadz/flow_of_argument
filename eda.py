import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_tsne(coords, labels, podcasts, title, out_path):
    """
    Plots a t-SNE scatter of the given coords,
    color-coded by Podcast, without annotations.
    """
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')

    # Plot all points by podcast
    for idx, podcast in enumerate(podcasts):
        mask = (labels == idx)
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(idx)],
            label=podcast,
            s=40,
            alpha=0.5
        )

    plt.title(title)
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.legend(title="Podcast", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    plt.close()


def main():
    # --- CONFIG ---
    INPUT_PATH    = 'results/news_sample_sliding_window_vllm.json'
    SELECTED_JSON = 'results/selected_windows.json'
    OUT_ASS_PLOT  = 'results/assumptions_tsne_selected.png'
    OUT_KP_PLOT   = 'results/keypoints_tsne_selected.png'
    SEED          = 42
    MODEL_NAME    = 'all-MiniLM-L6-v2'
    HF_CACHE      = os.getenv('HF_HOME', None)
    MAX_SAMPLES   = 3
    RANDOM_STATE  = 42

    # ensure results dir
    os.makedirs('results', exist_ok=True)
    random.seed(SEED)

    # 1) Load and filter windows
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_windows = json.load(f)

    filtered = [
        w for w in all_windows
        if w.get("KeyPoints") and w.get("Assumptions")
    ]
    n = len(filtered)
    print(f"Kept {n} windows after filtering non-empty KeyPoints & Assumptions.")

    # 2) Sample up to MAX_SAMPLES windows per podcast
    by_podcast = defaultdict(list)
    for i, w in enumerate(filtered):
        by_podcast[w['Podcast']].append(i)

    selected_indices = []
    for idxs in by_podcast.values():
        if len(idxs) <= MAX_SAMPLES:
            selected_indices.extend(idxs)
        else:
            selected_indices.extend(random.sample(idxs, MAX_SAMPLES))

    selected_windows = [filtered[i] for i in selected_indices]
    with open(SELECTED_JSON, 'w', encoding='utf-8') as f:
        json.dump(selected_windows, f, indent=2)
    print(f"Saved {len(selected_windows)} sampled windows to {SELECTED_JSON}")

    # 3) Prepare DataFrame
    df = pd.DataFrame(selected_windows)
    df['AssumpText'] = df['Assumptions'].apply(lambda lst: ' '.join(lst))
    df['KPText']     = df['KeyPoints'].apply(lambda lst: ' '.join(lst))

    # 4) Embedding
    embedder = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)

    ass_emb = embedder.encode(df['AssumpText'].tolist(), show_progress_bar=True)
    kp_emb  = embedder.encode(df['KPText'].tolist(), show_progress_bar=True)

    # Determine appropriate perplexity (must be < n_samples)
    perplexity = min(30, max(2, len(ass_emb) - 1))

    # 5) TSNE on same random embedding init
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    tsne_ass = tsne.fit_transform(ass_emb)
    tsne_kp  = tsne.fit_transform(kp_emb)

    # 6) Labels for plotting
    labels, podcasts = pd.factorize(df['Podcast'])

    # 7) Plot both
    plot_tsne(tsne_ass, labels, podcasts, "t-SNE of Assumption Embeddings (Selected)", OUT_ASS_PLOT)
    plot_tsne(tsne_kp,  labels, podcasts, "t-SNE of Key-Point Embeddings (Selected)", OUT_KP_PLOT)

if __name__ == "__main__":
    main()
