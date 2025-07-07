import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_tsne(coords, labels, podcasts, selected_info, title, out_path):
    """
    Plots a t-SNE scatter of all context windows (coords), color-coded by Podcast,
    and annotates the selected windows by their WindowIndex.

    selected_info: list of (filtered_idx, window_index)
    """
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')

    # Plot all points per podcast
    for idx, podcast in enumerate(podcasts):
        mask = (labels == idx)
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(idx)],
            label=podcast,
            s=40,
            alpha=0.3
        )

    # Annotate only selected windows with their true WindowIndex
    for filtered_idx, win_idx in selected_info:
        x, y = coords[filtered_idx]
        plt.annotate(
            str(win_idx),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            weight='bold',
            color='black'
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
    OUT_ASS_PLOT  = 'results/assumptions_tsne_all.png'
    OUT_KP_PLOT   = 'results/keypoints_tsne_all.png'
    SEED          = 42
    MODEL_NAME    = 'all-MiniLM-L6-v2'
    HF_CACHE      = os.getenv('HF_HOME', None)
    MAX_SAMPLES   = 3
    RANDOM_STATE  = 42

    # Ensure results dir
    os.makedirs('results', exist_ok=True)
    random.seed(SEED)

    # 1) Load and filter windows with non-empty KeyPoints & Assumptions
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_windows = json.load(f)

    filtered = [
        w for w in all_windows
        if w.get("KeyPoints") and len(w["KeyPoints"]) > 0
        and w.get("Assumptions") and len(w["Assumptions"]) > 0
    ]
    n_total = len(filtered)
    print(f"Filtered to {n_total} windows with non-empty KeyPoints & Assumptions.")

    # 2) Sample up to MAX_SAMPLES windows per podcast
    by_podcast = defaultdict(list)
    for i, w in enumerate(filtered):
        by_podcast[w['Podcast']].append(i)

    # Build selected info: list of tuples (filtered_idx, WindowIndex)
    selected_info = []  # (filtered_idx, original WindowIndex)
    for idxs in by_podcast.values():
        sampled = idxs if len(idxs) <= MAX_SAMPLES else random.sample(idxs, MAX_SAMPLES)
        for fi in sampled:
            selected_info.append((fi, filtered[fi]['WindowIndex']))

    # Save selected windows
    selected_windows = [filtered[fi] for fi, _ in selected_info]
    with open(SELECTED_JSON, 'w', encoding='utf-8') as f:
        json.dump(selected_windows, f, indent=2)
    print(f"Saved {len(selected_windows)} sampled windows to {SELECTED_JSON}")

    # 3) Prepare DataFrame of all filtered windows
    df_all = pd.DataFrame(filtered)
    df_all['AssumpText'] = df_all['Assumptions'].apply(lambda lst: ' '.join(lst))
    df_all['KPText']     = df_all['KeyPoints'].apply(lambda lst: ' '.join(lst))

    # 4) Embedding
    embedder = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)
    ass_emb = embedder.encode(df_all['AssumpText'].tolist(), show_progress_bar=True)
    kp_emb  = embedder.encode(df_all['KPText'].tolist(), show_progress_bar=True)

    # Determine valid perplexity (< n_samples)
    perplexity = min(30, max(2, n_total - 1))

    # 5) TSNE fits
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    tsne_ass = tsne.fit_transform(ass_emb)
    tsne_kp  = tsne.fit_transform(kp_emb)

    # 6) Labels for plotting
    labels, podcasts = pd.factorize(df_all['Podcast'])

    # 7) Plot all windows, label selected by WindowIndex
    plot_tsne(
        tsne_ass, labels, podcasts, selected_info,
        "t-SNE of All Assumption Embeddings (Selected Highlighted)", OUT_ASS_PLOT
    )
    plot_tsne(
        tsne_kp, labels, podcasts, selected_info,
        "t-SNE of All Key-Point Embeddings (Selected Highlighted)", OUT_KP_PLOT
    )

if __name__ == "__main__":
    main()
