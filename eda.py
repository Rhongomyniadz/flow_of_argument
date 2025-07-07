import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_tsne(coords, labels, podcasts, selected_indices, title, out_path):
    """
    Plots a t-SNE scatter of the given coords,
    color-coded by Podcast, and annotates the selected indices.
    """
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')

    # plot all points
    for idx, podcast in enumerate(podcasts):
        mask = (labels == idx)
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(idx)],
            label=podcast,
            s=40,
            alpha=0.5
        )

    # annotate sampled points
    for sel in selected_indices:
        x, y = coords[sel]
        plt.annotate(
            str(sel),
            (x, y),
            textcoords="offset points",
            xytext=(3, 3),
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
    INPUT_PATH     = 'results/news_sample_sliding_window_vllm.json'
    SELECTED_JSON  = 'results/selected_windows.json'
    OUT_ASS_PLOT   = 'results/assumptions_tsne_selected.png'
    OUT_KP_PLOT    = 'results/keypoints_tsne_selected.png'
    SEED           = 42
    MODEL_NAME     = 'all-MiniLM-L6-v2'
    HF_CACHE       = os.getenv('HF_HOME', None)
    MAX_SAMPLES    = 3
    TSNE_STATE     = 42

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
    print(f"Kept {len(filtered)} windows after filtering non-empty KeyPoints & Assumptions.")

    # 2) Sample up to MAX_SAMPLES windows per podcast
    by_podcast = defaultdict(list)
    for i, w in enumerate(filtered):
        by_podcast[w['Podcast']].append(i)  # store index

    selected_indices = []
    for podcast, idxs in by_podcast.items():
        if len(idxs) <= MAX_SAMPLES:
            sel = idxs
        else:
            sel = random.sample(idxs, MAX_SAMPLES)
        selected_indices.extend(sel)

    # 3) Save the selected windows
    selected_windows = [filtered[i] for i in selected_indices]
    with open(SELECTED_JSON, 'w', encoding='utf-8') as f:
        json.dump(selected_windows, f, indent=2)
    print(f"Saved {len(selected_windows)} sampled windows to {SELECTED_JSON}")

    # 4) Prepare DataFrame of selected windows
    df_sel = pd.DataFrame(selected_windows)
    df_sel['AssumpText'] = df_sel['Assumptions'].apply(lambda lst: ' '.join(lst))
    df_sel['KPText']     = df_sel['KeyPoints'].apply(lambda lst: ' '.join(lst))

    # 5) Initialize embedder
    embedder = SentenceTransformer(
        MODEL_NAME,
        cache_folder=HF_CACHE
    )

    # 6a) Embed & TSNE assumptions on selected windows only
    ass_emb = embedder.encode(df_sel['AssumpText'].tolist(), show_progress_bar=True)
    tsne_ass = TSNE(n_components=2, random_state=TSNE_STATE).fit_transform(ass_emb)
    # 6b) Embed & TSNE keypoints on selected windows only
    kp_emb = embedder.encode(df_sel['KPText'].tolist(), show_progress_bar=True)
    tsne_kp = TSNE(n_components=2, random_state=TSNE_STATE).fit_transform(kp_emb)

    # 7) Plot using same selected_indices ordering
    # prepare labels/podcasts ordering
    labels, podcasts = pd.factorize(df_sel['Podcast'])

    plot_tsne(
        coords=tsne_ass,
        labels=labels,
        podcasts=podcasts,
        selected_indices=list(range(len(df_sel))),
        title="t-SNE of Assumption Embeddings (Selected Windows)",
        out_path=OUT_ASS_PLOT
    )
    plot_tsne(
        coords=tsne_kp,
        labels=labels,
        podcasts=podcasts,
        selected_indices=list(range(len(df_sel))),
        title="t-SNE of Key-Point Embeddings (Selected Windows)",
        out_path=OUT_KP_PLOT
    )

if __name__ == "__main__":
    main()
