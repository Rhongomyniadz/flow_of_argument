import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_tsne(coords, labels, podcasts, title, out_path, annotate=None):
    """
    coords: numpy array of shape (n_points, 2)
    labels: array of length n_points (int label per point)
    podcasts: array of podcast names by label index
    annotate: list of (point_idx, text) to annotate specific points
    """
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')

    # Plot all points per podcast
    for idx, podcast in enumerate(podcasts):
        mask = (labels == idx)
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(idx)], label=podcast,
            s=40, alpha=0.6
        )

    # Optional annotation
    if annotate:
        for pt_idx, text in annotate:
            x, y = coords[pt_idx]
            plt.annotate(
                text,
                (x, y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=8,
                weight='bold'
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
    ASS_PLOT      = 'results/assumptions_individual_tsne.png'
    KP_PLOT       = 'results/keypoints_individual_tsne.png'
    SEED          = 42
    MODEL_NAME    = 'all-MiniLM-L6-v2'
    HF_CACHE      = os.getenv('HF_HOME', None)
    RANDOM_STATE  = 42

    os.makedirs('results', exist_ok=True)
    random.seed(SEED)

    # Load filtered windows
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        windows = json.load(f)

    # Filter non-empty
    windows = [w for w in windows if w.get('Assumptions') or w.get('KeyPoints')]

    # Flatten assumptions
    ass_records = []  # each item: dict with Podcast, WindowIndex, Assumption
    for w in windows:
        for asm in w.get('Assumptions', []):
            ass_records.append({'Podcast': w['Podcast'],
                                'WindowIndex': w['WindowIndex'],
                                'Text': asm})
    df_ass = pd.DataFrame(ass_records)
    print(f"Total individual assumptions: {len(df_ass)}")

    # Flatten key points
    kp_records = []
    for w in windows:
        for kp in w.get('KeyPoints', []):
            kp_records.append({'Podcast': w['Podcast'],
                               'WindowIndex': w['WindowIndex'],
                               'Text': kp})
    df_kp = pd.DataFrame(kp_records)
    print(f"Total individual key points: {len(df_kp)}")

    # Initialize embedder
    embedder = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)

    # Embed and TSNE assumptions
    ass_emb = embedder.encode(df_ass['Text'].tolist(), show_progress_bar=True)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    ass_coords = tsne.fit_transform(ass_emb)

    # Factor label podcasts
    ass_labels, ass_podcasts = pd.factorize(df_ass['Podcast'])
    plot_tsne(
        ass_coords, ass_labels, ass_podcasts,
        "t-SNE of Individual Assumption Embeddings",
        ASS_PLOT
    )

    # Embed and TSNE key points
    kp_emb = embedder.encode(df_kp['Text'].tolist(), show_progress_bar=True)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    kp_coords = tsne.fit_transform(kp_emb)

    kp_labels, kp_podcasts = pd.factorize(df_kp['Podcast'])
    plot_tsne(
        kp_coords, kp_labels, kp_podcasts,
        "t-SNE of Individual Key-Point Embeddings",
        KP_PLOT
    )

if __name__ == '__main__':
    main()
