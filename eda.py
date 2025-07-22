import os
import json
import random
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    # ─── Config ────────────────────────────────────────────────────────────────
    INPUT_PATH         = 'results/hosts/ryan_reynolds.json'
    OUTPUT_PLOT        = 'results/plots/ryan_reynolds.png'
    SEED               = 42
    MODEL_NAME         = 'all-MiniLM-L6-v2'
    HF_CACHE           = os.getenv('HF_HOME', None)
    SAMPLE_PER_PODCAST = 1
    random.seed(SEED)

    # ─── Load & Flatten ───────────────────────────────────────────────────────
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        windows = json.load(f)

    records = []
    for w in windows:
        pod = w.get('Podcast', 'Unknown')
        win = w.get('WindowIndex', -1)
        for a_idx, asm in enumerate(w.get('Assumptions', [])):
            records.append({
                'Podcast': pod,
                'WindowIndex': win,
                'AssumptionIndex': a_idx,
                'Text': asm
            })
    df = pd.DataFrame(records)
    print(f"Total individual assumptions: {len(df)}")

    # ─── Embed & t-SNE ─────────────────────────────────────────────────────────
    embedder   = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)
    embeddings = embedder.encode(df['Text'].tolist(), show_progress_bar=True)

    tsne = TSNE(n_components=2, random_state=SEED)
    coords = tsne.fit_transform(embeddings)
    df['x'], df['y'] = coords[:,0], coords[:,1]

    # ─── Sample up to N per podcast ────────────────────────────────────────────
    def sample_n(group):
        return group.sample(n=min(len(group), SAMPLE_PER_PODCAST), random_state=SEED)

    selected_df = (
        df
        .groupby('Podcast', group_keys=False)
        .apply(sample_n)
        .reset_index(drop=True)
    )

    # ─── Plot ──────────────────────────────────────────────────────────────────
    # factorize to get integer codes and unique podcast names
    labels, podcast_names = pd.factorize(df['Podcast'])

    # UPDATED: sample a colormap with as many discrete colors as you have podcasts
    cmap = plt.colormaps['tab20']
    # precompute a color array
    colors = cmap(np.linspace(0, 1, len(podcast_names)))

    fig, ax = plt.subplots(figsize=(24, 18), dpi=300)

    for idx, name in enumerate(podcast_names):
        mask = (labels == idx)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            color=colors[idx],
            label=name if name in selected_df['Podcast'].values else None,
            s=20,
            alpha=0.7
        )

    # 3) Annotate each sampled point
    for _, row in selected_df.iterrows():
        ax.text(
            row['x'],
            row['y'],
            f"{row['WindowIndex']}-{row['AssumptionIndex']}",
            fontsize=8,
            weight='bold',
            va='center',
            ha='center',
            color='black'
        )

    # 4) Tidy up axes
    ax.margins(x=0.1, y=0.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('t-SNE of Individual Assumption Embeddings', fontsize=16)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    # 5) FIGURE-LEVEL legend below the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title='Podcast',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),  # y offset below the axes
        ncol=4,
        fontsize='small',
        title_fontsize='medium',
        frameon=False
    )

    # 6) Make room under & around the axes so the legend isn’t clipped
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.20)

    # 7) Save with tight bounding box so nothing gets cut off
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
