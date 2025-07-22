import os
import json
import random
import pandas as pd
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

    # ─── Sample up to 1 per podcast ────────────────────────────────────────────
    def sample_one(group):
        return group.sample(n=min(len(group), SAMPLE_PER_PODCAST), random_state=SEED)

    selected_df = (
        df
        .groupby('Podcast', group_keys=False)
        .apply(sample_one)
        .reset_index(drop=True)
    )

    # ─── Plot ──────────────────────────────────────────────────────────────────
    labels, podcast_names = pd.factorize(df['Podcast'])
    cmap = plt.get_cmap('tab20')

    # 1) Large canvas
    fig, ax = plt.subplots(figsize=(24, 18), dpi=300)

    # 2) Draw points, only label the sampled ones
    for idx, name in enumerate(podcast_names):
        mask = (labels == idx)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            color=cmap(idx % 20),
            label=name if name in selected_df['Podcast'].values else None,
            s=20,
            alpha=0.7
        )

    # 3) Annotate samples
    for _, row in selected_df.iterrows():
        ax.text(
            row['x'], row['y'],
            f"{row['WindowIndex']}-{row['AssumptionIndex']}",
            fontsize=8, weight='bold',
            va='center', ha='center',
            color='black'
        )

    # 4) Axes styling
    ax.margins(0.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('t-SNE of Individual Assumption Embeddings', fontsize=16)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    # 5) Put legend above the plot in 8 columns
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        title='Podcast',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.20),   # 20% above the top of the axes
        ncol=8,                       # eight columns
        fontsize='xx-small',
        title_fontsize='small',
        frameon=False
    )

    # 6) Make room for the legend at the top
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # leave top 15% for legend

    # 7) Save
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
