import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    # ─── Config ────────────────────────────────────────────────────────────────
    INPUT_PATH         = 'results/hosts/christopher_kai.json'
    OUTPUT_PLOT        = 'results/plots/christopher_kai.png'
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

    # ─── Sample up to 3 per podcast ─────────────────────────────────────────────
    def sample_three(group):
        n = min(len(group), SAMPLE_PER_PODCAST)
        return group.sample(n=n, random_state=SEED)

    selected_df = (
        df
        .groupby('Podcast', group_keys=False)
        .apply(sample_three)
        .reset_index(drop=True)
    )

    # ─── Plot ──────────────────────────────────────────────────────────────────
    labels, podcast_names = pd.factorize(df['Podcast'])
    cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(16, 12))  # Increase figure size

    # plot all points
    for idx, name in enumerate(podcast_names):
        mask = (labels == idx)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            color=cmap(idx % 20),  # tab20 only has 20 colors, use modulo
            label=name if name in selected_df['Podcast'].values else None,  # only label sampled
            s=20,
            alpha=0.7
        )

    # annotate each sampled point by its WindowIndex-AssumptionIndex
    for _, row in selected_df.iterrows():
        ax.text(
            row['x'],
            row['y'],
            f"{row['WindowIndex']}-{row['AssumptionIndex']}",
            fontsize=8,
            weight='bold',
            va='center', ha='center',
            color='black'
        )

    # pad and equalize aspect
    ax.margins(x=0.1, y=0.1)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title('t-SNE of Individual Assumption Embeddings', fontsize=14)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    # smaller, multi-column legend outside plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title='Podcast',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize='x-small',
        title_fontsize='small',
        ncol=1 if len(labels) < 20 else 2,  # auto-multi-column if many labels
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
