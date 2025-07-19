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
    OUTPUT_PLOT        = 'results/hosts/ryan_reynolds.png'
    SEED               = 42
    MODEL_NAME         = 'all-MiniLM-L6-v2'
    HF_CACHE           = os.getenv('HF_HOME', None)
    SAMPLE_PER_PODCAST = 3
    random.seed(SEED)

    # ─── Load & Flatten ───────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
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

    fig, ax = plt.subplots(figsize=(20, 10))

    # plot all points
    for idx, name in enumerate(podcast_names):
        mask = (labels == idx)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            color=cmap(idx),
            label=name,
            s=30,
            alpha=0.5
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

    ax.set_title('t-SNE of Individual Assumption Embeddings')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(
        title='Podcast',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize='small'
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
