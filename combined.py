import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    # ─── Config ────────────────────────────────────────────────────────────────
    POLITICS_JSON = 'results/politics.json'
    RANDOM_JSON   = 'results/random_assumptions.json'
    OUTPUT_PLOT   = 'results/combined_assumptions_tsne.png'
    MODEL_NAME    = 'all-MiniLM-L6-v2'
    RANDOM_STATE  = 42

    os.makedirs('results', exist_ok=True)

    # ─── Load & flatten ORIGINAL ───────────────────────────────────────────────
    with open(POLITICS_JSON, 'r', encoding='utf-8') as f:
        orig = json.load(f)
    orig_recs = []
    for w in orig:
        pod = w['Podcast']
        wi  = w.get('WindowIndex', -1)
        for ai, t in enumerate(w['Assumptions']):
            orig_recs.append({
                'Dataset': 'Original',
                'Podcast': pod,
                'WindowIndex': wi,
                'AssumptionIndex': ai,
                'Text': t
            })

    # ─── Load & flatten RANDOM ─────────────────────────────────────────────────
    with open(RANDOM_JSON, 'r', encoding='utf-8') as f:
        rand = json.load(f)
    rand_recs = []
    for w in rand:
        pod = w['Podcast']
        wi  = w['WindowIndex']
        for ai, t in enumerate(w['Assumptions']):
            rand_recs.append({
                'Dataset': 'Random',
                'Podcast': pod,
                'WindowIndex': wi,
                'AssumptionIndex': ai,
                'Text': t
            })

    # ─── Combine & embed + t-SNE ───────────────────────────────────────────────
    df = pd.DataFrame(orig_recs + rand_recs)
    print(f"Total points: {len(df)}")

    embedder   = SentenceTransformer(MODEL_NAME,
                                     cache_folder=os.getenv('HF_HOME', None))
    embeddings = embedder.encode(df['Text'].tolist(),
                                 show_progress_bar=True)
    coords     = TSNE(n_components=2,
                      random_state=RANDOM_STATE).fit_transform(embeddings)
    df['x'], df['y'] = coords[:,0], coords[:,1]

    # ─── Plot ──────────────────────────────────────────────────────────────────
    # first, draw all ORIGINAL in one pass
    podcasts = df['Podcast'].unique()
    cmap     = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, pod in enumerate(podcasts):
        mask = (df['Dataset']=='Original') & (df['Podcast']==pod)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            marker='o',
            color=cmap(i),
            label=pod,
            s=40,
            alpha=0.8
        )

    # now overlay RANDOM as unfilled rings
    random_df = df[df['Dataset']=='Random']
    pos_mask  = random_df['WindowIndex'] % 2 == 0  # even → “positive”
    neg_mask  = random_df['WindowIndex'] % 2 == 1  # odd  → “negative”

    # lime rings for positives
    ax.scatter(
        random_df.loc[pos_mask, 'x'],
        random_df.loc[pos_mask, 'y'],
        marker='o',
        facecolors='none',
        edgecolors='lime',
        linewidths=2,
        s=120,
        label='Random Positive'
    )
    # magenta rings for negatives
    ax.scatter(
        random_df.loc[neg_mask, 'x'],
        random_df.loc[neg_mask, 'y'],
        marker='o',
        facecolors='none',
        edgecolors='magenta',
        linewidths=2,
        s=120,
        label='Random Negative'
    )

    # ─── Final formatting ─────────────────────────────────────────────────────
    ax.margins(0.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('t-SNE of Original (solid) & Random (rings) Assumptions', fontsize=16)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(title='Legend', bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved combined plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
