import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    # ─── Configuration ─────────────────────────────────────────────────────────
    POLITICS_JSON = 'results/politics.json'
    RANDOM_JSON   = 'results/random_assumptions.json'
    OUTPUT_PLOT   = 'results/combined_assumptions_tsne.png'
    MODEL_NAME    = 'all-MiniLM-L6-v2'
    RANDOM_STATE  = 42

    os.makedirs('results', exist_ok=True)

    # ─── Load & flatten the ORIGINAL politics.json ─────────────────────────────
    with open(POLITICS_JSON, 'r', encoding='utf-8') as f:
        orig_windows = json.load(f)

    orig_records = []
    for w in orig_windows:
        podcast = w.get('Podcast', 'Unknown')
        wi = w.get('WindowIndex', -1)
        for ai, txt in enumerate(w.get('Assumptions', [])):
            orig_records.append({
                'Dataset': 'Original',
                'Podcast': podcast,
                'WindowIndex': wi,
                'AssumptionIndex': ai,
                'Text': txt
            })

    # ─── Load & flatten the RANDOM assumptions ─────────────────────────────────
    with open(RANDOM_JSON, 'r', encoding='utf-8') as f:
        rand_windows = json.load(f)

    rand_records = []
    for w in rand_windows:
        podcast = w.get('Podcast', 'Unknown')
        wi = w.get('WindowIndex', -1)
        for ai, txt in enumerate(w.get('Assumptions', [])):
            rand_records.append({
                'Dataset': 'Random',
                'Podcast': podcast,
                'WindowIndex': wi,
                'AssumptionIndex': ai,
                'Text': txt
            })

    # ─── Combine into one DataFrame ────────────────────────────────────────────
    df = pd.DataFrame(orig_records + rand_records)
    print(f"Total points (orig + random): {len(df)}")

    # ─── Embed & run t-SNE ─────────────────────────────────────────────────────
    embedder = SentenceTransformer(MODEL_NAME,
                                   cache_folder=os.getenv('HF_HOME', None))
    embeddings = embedder.encode(df['Text'].tolist(),
                                 show_progress_bar=True)
    coords = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)
    df['x'], df['y'] = coords[:, 0], coords[:, 1]

    # ─── Plot ──────────────────────────────────────────────────────────────────
    podcasts = df['Podcast'].unique()
    cmap     = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(20, 10))

    # plot originals as circles
    for i, pod in enumerate(podcasts):
        mask = (df['Dataset']=='Original') & (df['Podcast']==pod)
        if not mask.any(): 
            continue
        ax.scatter(df.loc[mask, 'x'],
                   df.loc[mask, 'y'],
                   marker='o',
                   color=cmap(i),
                   label=pod,
                   s=30,
                   alpha=0.6)

    # plot randoms as crosses
    for i, pod in enumerate(podcasts):
        mask = (df['Dataset']=='Random') & (df['Podcast']==pod)
        if not mask.any():
            continue
        ax.scatter(df.loc[mask, 'x'],
                   df.loc[mask, 'y'],
                   marker='x',
                   color=cmap(i),
                   s=60,
                   alpha=0.8)

    # pad and equalize
    ax.margins(0.1)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title('t-SNE of Original (o) vs Random (×) Assumptions', fontsize=16)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    ax.legend(title='Podcast', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved combined plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
