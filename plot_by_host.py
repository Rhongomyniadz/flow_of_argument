import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def main():
    # ─── Config ────────────────────────────────────────────────────────────────
    INPUT_FILES = [
        ('Ryan Reynolds',      'results/hosts/ryan_reynolds.json'),
        ('Adam Torres',        'results/hosts/adam_torres.json'),
        ('John Doe',           'results/hosts/john_doe.json'),
        ('Marshall Poe',       'results/hosts/marshall_poe.json'),
        ('Christopher Kai',    'results/hosts/christopher_kai.json'),
    ]
    OUTPUT_PLOT     = 'results/plots/five_hosts_tsne.png'
    SEED            = 42
    MODEL_NAME      = 'all-MiniLM-L6-v2'
    HF_CACHE        = os.getenv('HF_HOME', None)
    random.seed(SEED)

    # ─── Load & Flatten all hosts ─────────────────────────────────────────────
    records = []
    for host, path in INPUT_FILES:
        with open(path, 'r', encoding='utf-8') as f:
            windows = json.load(f)
        for w in windows:
            win = w.get('WindowIndex', -1)
            for a_idx, asm in enumerate(w.get('Assumptions', [])):
                records.append({
                    'Host': host,
                    'WindowIndex': win,
                    'AssumptionIndex': a_idx,
                    'Text': asm
                })

    df = pd.DataFrame(records)
    print(f"Total individual assumptions across all hosts: {len(df)}")

    # ─── Embed & t-SNE ─────────────────────────────────────────────────────────
    embedder   = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)
    embeddings = embedder.encode(df['Text'].tolist(), show_progress_bar=True)

    tsne       = TSNE(n_components=2, random_state=SEED)
    coords     = tsne.fit_transform(embeddings)
    df['x'], df['y'] = coords[:,0], coords[:,1]

    # ─── Plot (no sampling) ────────────────────────────────────────────────────
    labels, host_names = pd.factorize(df['Host'])
    cmap   = plt.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, len(host_names)))

    fig, ax = plt.subplots(figsize=(20, 16), dpi=200)

    # scatter all points, labeling each host once
    for idx, host in enumerate(host_names):
        mask = (labels == idx)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            color=colors[idx],
            label=host,
            s=25,
            alpha=0.7,
        )

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('t-SNE of Individual Assumption Embeddings by Host', fontsize=18)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.margins(0.1)

    # figure‐level legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title='Host',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=5,
        fontsize='small',
        title_fontsize='medium',
        frameon=False
    )

    # make room for legend
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.17)

    # save tightly so nothing is clipped
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches='tight')
    print(f"Saved combined‑hosts plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
