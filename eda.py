#!/usr/bin/env python3
import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    # Configuration
    INPUT_PATH = 'results/politics.json'
    OUTPUT_PLOT = 'results/individual_assumptions_tsne.png'
    SEED = 42
    MODEL_NAME = 'all-MiniLM-L6-v2'
    HF_CACHE = os.getenv('HF_HOME', None)

    # Ensure output directory
    os.makedirs('results', exist_ok=True)
    random.seed(SEED)

    # Load all windows
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        windows = json.load(f)

    # Flatten each individual assumption into its own record
    records = []
    for w in windows:
        podcast = w.get('Podcast', 'Unknown')
        win_idx = w.get('WindowIndex', -1)
        for asm_idx, asm in enumerate(w.get('Assumptions', [])):
            records.append({
                'Podcast': podcast,
                'WindowIndex': win_idx,
                'AssumptionIndex': asm_idx,
                'Text': asm
            })
    df = pd.DataFrame(records)
    print(f"Total individual assumptions: {len(df)}")

    # Initialize encoder
    embedder = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)

    # Encode each assumption separately
    texts = df['Text'].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Run t-SNE on individual embeddings
    tsne = TSNE(n_components=2, random_state=SEED)
    coords = tsne.fit_transform(embeddings)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]

    # Plot each point, colored by podcast
    labels, podcast_names = pd.factorize(df['Podcast'])
    cmap = plt.get_cmap('tab20')  # up to 20 distinct colors

    plt.figure(figsize=(12, 10))
    for idx, name in enumerate(podcast_names):
        mask = (labels == idx)
        plt.scatter(
            df.loc[mask, 'x'], df.loc[mask, 'y'],
            c=[cmap(idx)], label=name,
            s=40, alpha=0.6
        )

    # Expand x-axis limits for better spread
    x_min, x_max = df['x'].min(), df['x'].max()
    x_margin = (x_max - x_min) * 0.1  # 10% margin
    plt.xlim(x_min - x_margin, x_max + x_margin)

    # Optional: similarly expand y-axis if needed
    y_min, y_max = df['y'].min(), df['y'].max()
    y_margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.title('t-SNE of Individual Assumption Embeddings')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.legend(title='Podcast', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()

    # Save the plot
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
