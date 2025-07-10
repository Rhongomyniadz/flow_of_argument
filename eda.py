import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    # Configuration
    INPUT_PATH   = 'results/politics.json'
    OUTPUT_PLOT  = 'results/individual_assumptions_tsne.png'
    SEED         = 42
    MODEL_NAME   = 'all-MiniLM-L6-v2'
    HF_CACHE     = os.getenv('HF_HOME', None)

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

    # Initialize encoder & embed
    embedder   = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE)
    texts      = df['Text'].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Run t-SNE
    tsne   = TSNE(n_components=2, random_state=SEED)
    coords = tsne.fit_transform(embeddings)
    df['x'], df['y'] = coords[:, 0], coords[:, 1]

    # Prepare coloring
    labels, podcast_names = pd.factorize(df['Podcast'])
    cmap = plt.get_cmap('tab20')  # up to 20 distinct colors

    # Plot each point, colored by podcast, on a wide canvas
    fig, ax = plt.subplots(figsize=(20, 10))
    for idx, name in enumerate(podcast_names):
        mask = (labels == idx)
        ax.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            color=cmap(idx),
            label=name,
            s=40,
            alpha=0.6
        )

    # Automatically add a 10% data margin on each axis
    ax.margins(x=0.1, y=0.1)

    # Keep 1:1 data-unit aspect ratio so X and Y scales aren’t skewed
    ax.set_aspect('equal', adjustable='box')

    # Labels, legend, layout
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

    # Save
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
