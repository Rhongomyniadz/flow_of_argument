import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_tsne(df, coords_col_x, coords_col_y, selected, title, out_path):
    """
    Plots a t-SNE scatter of all windows in df, color-coded by Podcast,
    and annotates the sampled windows by their WindowIndex.
    """
    plt.figure(figsize=(12, 10))
    labels, podcasts = pd.factorize(df['Podcast'])
    cmap = plt.get_cmap('tab10')

    # plot all points
    for idx, podcast in enumerate(podcasts):
        mask = labels == idx
        plt.scatter(
            df.loc[mask, coords_col_x],
            df.loc[mask, coords_col_y],
            c=[cmap(idx)],
            label=podcast,
            s=40,
            alpha=0.5
        )

    # annotate sampled windows
    for w in selected:
        row = df[(df['Podcast'] == w['Podcast']) & (df['WindowIndex'] == w['WindowIndex'])].iloc[0]
        x, y = row[coords_col_x], row[coords_col_y]
        plt.annotate(
            str(row['WindowIndex']),
            (x, y),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=9,
            weight='bold',
            color='black'
        )

    plt.title(title)
    plt.xlabel(coords_col_x)
    plt.ylabel(coords_col_y)
    plt.legend(title="Podcast", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    plt.close()

def main():
    # --- CONFIG ---
    INPUT_PATH     = 'results/news_sample_sliding_window_vllm.json'
    SELECTED_JSON  = 'results/selected_windows.json'
    OUT_ASS_PLOT   = 'results/assumptions_tsne_labeled.png'
    OUT_KP_PLOT    = 'results/keypoints_tsne_labeled.png'
    SEED           = 42
    MODEL_NAME     = 'all-MiniLM-L6-v2'
    HF_CACHE       = os.getenv('HF_HOME', None)
    MAX_SAMPLES    = 3
    RANDOM_STATE   = 42

    # ensure results dir
    os.makedirs('results', exist_ok=True)
    random.seed(SEED)

    # 1) Load and filter windows
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_windows = json.load(f)

    filtered = [
        w for w in all_windows
        if w.get("KeyPoints") and len(w["KeyPoints"]) > 0
        and w.get("Assumptions") and len(w["Assumptions"]) > 0
    ]
    print(f"Kept {len(filtered)} windows after filtering non-empty KeyPoints & Assumptions.")

    # 2) Sample up to MAX_SAMPLES windows per podcast
    by_podcast = defaultdict(list)
    for w in filtered:
        by_podcast[w['Podcast']].append(w)

    selected = []
    for podcast, wins in by_podcast.items():
        if len(wins) <= MAX_SAMPLES:
            sel = wins
        else:
            sel = random.sample(wins, MAX_SAMPLES)
        selected.extend(sel)

    # 3) Save the selected windows
    with open(SELECTED_JSON, 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2)
    print(f"Saved {len(selected)} sampled windows to {SELECTED_JSON}")

    # 4) Build DataFrame for embedding & plotting
    df = pd.DataFrame(filtered)
    # join lists into one text per window
    df['AssumpText'] = df['Assumptions'].apply(lambda lst: ' '.join(lst))
    df['KPText']     = df['KeyPoints'].apply(lambda lst: ' '.join(lst))

    # 5) Initialize embedder
    embedder = SentenceTransformer(
        MODEL_NAME,
        cache_folder=HF_CACHE
    )

    # 6a) Embed & TSNE assumptions
    ass_emb = embedder.encode(df['AssumpText'].tolist(), show_progress_bar=True)
    tsne_ass = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(ass_emb)
    df['TSNE_ASS_X'], df['TSNE_ASS_Y'] = tsne_ass[:, 0], tsne_ass[:, 1]

    # 6b) Embed & TSNE key points
    kp_emb = embedder.encode(df['KPText'].tolist(), show_progress_bar=True)
    tsne_kp = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(kp_emb)
    df['TSNE_KP_X'], df['TSNE_KP_Y'] = tsne_kp[:, 0], tsne_kp[:, 1]

    # 7) Plot
    plot_tsne(
        df,
        coords_col_x='TSNE_ASS_X',
        coords_col_y='TSNE_ASS_Y',
        selected=selected,
        title="t-SNE of Assumption Embeddings (Labeled)",
        out_path=OUT_ASS_PLOT
    )
    plot_tsne(
        df,
        coords_col_x='TSNE_KP_X',
        coords_col_y='TSNE_KP_Y',
        selected=selected,
        title="t-SNE of Key-Point Embeddings (Labeled)",
        out_path=OUT_KP_PLOT
    )

if __name__ == "__main__":
    main()
