import os
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

# 0) Setup
os.makedirs('results', exist_ok=True)
random.seed(42)

# 1) Load all windows
with open('results/news_sample_sliding_window_vllm.json', 'r', encoding='utf-8') as f:
    all_windows = json.load(f)

# 2) Filter to those with both KeyPoints and Assumptions non-empty
filtered = [
    w for w in all_windows
    if w.get("KeyPoints") and len(w["KeyPoints"]) > 0
    and w.get("Assumptions") and len(w["Assumptions"]) > 0
]
print(f"Kept {len(filtered)} windows out of {len(all_windows)} after filtering non‐empty fields.")

# 3) Sample up to 3 windows per podcast from the filtered set
by_podcast = defaultdict(list)
for w in filtered:
    by_podcast[w['Podcast']].append(w)

selected = []
for podcast, wins in by_podcast.items():
    sel = wins if len(wins) <= 3 else random.sample(wins, 3)
    selected.extend(sel)

# 4) Save the selected windows
with open('results/selected_windows.json', 'w', encoding='utf-8') as f:
    json.dump(selected, f, indent=2)
print(f"Saved {len(selected)} sampled windows to results/selected_windows.json")

# 5) Prepare DataFrame from the filtered windows for embedding
df = pd.DataFrame(filtered)
df['CombinedText'] = df['WindowText']

# 6) Embed each window’s combined text
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=os.getenv('HF_HOME'))
embeddings = model.encode(df['CombinedText'].tolist(), show_progress_bar=True)

# 7) Run t-SNE
tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(embeddings)
df['TSNE_1'] = coords[:, 0]
df['TSNE_2'] = coords[:, 1]

# 8) Plot all filtered windows, label the selected ones
plt.figure(figsize=(12,10))
pod_labels, podcasts = pd.factorize(df['Podcast'])
cmap = plt.get_cmap('tab10')

# background scatter
for idx, podcast in enumerate(podcasts):
    mask = (pod_labels == idx)
    plt.scatter(
        df.loc[mask, 'TSNE_2'],
        df.loc[mask, 'TSNE_1'],
        c=[cmap(idx)], label=podcast,
        s=40, alpha=0.5
    )

# annotate only the selected
for win in selected:
    row = df[(df['Podcast']==win['Podcast']) & (df['WindowIndex']==win['WindowIndex'])].iloc[0]
    x, y = row['TSNE_2'], row['TSNE_1']
    plt.annotate(str(row['WindowIndex']), (x, y),
                 textcoords="offset points", xytext=(3,3),
                 fontsize=10, weight='bold', color='black')

plt.title("t-SNE of Filtered Context Windows (Non-Empty KP & Assumptions)")
plt.xlabel("t-SNE dim 2")
plt.ylabel("t-SNE dim 1")
plt.legend(title="Podcast", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()

out_path = 'results/windows_tsne_labeled_filtered.png'
plt.savefig(out_path, dpi=300)
print(f"Saved filtered & labeled t-SNE plot to {out_path}")
plt.show()
