import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Load the processed results
with open('results/news_sample_sliding_window_vllm.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten assumptions into a DataFrame
records = []
for entry in data:
    podcast = entry.get("Podcast", "Unknown")
    for assumption in entry.get("Assumptions", []):
        records.append({"Podcast": podcast, "Assumption": assumption})

df = pd.DataFrame(records)

# Embed assumptions
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=os.getenv('HF_HOME'))
embeddings = model.encode(df["Assumption"].tolist(), show_progress_bar=True)

# t-SNE reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

x_vals = tsne_results[:, 0]
y_vals = tsne_results[:, 1]

# Choose a discrete colormap with enough distinct colors
cmap = plt.get_cmap('tab10')
labels, podcasts = pd.factorize(df["Podcast"])
n_categories = len(podcasts)

# Map each label index to a color from the colormap
colors = [cmap(i % cmap.N) for i in range(n_categories)]

plt.figure(figsize=(12,10))
# scatter with our discrete colors
for idx, podcast in enumerate(podcasts):
    mask = (labels == idx)
    plt.scatter(
        x_vals[mask], y_vals[mask],
        c=[colors[idx]],
        label=podcast,
        s=50,
        alpha=0.7
    )

# Expand the axes by 20%
x_min, x_max = x_vals.min(), x_vals.max()
y_min, y_max = y_vals.min(), y_vals.max()
plt.xlim(x_min - 0.2*(x_max-x_min), x_max + 0.2*(x_max-x_min))
plt.ylim(y_min - 0.2*(y_max-y_min), y_max + 0.2*(y_max-y_min))

plt.title("t-SNE of Assumption Embeddings")
plt.xlabel("t-SNE dim 2")
plt.ylabel("t-SNE dim 1")

# Manual legend from the same handles/labels above
plt.legend(title="Podcast", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()

# Save & show
out_path = 'results/assumptions_tsne.png'
plt.savefig(out_path, dpi=300)
print(f"Saved corrected TSNE plot to {out_path}")
plt.show()
