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

# Swap axes: dim2→x, dim1→y
x_vals = tsne_results[:, 1]
y_vals = tsne_results[:, 0]

# Plot with larger figure
plt.figure(figsize=(12, 10))
labels, uniques = pd.factorize(df["Podcast"])
scatter = plt.scatter(x_vals, y_vals, c=labels, alpha=0.7, s=50)

# Compute margins (20% extra)
x_min, x_max = x_vals.min(), x_vals.max()
y_min, y_max = y_vals.min(), y_vals.max()
x_margin = (x_max - x_min) * 0.2
y_margin = (y_max - y_min) * 0.2
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

plt.title("t-SNE of Assumption Embeddings (Axes Swapped & Scaled)")
plt.xlabel("t-SNE dim 2")
plt.ylabel("t-SNE dim 1")

# Legend
handles, _ = scatter.legend_elements(prop="sizes", num=None)
plt.legend(handles, uniques, title="Podcast", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Save to file
plot_path = 'results/assumptions_tsne.png'
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")

plt.show()
