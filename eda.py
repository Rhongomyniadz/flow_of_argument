import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# 1) Load the processed results
with open('results/news_sample_sliding_window_vllm.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2) Flatten assumptions into a DataFrame
records = []
for entry in data:
    podcast = entry.get("Podcast", "Unknown")
    for assumption in entry.get("Assumptions", []):
        records.append({"Podcast": podcast, "Assumption": assumption})

df = pd.DataFrame(records)

# Basic EDA
print("Total assumptions:", len(df))
print("Assumptions per podcast:")
print(df["Podcast"].value_counts(), "\n")

# 3) Embed each assumption
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=os.getenv('HF_HOME'))
embeddings = model.encode(df["Assumption"].tolist(), show_progress_bar=True)

# 4) t-SNE reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# 5) Scatter plot
plt.figure(figsize=(8, 6))
labels, uniques = pd.factorize(df["Podcast"])
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1],
    c=labels, alpha=0.7
)
plt.title("t-SNE of Podcast-Assumption Embeddings")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")

# Legend
handles, _ = scatter.legend_elements()
plt.legend(handles, uniques, title="Podcast", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save to file
plot_path = 'results/assumptions_tsne.png'
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")
