#!/usr/bin/env python3
"""
visualize_quality.py — semantic evaluation of proposition & assumption quality across prompts.
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# ============================================================
# Utility functions
# ============================================================
def flatten_texts(items, key):
    """Extract a list of text strings under a given key, ignoring malformed data."""
    vals = []
    for obj in items:
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], list):
                vals.extend(obj[key])
        elif key in obj:
            vals.extend(obj[key])
    return [str(v).strip() for v in vals if v]


def mean_cosine(a, b):
    """Compute average pairwise cosine similarity between two lists of embeddings."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sim = util.cos_sim(a, b).cpu().numpy()
    return float(np.mean(sim))


# ============================================================
# Semantic metrics
# ============================================================
def compute_metrics_for_turn(model, item):
    """Compute semantic and lexical quality metrics for one turn."""

    exp = item.get("explicit_propositions", [])
    imp = item.get("implicit_propositions", [])
    asm = [a["text"] for a in item.get("assumptions", []) if isinstance(a, dict) and "text" in a]

    if not any([exp, imp, asm]):
        return None

    # Embeddings
    texts = {"exp": exp, "imp": imp, "asm": asm}
    embeds = {k: model.encode(v, convert_to_tensor=True, show_progress_bar=False) if v else [] for k, v in texts.items()}

    # Lexical diversity
    toks = " ".join(asm).lower().split()
    ld = len(set(toks)) / len(toks) if toks else 0.0

    # Semantic redundancy (mean pairwise similarity among assumptions)
    if len(embeds["asm"]) >= 2:
        sr = float(util.cos_sim(embeds["asm"], embeds["asm"]).mean())
    else:
        sr = 0.0

    # Explicit–Implicit Coherence
    eic = mean_cosine(embeds["exp"], embeds["imp"])

    # ✅ FIXED: Concatenate explicit + implicit tensors
    import torch
    combined_props = []
    if len(embeds["exp"]) > 0:
        combined_props.append(embeds["exp"])
    if len(embeds["imp"]) > 0:
        combined_props.append(embeds["imp"])
    if combined_props:
        combined_props = torch.cat(combined_props, dim=0)
    else:
        combined_props = torch.empty(0)

    # Assumption Depth (alignment of assumptions with propositions)
    ad = mean_cosine(embeds["asm"], combined_props)

    # Confidence Consistency
    confs = [float(a.get("confidence", 0)) for a in item.get("assumptions", []) if isinstance(a, dict)]
    cc = np.std(confs) if confs else 0.0

    return {
        "lexical_diversity": ld,
        "semantic_redundancy": sr,
        "explicit_implicit_coherence": eic,
        "assumption_depth": ad,
        "confidence_consistency": cc,
    }

# ============================================================
# Evaluate across all prompt outputs
# ============================================================
def evaluate_prompt_dir(base_dir="results/prompt_camprison"):
    base = Path(base_dir)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_rows = []

    prompt_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("prompt")])
    for prompt_path in prompt_dirs:
        prompt_name = prompt_path.name
        files = sorted(prompt_path.glob("*.json"))
        print(f"🔍 Evaluating {prompt_name} ({len(files)} files)...")

        for fpath in tqdm(files, desc=prompt_name):
            try:
                data = json.load(open(fpath, "r", encoding="utf-8"))
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]

            for item in items:
                m = compute_metrics_for_turn(model, item)
                if m:
                    m["prompt"] = prompt_name
                    all_rows.append(m)

    return pd.DataFrame(all_rows)


# ============================================================
# Visualization
# ============================================================
def visualize_metrics(df, output_dir="results/quality_charts"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("⚠️ No data to visualize.")
        return

    sns.set(style="whitegrid", font_scale=1.1)

    metric_names = [
        "lexical_diversity",
        "semantic_redundancy",
        "explicit_implicit_coherence",
        "assumption_depth",
        "confidence_consistency",
    ]

    # Boxplots for each metric
    for metric in metric_names:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="prompt", y=metric, data=df)
        plt.title(f"{metric.replace('_', ' ').title()} per Prompt")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_per_prompt.png")
        plt.close()

    # Summary table
    summary = df.groupby("prompt")[metric_names].mean().reset_index()
    summary.to_csv(f"{output_dir}/semantic_quality_summary.csv", index=False)
    print(f"📊 Saved summary and charts to {output_dir}/")


# ============================================================
# Main entry
# ============================================================
def main():
    latest_dir = max(
        [p for p in Path("results").glob("prompt*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    print(f"📂 Using latest prompt directory: {latest_dir}")

    df = evaluate_prompt_dir(str(latest_dir))
    print(f"✅ Computed semantic metrics for {len(df)} turns.")

    visualize_metrics(df)


if __name__ == "__main__":
    main()
