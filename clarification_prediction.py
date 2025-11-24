import json
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------
# Regex: clarification questions
# -----------------------------
CLARIFICATION_PATTERNS = [
    r"\b(could|can|would)\s+you\s+(please\s+)?(explain|clarify|elaborate|unpack|expand|walk me through|say more about|tell me more about)\b",
    r"\bwhat\s+do\s+you\s+mean\b",
    r"\bwhen\s+you\s+say\s+that\b",
    r"\bdo\s+you\s+mean\b",
    r"\bi'm\s+not\s+sure\s+i\s+follow\b",
    r"\bhelp\s+me\s+understand\b",
    r"\bcan\s+you\s+go\s+over\s+that\b",
    r"\bcan\s+you\s+repeat\s+that\b",
    r"\bso\s+you're\s+saying\b",
    r"\bjust\s+to\s+be\s+clear\b",
]

CLARIFICATION_RE = re.compile("|".join(CLARIFICATION_PATTERNS), re.IGNORECASE)


def is_clarification_question(text: str) -> bool:
    """Return True if text contains a clarification-style question."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    return bool(CLARIFICATION_RE.search(t))


# -----------------------------
# Loading prompt outputs
# -----------------------------
def load_prompt_dirs(base_dir: str) -> List[Path]:
    base = Path(base_dir)
    prompt_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("prompt")],
        key=lambda p: p.name,
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt directories found under {base_dir}")
    return prompt_dirs


def load_episode(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return [data]


# -----------------------------
# Build A->B pairs
# -----------------------------
def build_pairs(base_dir: str = "results/prompt_camprison") -> pd.DataFrame:
    prompt_dirs = load_prompt_dirs(base_dir)
    rows = []

    for pdir in prompt_dirs:
        prompt_name = pdir.name
        files = sorted(pdir.glob("*.json"))
        print(f"🔍 Scanning {prompt_name} ({len(files)} episodes)...")

        for fpath in files:
            episode_key = fpath.stem
            turns = load_episode(fpath)
            if len(turns) < 2:
                continue

            # Walk adjacent pairs
            for i in range(len(turns) - 1):
                a = turns[i]
                b = turns[i + 1]

                spk_a = a.get("speaker_id")
                spk_b = b.get("speaker_id")

                # Only consider speaker-change pairs A->B
                if spk_a is None or spk_b is None or spk_a == spk_b:
                    continue

                assumptions_a = a.get("assumptions", [])
                if not isinstance(assumptions_a, list):
                    assumptions_a = []

                num_assumptions_a = len(assumptions_a)

                b_text = b.get("turn_text", "")
                b_is_clarif = is_clarification_question(b_text)

                rows.append({
                    "prompt": prompt_name,
                    "episode_key": episode_key,
                    "turn_num_a": i + 1,        # 1-indexed
                    "turn_num_b": i + 2,
                    "speaker_a": spk_a,
                    "speaker_b": spk_b,
                    "num_assumptions_a": num_assumptions_a,
                    "b_is_clarification": int(b_is_clarif),
                    "turn_text_a": a.get("turn_text", ""),
                    "turn_text_b": b_text,
                })

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Analysis + Plotting
# -----------------------------
def plot_probability(df_pairs: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    if df_pairs.empty:
        print("⚠️ No valid A->B pairs found.")
        return

    # Cap very large counts (shouldn't happen with your prompts, but safe)
    df_pairs["num_assumptions_a_capped"] = df_pairs["num_assumptions_a"].clip(upper=10)

    # ---- Overall empirical probability by count ----
    overall = (
        df_pairs
        .groupby("num_assumptions_a_capped")["b_is_clarification"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "prob"})
    )

    # ---- By prompt ----
    by_prompt = (
        df_pairs
        .groupby(["prompt", "num_assumptions_a_capped"])["b_is_clarification"]
        .mean()
        .reset_index()
        .rename(columns={"b_is_clarification": "prob"})
    )

    # Save tables
    df_pairs.to_csv(outdir / "pairs.csv", index=False)
    overall.to_csv(outdir / "binned_overall.csv", index=False)
    by_prompt.to_csv(outdir / "binned_by_prompt.csv", index=False)

    # Plot settings
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    # ---- Plot 1: overall probability ----
    plt.figure(figsize=(8.5, 5))
    sns.pointplot(
        data=df_pairs,
        x="num_assumptions_a_capped",
        y="b_is_clarification",
        estimator=np.mean,
        errorbar="se",
        color="tab:blue",
    )
    plt.title("P(next turn is clarification | # assumptions in A)")
    plt.xlabel("# assumptions in Speaker A (capped at 10)")
    plt.ylabel("Probability Speaker B clarifies next turn")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / "prob_next_clarification_overall.png", dpi=200)
    plt.close()

    # ---- Plot 2: per prompt ----
    plt.figure(figsize=(9.5, 5.5))
    sns.pointplot(
        data=df_pairs,
        x="num_assumptions_a_capped",
        y="b_is_clarification",
        hue="prompt",
        estimator=np.mean,
        errorbar="se",
        dodge=0.4,
    )
    plt.title("P(next clarification | # assumptions in A) by Prompt")
    plt.xlabel("# assumptions in Speaker A (capped at 10)")
    plt.ylabel("Probability Speaker B clarifies next turn")
    plt.ylim(0, 1)
    plt.legend(title="prompt", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "prob_next_clarification_by_prompt.png", dpi=200)
    plt.close()

    # ---- Basic numeric diagnostics ----
    # Point-biserial (Pearson with binary)
    pearson = df_pairs["num_assumptions_a"].corr(df_pairs["b_is_clarification"], method="pearson")
    spearman = df_pairs["num_assumptions_a"].corr(df_pairs["b_is_clarification"], method="spearman")
    print("\n=== Diagnostics ===")
    print(f"Total A->B pairs: {len(df_pairs)}")
    print(f"Clarification rate overall: {df_pairs['b_is_clarification'].mean():.3f}")
    print(f"Pearson (point-biserial) corr: {pearson:.3f}")
    print(f"Spearman corr: {spearman:.3f}")
    print("Binned counts:")
    print(overall[["num_assumptions_a_capped", "count", "prob"]].to_string(index=False))


def main():
    base_dir = "results/prompt_camprison"
    outdir = Path("results/analysis_charts/clarification_prediction")

    df_pairs = build_pairs(base_dir)
    plot_probability(df_pairs, outdir)


if __name__ == "__main__":
    main()