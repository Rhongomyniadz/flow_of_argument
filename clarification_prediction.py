import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

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
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    return bool(CLARIFICATION_RE.search(t))


# -----------------------------
# Loading prompt outputs (NEW)
# -----------------------------
def load_prompt_dirs(base_dir: str) -> List[Path]:
    """
    Works with:
      - output_root containing prompt3/ and raw/ (new pipeline)
      - a directory that itself IS prompt3/ (contains episode json files)
      - legacy layout with prompt1..prompt5 (still supported)
    """
    base = Path(base_dir)

    # Case 1: base_dir is already a prompt dir (contains episode JSONs)
    jsons_here = list(base.glob("*.json"))
    if jsons_here:
        return [base]

    # Case 2: output root contains prompt* dirs (new + legacy)
    prompt_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("prompt")],
        key=lambda p: p.name,
    )

    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt directories (prompt*) or episode JSONs found under {base_dir}")

    return prompt_dirs


def load_episode(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return [data]


def get_episode_id_from_file(turns: List[Dict[str, Any]], fpath: Path) -> Any:
    # Prefer explicit field in JSON (new pipeline stores episode_id per record)
    if turns and isinstance(turns[0], dict) and "episode_id" in turns[0]:
        return turns[0].get("episode_id")
    # Fall back to filename stem (new pipeline uses <id>.json)
    stem = fpath.stem
    if stem.isdigit():
        return int(stem)
    return stem


def sort_turns_in_episode(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    New pipeline outputs already in order, but this makes it robust.
    Uses turn_idx if present, otherwise keeps original order.
    """
    if not turns:
        return turns
    if all(isinstance(t, dict) and "turn_idx" in t for t in turns):
        try:
            return sorted(turns, key=lambda x: (x.get("turn_idx") is None, x.get("turn_idx")))
        except Exception:
            return turns
    return turns


# -----------------------------
# Build A->B pairs (per episode)
# -----------------------------
def build_pairs(base_dir: str = "results/political_prompt3_grouped") -> pd.DataFrame:
    prompt_dirs = load_prompt_dirs(base_dir)
    rows = []

    for pdir in prompt_dirs:
        prompt_name = pdir.name  # usually "prompt3"
        files = sorted(pdir.glob("*.json"))
        print(f"🔍 Scanning {prompt_name} ({len(files)} episodes) from: {pdir}")

        for fpath in files:
            turns = load_episode(fpath)
            if len(turns) < 2:
                continue

            turns = sort_turns_in_episode(turns)
            episode_id = get_episode_id_from_file(turns, fpath)

            # Walk adjacent pairs within the SAME episode file
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
                    "episode_id": episode_id,
                    "turn_idx_a": a.get("turn_idx", i),
                    "turn_idx_b": b.get("turn_idx", i + 1),
                    "speaker_a": spk_a,
                    "speaker_b": spk_b,
                    "num_assumptions_a": num_assumptions_a,
                    "b_is_clarification": int(b_is_clarif),
                    "turn_text_a": a.get("turn_text", ""),
                    "turn_text_b": b_text,
                })

    return pd.DataFrame(rows)


# -----------------------------
# Analysis + Plotting
# -----------------------------
def plot_probability(df_pairs: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    if df_pairs.empty:
        print("⚠️ No valid A->B pairs found.")
        return

    df_pairs["num_assumptions_a_capped"] = df_pairs["num_assumptions_a"].clip(upper=10)

    overall = (
        df_pairs
        .groupby("num_assumptions_a_capped")["b_is_clarification"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "prob"})
    )

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

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    # Plot 1: overall
    plt.figure(figsize=(8.5, 5))
    sns.pointplot(
        data=df_pairs,
        x="num_assumptions_a_capped",
        y="b_is_clarification",
        estimator=np.mean,
        errorbar="se",
    )
    plt.title("P(next turn is clarification | # assumptions in A)")
    plt.xlabel("# assumptions in Speaker A (capped at 10)")
    plt.ylabel("Probability Speaker B clarifies next turn")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / "prob_next_clarification_overall.png", dpi=200)
    plt.close()

    # Plot 2: by prompt (only if >1 prompt dirs)
    if df_pairs["prompt"].nunique() > 1:
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

    pearson = df_pairs["num_assumptions_a"].corr(df_pairs["b_is_clarification"], method="pearson")
    spearman = df_pairs["num_assumptions_a"].corr(df_pairs["b_is_clarification"], method="spearman")

    print("\n=== Diagnostics ===")
    print(f"Total A->B pairs: {len(df_pairs)}")
    print(f"Episodes covered: {df_pairs['episode_id'].nunique()}")
    print(f"Clarification rate overall: {df_pairs['b_is_clarification'].mean():.3f}")
    print(f"Pearson (point-biserial) corr: {pearson:.3f}")
    print(f"Spearman corr: {spearman:.3f}")
    print("Binned counts:")
    print(overall[["num_assumptions_a_capped", "count", "prob"]].to_string(index=False))


def main():
    # NEW default base_dir (your new pipeline output root)
    base_dir = "results/political_prompt3_grouped"  # contains prompt3/ and raw/
    outdir = Path("results/analysis_charts/clarification_prediction_political")

    df_pairs = build_pairs(base_dir)
    plot_probability(df_pairs, outdir)


if __name__ == "__main__":
    main()
