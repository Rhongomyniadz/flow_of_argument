import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def load_turns(fp: Path) -> List[Dict[str, Any]]:
    try:
        data = json.load(open(fp, "r", encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def compute_totals_grouped(base_root: Path, start: int = 1, end: int = 5) -> pd.DataFrame:
    """
    Returns a wide DataFrame grouped by prompt_num with columns:
      prompt_num, explicit, implicit, n_turns, n_episodes
    Guarantees prompt_num rows for ALL prompts in [start, end] (fills missing with 0).
    """
    rows = []

    for k in range(start, end + 1):
        pdir = base_root / f"prompt{k}"
        files = sorted(pdir.glob("*.json")) if pdir.exists() else []

        explicit_total = 0
        implicit_total = 0
        n_turns = 0
        n_eps = 0

        for fp in files:
            turns = load_turns(fp)
            if not turns:
                continue
            n_eps += 1
            for t in turns:
                n_turns += 1
                explicit_total += len(safe_list(t.get("explicit_propositions")))
                implicit_total += len(safe_list(t.get("assumptions")))

        rows.append(
            {
                "prompt_num": k,
                "explicit": explicit_total,
                "implicit": implicit_total,
                "n_turns": n_turns,
                "n_episodes": n_eps,
            }
        )

    wide = pd.DataFrame(rows).sort_values("prompt_num").reset_index(drop=True)
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", type=str, default="results/prompt_comparison")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="results/analysis_charts/prompt_comparison/explicit_implicit")
    args = ap.parse_args()

    base_root = Path(args.base_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Grouped totals by prompt_num = 1..5
    wide = compute_totals_grouped(base_root, args.start, args.end)

    # 2) Wide -> long for seaborn
    long_df = wide.melt(
        id_vars=["prompt_num"],
        value_vars=["explicit", "implicit"],
        var_name="statement_type",
        value_name="count",
    )

    # Make prompt_num categorical to preserve ordering 1..5 on the axis
    ordered_prompts = list(range(args.start, args.end + 1))
    long_df["prompt_num"] = pd.Categorical(long_df["prompt_num"], categories=ordered_prompts, ordered=True)

    # Save tables
    wide.to_csv(outdir / "summary_wide.csv", index=False)
    long_df.to_csv(outdir / "counts_long.csv", index=False)

    # 3) Barplot (requested)
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    plt.figure(figsize=(8.5, max(3.8, 0.55 * len(ordered_prompts) + 1)))
    sns.barplot(data=long_df, y="prompt_num", x="count", hue="statement_type")
    plt.title("Explicit vs Implicit counts by prompt (1..5)")
    plt.tight_layout()
    plt.savefig(outdir / "barplot_y.png", dpi=200)
    plt.close()

    print("✅ Done.")
    print("Saved:", outdir / "barplot_y.png")
    print("Saved:", outdir / "summary_wide.csv")
    print("Saved:", outdir / "counts_long.csv")


if __name__ == "__main__":
    main()
