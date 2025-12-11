import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression  # NEW
from sklearn.metrics import classification_report, roc_auc_score  # NEW


# -----------------------------
# Clarification detector
# -----------------------------
CLARIFICATION_PATTERNS = [
    r"\b(could|can|would)\s+you\s+(please\s+)?(explain|clarify|elaborate|unpack|expand|walk me through|say more about|tell me more about)\b",
    r"\bwhat\s+do\s+you\s+mean\b",
    r"\bwhen\s+you\s+say\s+that\b",
    r"\bdo\s+you\s+mean\b",
    r"\bi'?m\s+not\s+sure\s+i\s+follow\b",
    r"\bhelp\s+me\s+understand\b",
    r"\bcan\s+you\s+go\s+over\s+that\b",
    r"\bcan\s+you\s+repeat\s+that\b",
    r"\bso\s+you'?re\s+saying\b",
    r"\bjust\s+to\s+be\s+clear\b",
]
CLARIFICATION_RE = re.compile("|".join(CLARIFICATION_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def is_clarification_question(text: str, also_any_qmark: bool = False) -> bool:
    """
    If also_any_qmark=True, treat ANY '?' as a (looser) clarification proxy.
    Otherwise use your stricter patterns only.
    """
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    if CLARIFICATION_RE.search(t):
        return True
    if also_any_qmark and "?" in t:
        return True
    return False


# -----------------------------
# I/O
# -----------------------------
def load_episode(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else [data]


def sort_turns_in_episode(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not turns:
        return turns
    if all(isinstance(t, dict) and "turn_idx" in t for t in turns):
        try:
            return sorted(turns, key=lambda x: (x.get("turn_idx") is None, x.get("turn_idx")))
        except Exception:
            return turns
    return turns


def get_episode_id(turns: List[Dict[str, Any]], fpath: Path) -> Any:
    if turns and isinstance(turns[0], dict) and "episode_id" in turns[0]:
        return turns[0].get("episode_id")
    return int(fpath.stem) if fpath.stem.isdigit() else fpath.stem


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


# -----------------------------
# Build Prev -> Next dataset
# -----------------------------
def build_prev_next_pairs(
    base_dir: str,
    min_prev_words: int = 0,
    min_next_words: int = 0,
    also_any_qmark: bool = False,
) -> pd.DataFrame:
    pdir = Path(base_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

    files = sorted(pdir.glob("*.json"))
    print(f"🔍 Scanning {len(files)} episode files from: {pdir}")

    rows = []
    for fpath in files:
        turns = sort_turns_in_episode(load_episode(fpath))
        if len(turns) < 2:
            continue

        episode_id = get_episode_id(turns, fpath)

        for i in range(len(turns) - 1):
            prev = turns[i]
            nxt = turns[i + 1]

            prev_text = (prev.get("turn_text") or "")
            nxt_text = (nxt.get("turn_text") or "")

            if count_words(prev_text) < min_prev_words:
                continue
            if count_words(nxt_text) < min_next_words:
                continue

            num_assumptions_prev = len(safe_list(prev.get("assumptions")))
            # NEW: count explicit propositions on previous turn
            num_explicit_prev = len(safe_list(prev.get("explicit_propositions")))

            y_next_clarif = int(is_clarification_question(nxt_text, also_any_qmark=also_any_qmark))

            rows.append({
                "episode_id": episode_id,

                "turn_idx_prev": prev.get("turn_idx", i),
                "turn_idx_next": nxt.get("turn_idx", i + 1),

                "speaker_prev": prev.get("speaker_id"),
                "speaker_next": nxt.get("speaker_id"),

                "num_assumptions_prev": num_assumptions_prev,
                "num_explicit_prev": num_explicit_prev,   # NEW
                "prev_words": count_words(prev_text),
                "next_words": count_words(nxt_text),

                "next_is_clarification": y_next_clarif,

                "turn_text_prev": prev_text,
                "turn_text_next": nxt_text,
            })

    return pd.DataFrame(rows)


# -----------------------------
# Plot + diagnostics
# -----------------------------
def plot_probability(df: pd.DataFrame, outdir: Path, cap: int = 10):
    outdir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("⚠️ No valid Prev->Next pairs found.")
        return

    df["num_assumptions_prev_capped"] = df["num_assumptions_prev"].clip(upper=cap)

    overall = (
        df.groupby("num_assumptions_prev_capped")["next_is_clarification"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "prob"})
    )

    df.to_csv(outdir / "pairs_prev_next.csv", index=False)
    overall.to_csv(outdir / "binned_overall_prev_assumptions.csv", index=False)

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    plt.figure(figsize=(8.8, 5.2))
    sns.pointplot(
        data=df,
        x="num_assumptions_prev_capped",
        y="next_is_clarification",
        estimator=np.mean,
        errorbar="se",
    )
    plt.title("P(next turn is clarification | # assumptions in previous turn)")
    plt.xlabel(f"# assumptions in previous turn (capped at {cap})")
    plt.ylabel("Probability next turn clarifies")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / "prob_next_clarification_byPrevAssumptions.png", dpi=200)
    plt.close()

    pearson = df["num_assumptions_prev"].corr(df["next_is_clarification"], method="pearson")
    spearman = df["num_assumptions_prev"].corr(df["next_is_clarification"], method="spearman")

    print("\n=== Diagnostics ===")
    print(f"Total Prev->Next pairs: {len(df)}")
    print(f"Episodes covered: {df['episode_id'].nunique()}")
    print(f"Clarification rate overall: {df['next_is_clarification'].mean():.3f}")
    print(f"Pearson corr: {pearson:.3f}")
    print(f"Spearman corr: {spearman:.3f}")
    print("\nBinned counts:")
    print(overall[["num_assumptions_prev_capped", "count", "prob"]].to_string(index=False))


# -----------------------------
# Logistic regression (scikit-learn)
# -----------------------------
def run_regression(df: pd.DataFrame, outdir: Path):
    """
    Fit (logistic regression, scikit-learn):
      is_question_asked ~ num_assumptions + num_words_in_turn + num_explicit_statements

    where:
      is_question_asked       := next_is_clarification (binary)
      num_assumptions         := num_assumptions_prev
      num_words_in_turn       := prev_words
      num_explicit_statements := num_explicit_prev
    """
    if df.empty:
        print("⚠️ No data for regression.")
        return

    # Rename columns to match conceptual formula
    reg_df = df.rename(columns={
        "next_is_clarification": "is_question_asked",
        "num_assumptions_prev": "num_assumptions",
        "prev_words": "num_words_in_turn",
        "num_explicit_prev": "num_explicit_statements",
    })[[
        "is_question_asked",
        "num_assumptions",
        "num_words_in_turn",
        "num_explicit_statements",
    ]].dropna()

    # Need both classes present
    if reg_df["is_question_asked"].nunique() < 2:
        print("⚠️ is_question_asked has <2 unique values; skipping regression.")
        return

    y = reg_df["is_question_asked"].astype(int)
    feature_cols = ["num_assumptions", "num_words_in_turn", "num_explicit_statements"]
    X = reg_df[feature_cols]

    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # In-sample predictions
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    acc = (y_pred == y).mean()
    try:
        roc = roc_auc_score(y, y_proba)
    except ValueError:
        roc = float("nan")

    report = classification_report(y, y_pred, digits=3)

    print("\n=== Logistic regression (scikit-learn) ===")
    print("Formula: is_question_asked ~ num_assumptions + num_words_in_turn + num_explicit_statements")
    print(f"Samples: {len(reg_df)}")
    print(f"Accuracy (in-sample): {acc:.3f}")
    print(f"ROC AUC (in-sample): {roc:.3f}")
    print("\nClassification report:\n", report)

    outdir.mkdir(parents=True, exist_ok=True)

    # Save text report
    with open(outdir / "logit_sklearn_is_question_asked.txt", "w", encoding="utf-8") as f:
        f.write("Logistic regression (scikit-learn)\n")
        f.write("Formula: is_question_asked ~ num_assumptions + num_words_in_turn + num_explicit_statements\n")
        f.write(f"Samples: {len(reg_df)}\n")
        f.write(f"Accuracy (in-sample): {acc:.4f}\n")
        f.write(f"ROC AUC (in-sample): {roc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    # Save coefficients (including intercept)
    coef_df = pd.DataFrame({
        "feature": ["intercept"] + feature_cols,
        "coef": [clf.intercept_[0]] + list(clf.coef_[0]),
    })
    coef_df.to_csv(outdir / "logit_sklearn_is_question_asked_coefs.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="results/political/parsed")
    ap.add_argument("--outdir", type=str, default="results/analysis_charts/clarification_prediction")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--min_prev_words", type=int, default=0)
    ap.add_argument("--min_next_words", type=int, default=0)
    ap.add_argument("--also_any_qmark", action="store_true",
                    help="Looser: treat any '?' as clarification in addition to strict patterns.")
    args = ap.parse_args()

    df = build_prev_next_pairs(
        args.base_dir,
        min_prev_words=args.min_prev_words,
        min_next_words=args.min_next_words,
        also_any_qmark=args.also_any_qmark,
    )
    outdir = Path(args.outdir)

    plot_probability(df, outdir, cap=args.cap)
    run_regression(df, outdir)  # NEW


if __name__ == "__main__":
    main()
