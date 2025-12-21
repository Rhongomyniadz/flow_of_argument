import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


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
_PROMPT_RE = re.compile(r"(?:^|/|\\)prompt(\d+)(?:/|\\|$)", re.IGNORECASE)


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


def infer_prompt_num(p: Path) -> Optional[int]:
    m = _PROMPT_RE.search(str(p))
    return int(m.group(1)) if m else None


def iter_episode_files(base_dir: Path) -> List[Tuple[Optional[int], Path]]:
    """
    Supports two layouts:
      (A) base_dir contains *.json directly
      (B) base_dir contains prompt*/ subdirs, each containing *.json

    Returns list of (prompt_num, json_path).
    """
    direct = sorted(base_dir.glob("*.json"))
    if direct:
        pn = infer_prompt_num(base_dir)
        return [(pn, fp) for fp in direct]

    prompt_dirs = sorted([d for d in base_dir.glob("prompt*") if d.is_dir()])
    if prompt_dirs:
        out: List[Tuple[Optional[int], Path]] = []
        for d in prompt_dirs:
            pn = infer_prompt_num(d)
            for fp in sorted(d.glob("*.json")):
                out.append((pn, fp))
        return out

    return []


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

    episode_files = iter_episode_files(pdir)
    print(f"🔍 Scanning {len(episode_files)} episode files from: {pdir}")

    rows = []
    for prompt_num, fpath in episode_files:
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
            num_explicit_prev = len(safe_list(prev.get("explicit_propositions")))

            y_next_clarif = int(is_clarification_question(nxt_text, also_any_qmark=also_any_qmark))

            rows.append({
                "prompt_num": prompt_num if prompt_num is not None else "all",
                "episode_id": episode_id,

                "turn_idx_prev": prev.get("turn_idx", i),
                "turn_idx_next": nxt.get("turn_idx", i + 1),

                "speaker_prev": prev.get("speaker_id"),
                "speaker_next": nxt.get("speaker_id"),

                "num_assumptions_prev": num_assumptions_prev,
                "num_explicit_prev": num_explicit_prev,

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
# Extract only the question/clarification sentence
# -----------------------------
def extract_question_sentence(text: str, also_any_qmark: bool = False) -> str:
    """
    Return just the sentence within `text` that triggers the clarification detector.

    Heuristic:
      1. Split into sentences on [.!?] + whitespace.
      2. Return the first sentence that `is_clarification_question(...)` flags.
      3. If none, return the first sentence with '?'.
      4. If still none, fall back to the whole text.
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()
    if not t:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", t)

    for s in sentences:
        if is_clarification_question(s, also_any_qmark=also_any_qmark):
            return s.strip()

    for s in sentences:
        if "?" in s:
            return s.strip()

    return t


# -----------------------------
# Export clarification questions for Google Sheets
# -----------------------------
def export_questions_to_sheet(df: pd.DataFrame, outdir: Path, also_any_qmark: bool):
    """
    Extract all NEXT turns that are labeled as clarification questions and save
    ONLY the question/clarification sentence (plus some metadata) to a CSV.
    """
    if df.empty:
        print("⚠️ No data for question export.")
        return

    q_df = df[df["next_is_clarification"] == 1].copy()
    if q_df.empty:
        print("⚠️ No clarification questions found; skipping question export.")
        return

    q_df["clarification_sentence"] = q_df["turn_text_next"].apply(
        lambda t: extract_question_sentence(t, also_any_qmark=also_any_qmark)
    )

    cols = [
        "prompt_num",
        "episode_id",
        "turn_idx_prev",
        "speaker_prev",
        "turn_text_prev",
        "turn_idx_next",
        "speaker_next",
        "clarification_sentence",
        "num_assumptions_prev",
        "num_explicit_prev",
        "prev_words",
        "next_words",
    ]
    cols = [c for c in cols if c in q_df.columns]

    outdir.mkdir(parents=True, exist_ok=True)
    q_df[cols].to_csv(outdir / "questions.csv", index=False)
    print(f"💾 Wrote clarification questions to {outdir / 'questions.csv'}")


# -----------------------------
# Forest plot for statsmodels logit
# -----------------------------
def forest_plot_logit(result, outpath: Path, drop_intercept: bool = True):
    """
    Forest plot of coefficients (log-odds) with 95% CI from a fitted statsmodels Logit result.
    """
    params = result.params.copy()

    try:
        conf = result.conf_int(alpha=0.05)
        conf.columns = ["ci_lower", "ci_upper"]
    except Exception as e:
        print(f"⚠️ Could not compute conf_int() for forest plot ({e}); skipping.")
        return

    if drop_intercept and "Intercept" in params.index:
        params = params.drop("Intercept")
        conf = conf.drop("Intercept")

    if len(params) == 0:
        print("⚠️ No coefficients to plot (after dropping intercept).")
        return

    y_pos = np.arange(len(params.index))

    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.45 * len(params.index) + 1)))
    ax.errorbar(
        x=params.values,
        y=y_pos,
        xerr=[params.values - conf["ci_lower"].values, conf["ci_upper"].values - params.values],
        fmt="o",
        capsize=4,
    )
    ax.axvline(x=0, linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index.tolist())
    ax.set_xlabel("Log-odds (coefficient)")
    ax.set_title("Logistic Regression Coefficients (95% CI)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"🖼️ Saved forest plot to {outpath}")


# -----------------------------
# Logistic regression (statsmodels)
# -----------------------------
def run_regression(df: pd.DataFrame, outdir: Path):
    """
    Fit logistic regression (statsmodels, R-style formula):
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
        *([ "prompt_num" ] if "prompt_num" in df.columns else []),
    ]].dropna()

    if reg_df["is_question_asked"].nunique() < 2:
        print("⚠️ is_question_asked has <2 unique values; skipping regression.")
        return

    reg_df["is_question_asked"] = reg_df["is_question_asked"].astype(int)

    outdir.mkdir(parents=True, exist_ok=True)

    formula = "is_question_asked ~ num_assumptions + num_words_in_turn + num_explicit_statements"
    model = smf.logit(formula=formula, data=reg_df)

    try:
        result = model.fit(disp=False, maxiter=200)
        fit_mode = "fit"
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as e:
        print(f"⚠️ statsmodels logit fit failed ({e}); trying fit_regularized...")
        result = model.fit_regularized(disp=False)
        fit_mode = "fit_regularized"

    print("\n=== Logistic regression (statsmodels) ===")
    print("Formula:", formula)
    try:
        print(result.summary())
    except Exception:
        print("⚠️ summary() not available for this fit; printing params only:")
        print(result.params)

    with open(outdir / "logit_statsmodels_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Fit mode: {fit_mode}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Samples: {len(reg_df)}\n\n")
        try:
            f.write(str(result.summary()))
        except Exception:
            f.write("summary() not available; params:\n")
            f.write(str(result.params))

    try:
        coef_table = result.summary2().tables[1]
        coef_table.to_csv(outdir / "logit_statsmodels_coef_table.csv", index=True)
        print("\nCoefficient table:\n")
        print(coef_table)
    except Exception as e:
        print(f"⚠️ Could not export summary2() coef table ({e}).")

    forest_plot_logit(result, outdir / "logit_statsmodels_forest.png", drop_intercept=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="results/political/parsed")
    ap.add_argument("--outdir", type=str, default="results/analysis_charts/clarification_prediction")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--min_prev_words", type=int, default=0)
    ap.add_argument("--min_next_words", type=int, default=0)
    ap.add_argument(
        "--also_any_qmark",
        action="store_true",
        help="Looser: treat any '?' as clarification in addition to strict patterns.",
    )
    args = ap.parse_args()

    df = build_prev_next_pairs(
        args.base_dir,
        min_prev_words=args.min_prev_words,
        min_next_words=args.min_next_words,
        also_any_qmark=args.also_any_qmark,
    )
    outdir = Path(args.outdir)

    plot_probability(df, outdir, cap=args.cap)
    export_questions_to_sheet(df, outdir, also_any_qmark=args.also_any_qmark)
    run_regression(df, outdir)


if __name__ == "__main__":
    main()
