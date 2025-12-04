# import argparse
# import json
# import re
# from pathlib import Path
# from typing import Dict, List, Any

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# # -----------------------------
# # Clarification detector
# # -----------------------------
# CLARIFICATION_PATTERNS = [
#     r"\b(could|can|would)\s+you\s+(please\s+)?(explain|clarify|elaborate|unpack|expand|walk me through|say more about|tell me more about)\b",
#     r"\bwhat\s+do\s+you\s+mean\b",
#     r"\bwhen\s+you\s+say\s+that\b",
#     r"\bdo\s+you\s+mean\b",
#     r"\bi'?m\s+not\s+sure\s+i\s+follow\b",
#     r"\bhelp\s+me\s+understand\b",
#     r"\bcan\s+you\s+go\s+over\s+that\b",
#     r"\bcan\s+you\s+repeat\s+that\b",
#     r"\bso\s+you'?re\s+saying\b",
#     r"\bjust\s+to\s+be\s+clear\b",
# ]
# CLARIFICATION_RE = re.compile("|".join(CLARIFICATION_PATTERNS), re.IGNORECASE)

# _WORD_RE = re.compile(r"\w+")


# def count_words(text: str) -> int:
#     return len(_WORD_RE.findall(text or ""))


# def is_clarification_question(text: str, also_any_qmark: bool = False) -> bool:
#     """
#     If also_any_qmark=True, treat ANY '?' as a (looser) clarification proxy.
#     Otherwise use your stricter patterns only.
#     """
#     if not isinstance(text, str):
#         return False
#     t = text.strip()
#     if not t:
#         return False
#     if CLARIFICATION_RE.search(t):
#         return True
#     if also_any_qmark and "?" in t:
#         return True
#     return False


# # -----------------------------
# # I/O
# # -----------------------------
# def load_episode(path: Path) -> List[Dict[str, Any]]:
#     try:
#         data = json.load(open(path, "r", encoding="utf-8"))
#     except Exception:
#         return []
#     return data if isinstance(data, list) else [data]


# def sort_turns_in_episode(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     if not turns:
#         return turns
#     if all(isinstance(t, dict) and "turn_idx" in t for t in turns):
#         try:
#             return sorted(turns, key=lambda x: (x.get("turn_idx") is None, x.get("turn_idx")))
#         except Exception:
#             return turns
#     return turns


# def get_episode_id(turns: List[Dict[str, Any]], fpath: Path) -> Any:
#     if turns and isinstance(turns[0], dict) and "episode_id" in turns[0]:
#         return turns[0].get("episode_id")
#     return int(fpath.stem) if fpath.stem.isdigit() else fpath.stem


# def safe_list(x: Any) -> List[Any]:
#     return x if isinstance(x, list) else []


# # -----------------------------
# # Build Prev -> Next dataset
# # -----------------------------
# def build_prev_next_pairs(
#     base_dir: str,
#     min_prev_words: int = 0,
#     min_next_words: int = 0,
#     also_any_qmark: bool = False,
# ) -> pd.DataFrame:
#     pdir = Path(base_dir)
#     if not pdir.exists():
#         raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

#     files = sorted(pdir.glob("*.json"))
#     print(f"🔍 Scanning {len(files)} episode files from: {pdir}")

#     rows = []
#     for fpath in files:
#         turns = sort_turns_in_episode(load_episode(fpath))
#         if len(turns) < 2:
#             continue

#         episode_id = get_episode_id(turns, fpath)

#         for i in range(len(turns) - 1):
#             prev = turns[i]
#             nxt = turns[i + 1]

#             prev_text = (prev.get("turn_text") or "")
#             nxt_text = (nxt.get("turn_text") or "")

#             if count_words(prev_text) < min_prev_words:
#                 continue
#             if count_words(nxt_text) < min_next_words:
#                 continue

#             num_assumptions_prev = len(safe_list(prev.get("assumptions")))
#             y_next_clarif = int(is_clarification_question(nxt_text, also_any_qmark=also_any_qmark))

#             rows.append({
#                 "episode_id": episode_id,

#                 "turn_idx_prev": prev.get("turn_idx", i),
#                 "turn_idx_next": nxt.get("turn_idx", i + 1),

#                 "speaker_prev": prev.get("speaker_id"),
#                 "speaker_next": nxt.get("speaker_id"),

#                 "num_assumptions_prev": num_assumptions_prev,
#                 "prev_words": count_words(prev_text),
#                 "next_words": count_words(nxt_text),

#                 "next_is_clarification": y_next_clarif,

#                 "turn_text_prev": prev_text,
#                 "turn_text_next": nxt_text,
#             })

#     return pd.DataFrame(rows)


# # -----------------------------
# # Plot + diagnostics
# # -----------------------------
# def plot_probability(df: pd.DataFrame, outdir: Path, cap: int = 10):
#     outdir.mkdir(parents=True, exist_ok=True)

#     if df.empty:
#         print("⚠️ No valid Prev->Next pairs found.")
#         return

#     df["num_assumptions_prev_capped"] = df["num_assumptions_prev"].clip(upper=cap)

#     overall = (
#         df.groupby("num_assumptions_prev_capped")["next_is_clarification"]
#         .agg(["mean", "count", "std"])
#         .reset_index()
#         .rename(columns={"mean": "prob"})
#     )

#     df.to_csv(outdir / "pairs_prev_next.csv", index=False)
#     overall.to_csv(outdir / "binned_overall_prev_assumptions.csv", index=False)

#     sns.set_theme(style="whitegrid")
#     sns.set_context("notebook", font_scale=1.1)

#     plt.figure(figsize=(8.8, 5.2))
#     sns.pointplot(
#         data=df,
#         x="num_assumptions_prev_capped",
#         y="next_is_clarification",
#         estimator=np.mean,
#         errorbar="se",
#     )
#     plt.title("P(next turn is clarification | # assumptions in previous turn)")
#     plt.xlabel(f"# assumptions in previous turn (capped at {cap})")
#     plt.ylabel("Probability next turn clarifies")
#     plt.ylim(0, 1)
#     plt.tight_layout()
#     plt.savefig(outdir / "prob_next_clarification_byPrevAssumptions.png", dpi=200)
#     plt.close()

#     pearson = df["num_assumptions_prev"].corr(df["next_is_clarification"], method="pearson")
#     spearman = df["num_assumptions_prev"].corr(df["next_is_clarification"], method="spearman")

#     print("\n=== Diagnostics ===")
#     print(f"Total Prev->Next pairs: {len(df)}")
#     print(f"Episodes covered: {df['episode_id'].nunique()}")
#     print(f"Clarification rate overall: {df['next_is_clarification'].mean():.3f}")
#     print(f"Pearson corr: {pearson:.3f}")
#     print(f"Spearman corr: {spearman:.3f}")
#     print("\nBinned counts:")
#     print(overall[["num_assumptions_prev_capped", "count", "prob"]].to_string(index=False))


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--base_dir", type=str, default="results/political/parsed")
#     ap.add_argument("--outdir", type=str, default="results/analysis_charts/clarification_prediction")
#     ap.add_argument("--cap", type=int, default=10)
#     ap.add_argument("--min_prev_words", type=int, default=0)
#     ap.add_argument("--min_next_words", type=int, default=0)
#     ap.add_argument("--also_any_qmark", action="store_true",
#                     help="Looser: treat any '?' as clarification in addition to strict patterns.")
#     args = ap.parse_args()

#     df = build_prev_next_pairs(
#         args.base_dir,
#         min_prev_words=args.min_prev_words,
#         min_next_words=args.min_next_words,
#         also_any_qmark=args.also_any_qmark,
#     )
#     plot_probability(df, Path(args.outdir), cap=args.cap)


# if __name__ == "__main__":
#     main()




import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
BASE_DIR = Path("results/political/parsed")  # <-- ONLY read from here
OUTDIR = Path("results/analysis_charts/assumption_propagation")
OUTDIR.mkdir(parents=True, exist_ok=True)

CAP_ASSUMPTIONS = 10            # at most 10 assumptions per turn
JACCARD_THRESHOLD = 0.55        # match threshold for "propagated" assumption
RATIO_BINS = [0.0, 0.01, 0.2, 0.4, 0.6, 0.8, 1.01]

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)


# =========================
# Clarification detection (ONLY '?')
# =========================
def next_is_question_mark(text: str) -> bool:
    return isinstance(text, str) and ("?" in text)


# =========================
# IO helpers
# =========================
def load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ensure_turn_list(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def is_parsed_turn(turn: Dict[str, Any]) -> bool:
    return isinstance(turn, dict) and ("assumptions" in turn or "explicit_propositions" in turn)


def sort_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if turns and all(isinstance(t, dict) and "turn_idx" in t for t in turns):
        try:
            return sorted(turns, key=lambda x: (x.get("turn_idx") is None, x.get("turn_idx")))
        except Exception:
            return turns
    return turns


def infer_prompt_from_path(fpath: Path) -> str:
    # If you have parsed/prompt3/<ep>.json, returns 'prompt3', else 'parsed'
    try:
        rel = fpath.relative_to(BASE_DIR)
        if len(rel.parts) >= 2 and rel.parts[0].startswith("prompt"):
            return rel.parts[0]
    except Exception:
        pass
    return "parsed"


def infer_episode_id(turns: List[Dict[str, Any]], fpath: Path) -> Any:
    if turns and isinstance(turns[0], dict) and "episode_id" in turns[0]:
        return turns[0].get("episode_id")
    return int(fpath.stem) if fpath.stem.isdigit() else fpath.stem


# =========================
# Assumption normalization + propagation
# =========================
def normalize_assumptions(turn: Dict[str, Any], cap: int = 10) -> List[str]:
    items = turn.get("assumptions")
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for it in items:
        txt = it.get("text") if isinstance(it, dict) else it
        if not isinstance(txt, str):
            continue
        s = txt.strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= cap:
            break
    return out


def token_set(s: str) -> set:
    # no regex: a simple whitespace/punct-ish tokenization
    # (good enough for near-duplicate assumption matching)
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in (s or "")).split())


def jaccard(a: str, b: str) -> float:
    wa = token_set(a)
    wb = token_set(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def compute_propagation(prev_as: List[str], next_as: List[str], thr: float) -> Tuple[int, float]:
    """
    Greedy 1-to-1 matching from prev->next by best token-jaccard.
    Count a match if best >= thr.
    Returns (match_count, match_ratio=match_count/len(prev_as)).
    """
    if not prev_as:
        return 0, 0.0
    if not next_as:
        return 0, 0.0

    used = set()
    match = 0
    for pa in prev_as:
        best = -1.0
        best_j = None
        for j, na in enumerate(next_as):
            if j in used:
                continue
            sim = jaccard(pa, na)
            if sim > best:
                best = sim
                best_j = j
        if best_j is not None and best >= thr:
            used.add(best_j)
            match += 1

    return match, match / max(1, len(prev_as))


# =========================
# Build prev->next pairs
# =========================
def build_pairs() -> pd.DataFrame:
    json_files = sorted(BASE_DIR.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under {BASE_DIR}")

    rows: List[Dict[str, Any]] = []
    skipped_not_parsed = 0
    skipped_too_short = 0

    for fpath in json_files:
        obj = load_json(fpath)
        turns_all = ensure_turn_list(obj)
        if not turns_all:
            continue

        # Keep only parsed turns; drop raw-only rows if any
        turns = [t for t in turns_all if is_parsed_turn(t)]
        if len(turns) < 2:
            if len(turns_all) >= 2:
                skipped_not_parsed += 1
            else:
                skipped_too_short += 1
            continue

        turns = sort_turns(turns)
        prompt = infer_prompt_from_path(fpath)
        episode_id = infer_episode_id(turns, fpath)

        for i in range(len(turns) - 1):
            prev = turns[i]
            nxt = turns[i + 1]

            prev_as = normalize_assumptions(prev, cap=CAP_ASSUMPTIONS)
            next_as = normalize_assumptions(nxt, cap=CAP_ASSUMPTIONS)

            match_count, match_ratio = compute_propagation(prev_as, next_as, thr=JACCARD_THRESHOLD)

            rows.append({
                "prompt": prompt,
                "episode_id": episode_id,
                "file": str(fpath),
                "prev_turn_idx": prev.get("turn_idx", i),
                "next_turn_idx": nxt.get("turn_idx", i + 1),
                "speaker_prev": prev.get("speaker_id"),
                "speaker_next": nxt.get("speaker_id"),
                "k_prev": len(prev_as),
                "k_next": len(next_as),
                "prop_match_count": match_count,
                "prop_match_ratio": match_ratio,
                "next_has_question_mark": int(next_is_question_mark(nxt.get("turn_text") or "")),
            })

    df = pd.DataFrame(rows)
    print(f"✅ Loaded pairs: {len(df)}")
    print(f"↪ skipped (not parsed schema): {skipped_not_parsed} files")
    print(f"↪ skipped (too short): {skipped_too_short} files")
    return df


# =========================
# Outputs (minimal plots)
# =========================
def save_and_plot(df: pd.DataFrame):
    if df.empty:
        print("⚠️ No valid prev->next pairs found.")
        return

    df["k_prev_capped"] = df["k_prev"].clip(upper=CAP_ASSUMPTIONS)
    df["prop_ratio_bin"] = pd.cut(df["prop_match_ratio"], bins=RATIO_BINS, include_lowest=True, right=False)

    # Save tables
    df.to_csv(OUTDIR / "pairs.csv", index=False)

    k_summary = (
        df.groupby("k_prev_capped")["next_has_question_mark"]
          .agg(["mean", "count"])
          .reset_index()
          .rename(columns={"mean": "prob_next_has_qmark", "count": "n_pairs"})
    )
    k_summary.to_csv(OUTDIR / "qmark_by_kprev.csv", index=False)

    prop_summary = (
        df.groupby("prop_ratio_bin")["next_has_question_mark"]
          .agg(["mean", "count"])
          .reset_index()
          .rename(columns={"mean": "prob_next_has_qmark", "count": "n_pairs"})
    )
    prop_summary.to_csv(OUTDIR / "qmark_by_prop_ratio_bin.csv", index=False)

    # Diagnostics
    pearson = df["k_prev"].corr(df["next_has_question_mark"], method="pearson")
    spearman = df["k_prev"].corr(df["next_has_question_mark"], method="spearman")

    print("\n=== Diagnostics ===")
    print(f"Total Prev->Next pairs: {len(df)}")
    print(f"Episodes covered: {df['episode_id'].nunique()}")
    print(f"Next-turn '?' rate overall: {df['next_has_question_mark'].mean():.3f}")
    print(f"Pearson corr (k_prev vs next '?'): {pearson:.3f}")
    print(f"Spearman corr (k_prev vs next '?'): {spearman:.3f}")
    print("\nBinned counts (k_prev capped):")
    print(k_summary.to_string(index=False))

    # Plot 1: baseline
    plt.figure(figsize=(9, 5))
    sns.pointplot(
        data=df,
        x="k_prev_capped",
        y="next_has_question_mark",
        estimator=np.mean,
        errorbar="se",
    )
    plt.title("P(next turn contains '?' | #assumptions in previous turn)")
    plt.xlabel(f"# assumptions in previous turn (capped at {CAP_ASSUMPTIONS})")
    plt.ylabel("Probability next turn has '?'")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTDIR / "prob_qmark_by_kprev.png", dpi=200)
    plt.close()

    # Plot 2: propagation-focused
    plt.figure(figsize=(10, 5.2))
    sns.pointplot(
        data=prop_summary,
        x="prop_ratio_bin",
        y="prob_next_has_qmark",
        errorbar=None,
    )
    plt.title("P(next turn contains '?' | assumption propagation ratio bin)")
    plt.xlabel("Propagation ratio bin")
    plt.ylabel("Probability next turn has '?'")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "prob_qmark_by_propagation_bin.png", dpi=200)
    plt.close()

    print(f"\n💾 Wrote outputs to: {OUTDIR}")


def main():
    print(f"📂 Reading JSONs from: {BASE_DIR}")
    df = build_pairs()
    save_and_plot(df)


if __name__ == "__main__":
    main()






