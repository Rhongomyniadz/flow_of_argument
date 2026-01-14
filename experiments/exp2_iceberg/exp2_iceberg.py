from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def compute_duration_seconds(turn: Dict[str, Any]) -> float:
    """
    Tries to compute duration from timestamps if present.
    Falls back to a word-count-based approximation if missing.

    Supported patterns:
      - duration_s
      - end_time - start_time, where keys might be:
        start_time, end_time
        start_time_s, end_time_s
        start, end
        start_sec, end_sec
        start_seconds, end_seconds
        start_ms, end_ms (handled by unit inference)

    If none exist:
      - approximate as words / 2.5  (≈150 wpm)
    """
    d = safe_float(turn.get("duration_s"))
    if d is not None and d > 0:
        return d

    start_candidates = [
        "start_time",
        "start_time_s",
        "start",
        "start_sec",
        "start_seconds",
        "start_ms",
    ]
    end_candidates = [
        "end_time",
        "end_time_s",
        "end",
        "end_sec",
        "end_seconds",
        "end_ms",
    ]

    start_val = None
    end_val = None

    for k in start_candidates:
        v = safe_float(turn.get(k))
        if v is not None:
            start_val = (k, v)
            break

    for k in end_candidates:
        v = safe_float(turn.get(k))
        if v is not None:
            end_val = (k, v)
            break

    if start_val is not None and end_val is not None:
        sk, s = start_val
        ek, e = end_val

        # Unit inference:
        # - If keys explicitly say _ms, assume milliseconds
        # - Else if values are large (e.g., > 10^4), also likely ms
        if sk.endswith("_ms") or ek.endswith("_ms") or (max(abs(s), abs(e)) > 1e4):
            s = s / 1000.0
            e = e / 1000.0

        dur = e - s
        if dur > 0:
            return float(dur)

    # Fallback (no timestamps)
    text = turn.get("turn_text", "") or ""
    words = len(text.split())
    approx = max(words / 2.5, 0.25)  # clamp to avoid 0
    return float(approx)


def compute_iceberg_ratio(explicit_count: int, implicit_count: int, epsilon: float = 1e-6) -> float:
    """
    D_iceberg = explicit / implicit
    If implicit == 0, returns explicit/epsilon (very large), consistent with division-by-zero limit.
    """
    denom = implicit_count if implicit_count > 0 else epsilon
    return float(explicit_count) / float(denom)


@dataclass
class SentimentScorer:
    mode: str  # "vader" or "textblob" or "none"

    def __post_init__(self) -> None:
        self.analyzer = None

        # Try VADER (preferred, lexicon-based, no internet)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            self.analyzer = SentimentIntensityAnalyzer()
            self.mode = "vader"
            return
        except Exception:
            pass

        # Try NLTK VADER
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
            self.analyzer = SentimentIntensityAnalyzer()
            self.mode = "vader"
            return
        except Exception:
            pass

        # Try TextBlob (fallback)
        try:
            from textblob import TextBlob  # type: ignore
            self.analyzer = TextBlob
            self.mode = "textblob"
            return
        except Exception:
            pass

        self.mode = "none"
        self.analyzer = None

    def compound(self, text: str) -> float:
        """
        Returns sentiment compound score in [-1, 1].
        """
        t = (text or "").strip()
        if not t:
            return 0.0

        if self.mode == "vader" and self.analyzer is not None:
            scores = self.analyzer.polarity_scores(t)
            return float(scores.get("compound", 0.0))

        if self.mode == "textblob" and self.analyzer is not None:
            blob = self.analyzer(t)
            return float(blob.sentiment.polarity)

        return 0.0


def logistic(x: float) -> float:
    # numerically stable logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def stance_probability_agreement(sent_compound: float, sharpness: float = 3.0) -> float:
    """
    Map sentiment compound [-1,1] -> P(Agreement) in [0,1] using a logistic curve.
    compound=0 => 0.5
    """
    x = sharpness * float(sent_compound)
    return float(logistic(x))


def rolling_weighted_mean(values: np.ndarray, weights: np.ndarray, window: int) -> np.ndarray:
    """
    For each t, compute weighted mean over [t-window+1, t] with given weights.
    If sum(weights)==0, output NaN at that position.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - window + 1)
        v = values[lo : t + 1]
        w = weights[lo : t + 1]
        denom = np.nansum(w)
        if denom > 0:
            out[t] = float(np.nansum(v * w) / denom)
    return out


def pearson_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xc = x[mask]
    yc = y[mask]
    if np.std(xc) == 0 or np.std(yc) == 0:
        return None
    return float(np.corrcoef(xc, yc)[0, 1])


def granger_test(x: np.ndarray, y: np.ndarray, max_lag: int = 4) -> Dict[str, Any]:
    """
    Test if x Granger-causes y using statsmodels.
    Returns p-values for lags 1..max_lag (ssr_ftest), plus min p-value.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests  # type: ignore
    except Exception as e:
        return {
            "available": False,
            "error": f"statsmodels not available: {e}",
        }

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < (max_lag + 5):
        return {
            "available": True,
            "error": "Not enough data points for Granger test.",
        }

    # statsmodels expects shape (n,2) with columns [y, x]
    data = np.column_stack([y[mask], x[mask]])

    # silence verbose output
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    pvals = {}
    for lag, res in results.items():
        ssr_ftest = res[0].get("ssr_ftest")
        if ssr_ftest is not None and len(ssr_ftest) >= 2:
            pvals[str(lag)] = float(ssr_ftest[1])

    min_p = min(pvals.values()) if pvals else None
    return {
        "available": True,
        "max_lag": max_lag,
        "p_values_ssr_ftest": pvals,
        "min_p_value": min_p,
    }


def compute_episode(episode_path: str, scorer: SentimentScorer, roll_window: int) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    turns = load_json(episode_path)
    if not isinstance(turns, list) or len(turns) == 0:
        raise ValueError(f"Episode file is not a non-empty list: {episode_path}")

    episode_id = turns[0].get("episode_id")
    if episode_id is None:
        # fallback from filename stem
        stem = os.path.splitext(os.path.basename(episode_path))[0]
        try:
            episode_id = int(stem)
        except Exception:
            episode_id = stem

    # Sort by turn_idx for timeline tracking
    turns_sorted = sorted(turns, key=lambda t: int(t.get("turn_idx", 0)))

    per_turn_rows: List[Dict[str, Any]] = []
    for t in turns_sorted:
        if t.get("turn_type_label") != "Substantive":
            continue

        explicit_count = len(t.get("explicit_propositions", []) or [])
        implicit_count = len(t.get("assumptions", []) or [])
        d_iceberg = compute_iceberg_ratio(explicit_count, implicit_count)
        duration_s = compute_duration_seconds(t)
        d_norm = d_iceberg / duration_s if duration_s > 0 else float("nan")

        text = t.get("turn_text", "") or ""
        comp = scorer.compound(text)
        p_agree = stance_probability_agreement(compound_to_float(comp), sharpness=3.0)
        stance_label = "Agreement" if p_agree >= 0.5 else "Disagreement"

        per_turn_rows.append(
            {
                "episode_id": episode_id,
                "turn_idx": t.get("turn_idx"),
                "speaker_id": t.get("speaker_id"),
                "duration_s": duration_s,
                "explicit_count": explicit_count,
                "implicit_count": implicit_count,
                "d_iceberg": d_iceberg,
                "d_norm": d_norm,
                "sentiment_compound": comp,
                "p_agreement": p_agree,
                "stance_label": stance_label,
            }
        )

    # If too few substantive turns, still produce a minimal summary
    d_norm_arr = np.array([r["d_norm"] for r in per_turn_rows], dtype=float) if per_turn_rows else np.array([], dtype=float)
    p_agree_arr = np.array([r["p_agreement"] for r in per_turn_rows], dtype=float) if per_turn_rows else np.array([], dtype=float)
    p_disagree_arr = 1.0 - p_agree_arr

    corr = pearson_corr(p_agree_arr, d_norm_arr)

    # Granger: does D_norm predict Disagreement probability?
    granger = granger_test(d_norm_arr, p_disagree_arr, max_lag=4) if len(per_turn_rows) > 0 else {"available": True, "error": "No substantive turns."}

    summary = {
        "episode_id": episode_id,
        "source_file": episode_path,
        "num_turns_total": len(turns_sorted),
        "num_turns_substantive": len(per_turn_rows),
        "sentiment_backend": scorer.mode,
        "pearson_corr_p_agreement_vs_d_norm": corr,
        "granger_d_norm_causes_p_disagreement": granger,
    }

    # Add rolling series (for plotting + export)
    if len(per_turn_rows) > 0:
        dvals = d_norm_arr.copy()
        agree_series = rolling_weighted_mean(dvals, p_agree_arr, window=roll_window)
        disagree_series = rolling_weighted_mean(dvals, p_disagree_arr, window=roll_window)

        for i, r in enumerate(per_turn_rows):
            r["roll_agreement_d_norm"] = float(agree_series[i]) if np.isfinite(agree_series[i]) else None
            r["roll_disagreement_d_norm"] = float(disagree_series[i]) if np.isfinite(disagree_series[i]) else None

    return episode_id, per_turn_rows, summary


def compound_to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def plot_episode(per_turn_rows: List[Dict[str, Any]], out_png: str, title: str, roll_window: int) -> None:
    if not per_turn_rows:
        return

    xs = np.arange(len(per_turn_rows), dtype=int)
    agree_line = np.array(
        [(r.get("roll_agreement_d_norm") if r.get("roll_agreement_d_norm") is not None else np.nan) for r in per_turn_rows],
        dtype=float,
    )
    disagree_line = np.array(
        [(r.get("roll_disagreement_d_norm") if r.get("roll_disagreement_d_norm") is not None else np.nan) for r in per_turn_rows],
        dtype=float,
    )

    plt.figure(figsize=(10, 4.5))
    plt.plot(xs, agree_line, label=f"Agreement (rolling w={roll_window})")
    plt.plot(xs, disagree_line, label=f"Disagreement (rolling w={roll_window})")
    plt.xlabel("Substantive turn index (in-episode)")
    plt.ylabel("Normalized Iceberg Ratio  D_iceberg / duration_s")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/labeled", help="Folder containing per-episode labeled JSON files.")
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_iceberg", help="Experiment output root folder.")
    parser.add_argument("--roll_window", type=int, default=10, help="Rolling window size (in substantive turns) for time-series lines.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    roll_window = int(args.roll_window)

    per_turn_dir = os.path.join(output_dir, "per_turn")
    summary_dir = os.path.join(output_dir, "summary")
    plots_dir = os.path.join(output_dir, "plots")

    ensure_dir(per_turn_dir)
    ensure_dir(summary_dir)
    ensure_dir(plots_dir)

    scorer = SentimentScorer(mode="vader")

    episode_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not episode_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    all_summaries: List[Dict[str, Any]] = []
    for path in episode_files:
        episode_id, per_turn_rows, summary = compute_episode(path, scorer=scorer, roll_window=roll_window)

        out_per_turn = os.path.join(per_turn_dir, f"{episode_id}.json")
        out_summary = os.path.join(summary_dir, f"{episode_id}.json")
        out_plot = os.path.join(plots_dir, f"{episode_id}.png")

        write_json(out_per_turn, per_turn_rows)
        write_json(out_summary, summary)

        plot_title = f"Exp2 Iceberg Ratio — Episode {episode_id}"
        plot_episode(per_turn_rows, out_plot, title=plot_title, roll_window=roll_window)

        all_summaries.append(summary)
        print(f"[OK] episode={episode_id} substantive_turns={summary['num_turns_substantive']} plot={out_plot}")

    write_json(os.path.join(output_dir, "all_summaries.json"), all_summaries)
    print(f"[DONE] Wrote {len(all_summaries)} episode summaries to {os.path.join(output_dir, 'all_summaries.json')}")


if __name__ == "__main__":
    main()
