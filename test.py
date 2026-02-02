import argparse
import contextlib
import io
import json
import logging
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from statsmodels.tsa.stattools import grangercausalitytests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def compute_iceberg_ratio(
    explicit_cnt: int,
    implicit_cnt: int,
    duration_sec: float,
    metric_type: str = "prop",
) -> Tuple[float, float]:
    total = explicit_cnt + implicit_cnt

    if metric_type == "prop":
        iceberg_raw = explicit_cnt / total if total > 0 else 0.0
        iceberg_norm = iceberg_raw / max(duration_sec, 0.1)

    elif metric_type == "ratio":
        eps = 1e-6
        iceberg_raw = explicit_cnt / (implicit_cnt + eps)
        iceberg_norm = iceberg_raw / max(duration_sec, 0.1)

    elif metric_type == "log_ratio":
        alpha = 0.5
        iceberg_raw = math.log(explicit_cnt + alpha) - math.log(implicit_cnt + alpha)
        iceberg_norm = iceberg_raw / max(duration_sec, 0.1)

    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    return float(iceberg_norm), float(iceberg_raw)


def extract_turn_features(turn: Dict) -> Optional[Dict]:
    if turn.get("turn_type_label") != "Substantive":
        return None

    stance = turn.get("stance_5pt")
    if stance is None or not (1 <= stance <= 5):
        return None

    duration = turn.get("duration")
    if not (isinstance(duration, (int, float)) and duration > 0.5):
        st, et = turn.get("startTime"), turn.get("endTime")
        if not (
            isinstance(st, (int, float))
            and isinstance(et, (int, float))
            and et > st + 0.5
        ):
            return None
        duration = float(et - st)

    explicit = turn.get("explicit_propositions", []) or []
    implicit = turn.get("assumptions", []) or []
    exp_cnt, imp_cnt = len(explicit), len(implicit)

    if exp_cnt + imp_cnt < 1:
        return None

    speaker_id = turn.get("speaker_id") or turn.get("speaker")
    if not speaker_id or not isinstance(speaker_id, str):
        return None

    return {
        "turn_idx": turn.get("turn_idx"),
        "startTime": turn.get("startTime"),
        "stance_5pt": float(stance),
        "explicit_cnt": exp_cnt,
        "implicit_cnt": imp_cnt,
        "duration": float(duration),
        "speaker_id": speaker_id,
    }


def process_episode_turns(
    turns: List[Dict],
    metric_type: str = "prop",
    min_turns: int = 30,
) -> Optional[pd.DataFrame]:
    features: List[Dict[str, Any]] = []
    for turn in turns:
        feat = extract_turn_features(turn)
        if feat:
            iceberg_norm, iceberg_raw = compute_iceberg_ratio(
                feat["explicit_cnt"],
                feat["implicit_cnt"],
                feat["duration"],
                metric_type=metric_type,
            )
            feat["iceberg_norm"] = iceberg_norm
            feat["iceberg_raw"] = iceberg_raw
            features.append(feat)

    if len(features) < min_turns:
        return None

    df = pd.DataFrame(features)

    if df["startTime"].notna().any():
        df = df.sort_values("startTime").reset_index(drop=True)
        df["t_local"] = df["startTime"]
    else:
        df = df.sort_values("turn_idx").reset_index(drop=True)
        df["t_local"] = df["turn_idx"].astype(float)

    if df["speaker_id"].nunique() < 2:
        return None

    if df["stance_5pt"].nunique() < 3 or df["iceberg_norm"].nunique() < 5:
        return None

    df["d_stance"] = df["stance_5pt"].diff()
    return df


def _pearson_correlation(x: np.ndarray, y: np.ndarray, min_n: int = 10) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < min_n or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denominator == 0:
        return None
    return float(numerator / denominator)


def analyze_cross_speaker_pearson(
    df: pd.DataFrame,
    max_shift: int = 25,
    ice_smooth: int = 3,
    stance_smooth: int = 3,
) -> Dict[str, Any]:
    df = df.copy()
    df["iceberg_smooth"] = df["iceberg_norm"].rolling(ice_smooth, center=True, min_periods=1).mean()
    df["stance_smooth"] = df["stance_5pt"].rolling(stance_smooth, center=True, min_periods=1).mean()
    
    speakers = df["speaker_id"].unique()
    if len(speakers) < 2:
        return {
            "best_shift": None, "best_corr": None, "profile": {}, 
            "causal_direction": None, "n_valid_pairs": 0
        }
    
    offset_to_corrs: Dict[int, List[float]] = {k: [] for k in range(-max_shift, max_shift + 1)}
    
    for X in speakers:
        for Y in speakers:
            if X == Y:
                continue
            
            for offset in range(-max_shift, max_shift + 1):
                pairs = []
                for idx in range(len(df)):
                    if df.loc[idx, "speaker_id"] != X:
                        continue
                    target_idx = idx + offset
                    if target_idx < 0 or target_idx >= len(df):
                        continue
                    if df.loc[target_idx, "speaker_id"] != Y:
                        continue
                    
                    s_val = df.loc[idx, "stance_smooth"]
                    i_val = df.loc[target_idx, "iceberg_smooth"]
                    if np.isfinite(s_val) and np.isfinite(i_val):
                        pairs.append((s_val, i_val))
                
                if len(pairs) >= 12:
                    s_vals = np.array([p[0] for p in pairs])
                    i_vals = np.array([p[1] for p in pairs])
                    r = _pearson_correlation(s_vals, i_vals, min_n=12)
                    if r is not None:
                        offset_to_corrs[offset].append(r)
    
    profile: Dict[int, Optional[float]] = {}
    total_pairs = 0
    for offset in range(-max_shift, max_shift + 1):
        corrs = offset_to_corrs[offset]
        total_pairs += len(corrs)
        if corrs:
            profile[offset] = float(np.mean(corrs))
        else:
            profile[offset] = None
    
    valid = [(o, r) for o, r in profile.items() if r is not None and np.isfinite(r)]
    if not valid:
        return {
            "best_shift": None, "best_corr": None, "profile": profile,
            "causal_direction": None, "n_valid_pairs": total_pairs
        }
    
    best_shift, best_corr = min(valid, key=lambda t: t[1])
    
    if best_shift > 0:
        causal_direction = "H1_support"
    elif best_shift < 0:
        causal_direction = "H2_support"
    else:
        causal_direction = "synchronous"
    
    return {
        "best_shift": int(best_shift),
        "best_corr": float(best_corr),
        "profile": profile,
        "causal_direction": causal_direction,
        "n_valid_pairs": int(total_pairs),
    }


def _granger_best_pvalue(x: np.ndarray, y: np.ndarray, max_lag: int) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < max(30, 3 * max_lag + 10):
        return None

    data = np.column_stack([y, x])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with contextlib.redirect_stdout(io.StringIO()):
                res = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        pvals = []
        for lag in range(1, max_lag + 1):
            p = res[lag][0]["ssr_ftest"][1]
            if p is not None and np.isfinite(p) and 0 < p <= 1:
                pvals.append(float(p))
        if not pvals:
            return None
        return min(pvals)
    except Exception:
        return None


def analyze_cross_speaker_granger(
    df: pd.DataFrame,
    max_shift: int = 25,
    granger_lag: Optional[int] = None,
    ice_smooth: int = 3,
    stance_smooth: int = 3,
) -> Dict[str, Any]:
    if granger_lag is None:
        granger_lag = max(1, min(5, max_shift))
    
    df = df.copy()
    df["iceberg_smooth"] = df["iceberg_norm"].rolling(ice_smooth, center=True, min_periods=1).mean()
    df["stance_smooth"] = df["stance_5pt"].rolling(stance_smooth, center=True, min_periods=1).mean()
    
    speakers = df["speaker_id"].unique()
    if len(speakers) < 2:
        return {
            "best_shift": None, "best_corr": None, "profile": {}, 
            "causal_direction": None, "best_p": None, "granger_lag": granger_lag,
            "n_valid_pairs": 0
        }
    
    offset_to_scores: Dict[int, List[float]] = {k: [] for k in range(-max_shift, max_shift + 1)}
    offset_to_ps: Dict[int, List[float]] = {k: [] for k in range(-max_shift, max_shift + 1)}
    
    for X in speakers:
        for Y in speakers:
            if X == Y:
                continue
            
            for offset in range(-max_shift, max_shift + 1):
                x_vals, y_vals = [], []
                for idx in range(len(df)):
                    if df.loc[idx, "speaker_id"] != X:
                        continue
                    target_idx = idx + offset
                    if target_idx < 0 or target_idx >= len(df):
                        continue
                    if df.loc[target_idx, "speaker_id"] != Y:
                        continue
                    
                    x_val = df.loc[idx, "stance_smooth"]
                    y_val = df.loc[target_idx, "iceberg_smooth"]
                    if np.isfinite(x_val) and np.isfinite(y_val):
                        x_vals.append(x_val)
                        y_vals.append(y_val)
                
                if len(x_vals) >= max(30, 3 * granger_lag + 10):
                    x_arr = np.array(x_vals)
                    y_arr = np.array(y_vals)
                    p = _granger_best_pvalue(x_arr, y_arr, max_lag=granger_lag)
                    if p is not None and 0 < p <= 1:
                        score = -math.log10(p)
                        offset_to_scores[offset].append(score)
                        offset_to_ps[offset].append(p)
    
    profile: Dict[int, Optional[float]] = {}
    p_profile: Dict[int, Optional[float]] = {}
    total_pairs = 0
    for offset in range(-max_shift, max_shift + 1):
        scores = offset_to_scores[offset]
        ps = offset_to_ps[offset]
        total_pairs += len(scores)
        if scores:
            profile[offset] = float(np.mean(scores))
            p_profile[offset] = float(np.mean(ps))
        else:
            profile[offset] = None
            p_profile[offset] = None
    
    valid = [(o, sc) for o, sc in profile.items() if sc is not None and np.isfinite(sc)]
    if not valid:
        return {
            "best_shift": None, "best_corr": None, "profile": profile,
            "causal_direction": None, "best_p": None, "granger_lag": granger_lag,
            "n_valid_pairs": total_pairs
        }
    
    best_shift, best_score = max(valid, key=lambda t: t[1])
    best_p = p_profile.get(best_shift)
    
    if best_shift > 0:
        causal_direction = "H1_support"
    elif best_shift < 0:
        causal_direction = "H2_support"
    else:
        causal_direction = "synchronous"
    
    return {
        "best_shift": int(best_shift),
        "best_corr": float(best_score),
        "best_p": float(best_p) if best_p is not None else None,
        "profile": profile,
        "causal_direction": causal_direction,
        "granger_lag": int(granger_lag),
        "n_valid_pairs": int(total_pairs),
    }


def plot_offset_pearson_and_granger(
    offset_to_pearson: Dict[int, List[float]],
    offset_to_granger: Dict[int, List[float]],
    output_path: Path,
    metric_type: str,
) -> None:
    offsets = sorted(set(offset_to_pearson.keys()) | set(offset_to_granger.keys()))
    xs: List[int] = []
    pearson_mean: List[float] = []
    granger_mean: List[float] = []

    for o in offsets:
        pr = [v for v in offset_to_pearson.get(o, []) if v is not None and np.isfinite(v)]
        gr = [v for v in offset_to_granger.get(o, []) if v is not None and np.isfinite(v)]

        if len(pr) == 0 and len(gr) == 0:
            continue

        xs.append(int(o))
        pearson_mean.append(float(np.mean(pr)) if len(pr) else float("nan"))
        granger_mean.append(float(np.mean(gr)) if len(gr) else float("nan"))

    if len(xs) == 0:
        logger.error("No values to plot for Pearson+Granger overlay.")
        return

    xs_arr = np.array(xs, dtype=int)
    pearson_arr = np.array(pearson_mean, dtype=float)
    granger_arr = np.array(granger_mean, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(xs_arr, pearson_arr, marker="o", linestyle="-", linewidth=2, 
            markersize=6, color="tab:blue", label="Pearson r (cross-speaker)")
    ax.plot(xs_arr, granger_arr, marker="s", linestyle="-", linewidth=2, 
            markersize=6, color="tab:orange", label="Granger evidence (-log10 p)")
    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.6, color="gray")
    ax.axvline(0, linestyle=":", linewidth=1, alpha=0.6, color="gray")
    ax.set_xlabel(
        "Offset (turns)\n"
        "Negative: other speaker's stance preceded current speaker's iceberg (H2)\n"
        "Positive: current speaker's stance preceded other speaker's iceberg (H1)"
    )
    ax.set_ylabel("Correlation / Evidence Score")
    ax.set_title(f"Cross-Speaker Analysis: Stance vs Iceberg ({metric_type})\n"
                 "Correlating Speaker X's stance with Speaker Y's iceberg (X ≠ Y)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cross-speaker overlay plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-speaker Gricean Repair Mechanism Analysis"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="prop", choices=["prop", "ratio", "log_ratio"])
    parser.add_argument("--min_turns", type=int, default=30)
    parser.add_argument("--max_shift", type=int, default=25)
    parser.add_argument("--granger_lag", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_iceberg/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("CROSS-SPEAKER GRICEAN REPAIR MECHANISM ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Data directory     : {args.data_dir}")
    logger.info(f"Iceberg metric     : {args.metric}")
    logger.info(f"Offset range       : [-{args.max_shift}, +{args.max_shift}]")
    logger.info(f"Granger max lag    : {args.granger_lag}")
    logger.info(f"Min turns/episode  : {args.min_turns}")
    logger.info(f"Output directory   : {args.output_dir}")
    logger.info(f"Start time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    episode_paths = sorted(Path(args.data_dir).glob("*.json"))
    logger.info(f"Found {len(episode_paths)} JSON files")

    offset_to_pearson: Dict[int, List[float]] = {o: [] for o in range(-args.max_shift, args.max_shift + 1)}
    offset_to_granger: Dict[int, List[float]] = {o: [] for o in range(-args.max_shift, args.max_shift + 1)}

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for path in tqdm(episode_paths, desc="Processing episodes"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                turns = json.load(f)

            df = process_episode_turns(turns, metric_type=args.metric, min_turns=args.min_turns)
            if df is None or len(df) < args.min_turns:
                failures.append((path.stem, "quality_control_failed"))
                continue

            pearson_res = analyze_cross_speaker_pearson(df, max_shift=args.max_shift)
            granger_res = analyze_cross_speaker_granger(
                df, max_shift=args.max_shift, granger_lag=args.granger_lag
            )

            p_prof = pearson_res.get("profile", {})
            g_prof = granger_res.get("profile", {})

            for o in range(-args.max_shift, args.max_shift + 1):
                pv = p_prof.get(o)
                if pv is not None and np.isfinite(pv):
                    offset_to_pearson[o].append(float(pv))

                gv = g_prof.get(o)
                if gv is not None and np.isfinite(gv):
                    offset_to_granger[o].append(float(gv))

            results.append(
                {
                    "episode_id": path.stem,
                    "n_turns": int(len(df)),
                    "n_speakers": int(df["speaker_id"].nunique()),
                    "pearson_best_shift": pearson_res.get("best_shift"),
                    "pearson_best_r": pearson_res.get("best_corr"),
                    "pearson_direction": pearson_res.get("causal_direction"),
                    "pearson_n_pairs": pearson_res.get("n_valid_pairs", 0),
                    "granger_best_shift": granger_res.get("best_shift"),
                    "granger_best_score_neglog10p": granger_res.get("best_corr"),
                    "granger_best_p": granger_res.get("best_p"),
                    "granger_direction": granger_res.get("causal_direction"),
                    "granger_lag": granger_res.get("granger_lag"),
                    "granger_n_pairs": granger_res.get("n_valid_pairs", 0),
                    "mean_iceberg": float(df["iceberg_norm"].mean()),
                    "mean_stance": float(df["stance_5pt"].mean()),
                }
            )

        except Exception as e:
            failures.append((path.stem, f"exception: {str(e)[:120]}"))
            continue

    n_total = len(episode_paths)
    n_success = len(results)
    n_fail = len(failures)

    logger.info("\nProcessing complete:")
    logger.info(f"  Successfully processed : {n_success} / {n_total} episodes ({(n_success / max(n_total, 1)) * 100:.1f}%)")
    logger.info(f"  Failed (QC/filtering)  : {n_fail} episodes")

    df_agg = pd.DataFrame(results)
    df_agg.to_csv(output_path / f"episode_results_cross_speaker_{args.metric}.csv", index=False)

    pearson_df = pd.DataFrame.from_dict(offset_to_pearson, orient="index").T
    pearson_df.columns.name = "offset"
    pearson_df.to_csv(output_path / f"raw_pearson_by_offset_{args.metric}.csv", index=False)

    granger_df = pd.DataFrame.from_dict(offset_to_granger, orient="index").T
    granger_df.columns.name = "offset"
    granger_df.to_csv(output_path / f"raw_granger_by_offset_{args.metric}.csv", index=False)

    plot_offset_pearson_and_granger(
        offset_to_pearson=offset_to_pearson,
        offset_to_granger=offset_to_granger,
        output_path=output_path / f"cross_speaker_pearson_vs_granger_{args.metric}.png",
        metric_type=args.metric,
    )

    pct_p_h1 = float((df_agg["pearson_direction"] == "H1_support").mean() * 100) if len(df_agg) else None
    pct_p_h2 = float((df_agg["pearson_direction"] == "H2_support").mean() * 100) if len(df_agg) else None
    pct_g_h1 = float((df_agg["granger_direction"] == "H1_support").mean() * 100) if len(df_agg) else None
    pct_g_h2 = float((df_agg["granger_direction"] == "H2_support").mean() * 100) if len(df_agg) else None

    summary = {
        "timestamp": start_time.isoformat(),
        "data_dir": args.data_dir,
        "metric": args.metric,
        "min_turns": args.min_turns,
        "max_shift": args.max_shift,
        "granger_lag": args.granger_lag,
        "n_files": int(n_total),
        "n_success": int(n_success),
        "n_failed": int(n_fail),
        "pearson": {
            "mean_best_shift": float(df_agg["pearson_best_shift"].mean()) if len(df_agg) else None,
            "mean_best_r": float(df_agg["pearson_best_r"].mean()) if len(df_agg) else None,
            "pct_H1_support": pct_p_h1,
            "pct_H2_support": pct_p_h2,
        },
        "granger": {
            "mean_best_shift": float(df_agg["granger_best_shift"].mean()) if len(df_agg) else None,
            "mean_best_score_neglog10p": float(df_agg["granger_best_score_neglog10p"].mean()) if len(df_agg) else None,
            "mean_best_p": float(df_agg["granger_best_p"].mean()) if len(df_agg) else None,
            "pct_H1_support": pct_g_h1,
            "pct_H2_support": pct_g_h2,
            "score_definition": "score = -log10(min p over lags 1..granger_lag from ssr_ftest)",
        },
        "artifacts": {
            "episode_csv": str(output_path / f"episode_results_cross_speaker_{args.metric}.csv"),
            "raw_pearson_csv": str(output_path / f"raw_pearson_by_offset_{args.metric}.csv"),
            "raw_granger_csv": str(output_path / f"raw_granger_by_offset_{args.metric}.csv"),
            "overlay_plot": str(output_path / f"cross_speaker_pearson_vs_granger_{args.metric}.png"),
        },
    }

    with open(output_path / f"run_summary_cross_speaker_{args.metric}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if failures:
        with open(output_path / f"failures_cross_speaker_{args.metric}.json", "w", encoding="utf-8") as f:
            json.dump([{"episode_id": eid, "reason": reason} for eid, reason in failures], f, indent=2)

    logger.info(f"Saved run summary JSON to {output_path / f'run_summary_cross_speaker_{args.metric}.json'}")


if __name__ == "__main__":
    main()
