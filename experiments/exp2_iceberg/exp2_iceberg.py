import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# ============================================================================
# FEATURE EXTRACTION & ICEBERG METRIC COMPUTATION
# ============================================================================

def compute_iceberg_ratio(
    explicit_cnt: int, 
    implicit_cnt: int, 
    duration_sec: float, 
    metric_type: str = "prop"
) -> Tuple[float, float]:
    """Compute normalized iceberg ratio using specified metric formulation."""
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
    """Extract analyzable features from dialogue turn for iceberg analysis."""
    if turn.get("turn_type_label") != "Substantive":
        return None
    
    stance = turn.get("stance_5pt")
    if stance is None or not (1 <= stance <= 5):
        return None

    duration = turn.get("duration")
    if not (isinstance(duration, (int, float)) and duration > 0.5):
        st, et = turn.get("startTime"), turn.get("endTime")
        if not (isinstance(st, (int, float)) and isinstance(et, (int, float)) and et > st + 0.5):
            return None
        duration = float(et - st)

    explicit = turn.get("explicit_propositions", []) or []
    implicit = turn.get("assumptions", []) or []
    exp_cnt, imp_cnt = len(explicit), len(implicit)
    
    if exp_cnt + imp_cnt < 1:
        return None
    
    return {
        "turn_idx": turn.get("turn_idx"),
        "startTime": turn.get("startTime"),
        "stance_5pt": float(stance),
        "explicit_cnt": exp_cnt,
        "implicit_cnt": imp_cnt,
        "duration": float(duration),
    }


def process_episode_turns(
    turns: List[Dict],
    metric_type: str = "prop",
    min_turns: int = 20
) -> Optional[pd.DataFrame]:
    """Process dialogue turns into analyzable time-series dataframe."""
    features = []
    for turn in turns:
        feat = extract_turn_features(turn)
        if feat:
            iceberg_norm, iceberg_raw = compute_iceberg_ratio(
                feat["explicit_cnt"],
                feat["implicit_cnt"],
                feat["duration"],
                metric_type=metric_type
            )
            feat["iceberg_norm"] = iceberg_norm
            feat["iceberg_raw"] = iceberg_raw
            features.append(feat)
    
    if len(features) < min_turns:
        return None
    
    df = pd.DataFrame(features)
    
    # Establish temporal ordering
    if df["startTime"].notna().any():
        df = df.sort_values("startTime").reset_index(drop=True)
        df["t_local"] = df["startTime"]
    else:
        df = df.sort_values("turn_idx").reset_index(drop=True)
        df["t_local"] = df["turn_idx"].astype(float)
    
    # Quality control filters
    if df["stance_5pt"].nunique() < 3 or df["iceberg_norm"].nunique() < 5:
        return None
    
    df["d_stance"] = df["stance_5pt"].diff()
    
    return df


# ============================================================================
# TEMPORAL CAUSALITY ANALYSIS
# ============================================================================

def _pearson_correlation(x: np.ndarray, y: np.ndarray, min_n: int = 10) -> Optional[float]:
    """Compute Pearson correlation with robustness checks."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) < min_n or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    
    if denominator == 0:
        return None
    
    return float(numerator / denominator)


def analyze_temporal_causality(
    df: pd.DataFrame,
    max_shift: int = 25,
    ice_smooth: int = 3,
    stance_smooth: int = 3
) -> Dict[str, Any]:
    """Analyze temporal precedence between stance shifts and iceberg ratios."""
    ice_series = df["iceberg_norm"].rolling(
        ice_smooth, center=True, min_periods=1
    ).mean()
    stance_series = df["stance_5pt"].rolling(
        stance_smooth, center=True, min_periods=1
    ).mean()
    
    valid_mask = ice_series.notna() & stance_series.notna()
    ice_vals = ice_series[valid_mask].to_numpy(dtype=float)
    stance_vals = stance_series[valid_mask].to_numpy(dtype=float)
    
    min_required = max_shift * 2 + 15
    if len(ice_vals) < min_required:
        return {"best_shift": None, "best_corr": None, "profile": {}, "causal_direction": None}
    
    profile = {}
    for shift in range(-max_shift, max_shift + 1):
        try:
            if shift >= 0:
                if len(stance_vals) <= shift or len(ice_vals) <= shift:
                    profile[shift] = None
                    continue
                x = stance_vals[:len(stance_vals) - shift]
                y = ice_vals[shift:]
            else:
                k = -shift
                if len(stance_vals) <= k or len(ice_vals) <= k:
                    profile[shift] = None
                    continue
                x = stance_vals[k:]
                y = ice_vals[:len(ice_vals) - k]
            
            r = _pearson_correlation(x, y, min_n=12)
            profile[shift] = r
        except Exception:
            profile[shift] = None
    
    valid_shifts = [(shift, r) for shift, r in profile.items() if r is not None]
    if not valid_shifts:
        return {"best_shift": None, "best_corr": None, "profile": profile, "causal_direction": None}
    
    # Find shift with strongest negative correlation (anticipation)
    best_shift, best_corr = min(valid_shifts, key=lambda x: x[1])
    
    if best_shift > 0:
        causal_direction = "H1_support"  # Stance shift precedes explicitness
    elif best_shift < 0:
        causal_direction = "H2_support"  # Explicitness precedes stance shift
    else:
        causal_direction = "synchronous"
    
    return {
        "best_shift": int(best_shift),
        "best_corr": float(best_corr),
        "profile": profile,
        "causal_direction": causal_direction
    }


# ============================================================================
# STATISTICAL META-ANALYSIS & VISUALIZATION
# ============================================================================

def fisher_z_transform_correlations(correlations: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Apply Fisher Z-transform for meta-analytic aggregation of correlations."""
    valid_corrs = [c for c in correlations if c is not None and not np.isnan(c) and abs(c) < 1.0]
    
    if len(valid_corrs) == 0:
        return None, None, None
    
    z_scores = [math.atanh(min(max(r, -0.999), 0.999)) for r in valid_corrs]
    mean_z = np.mean(z_scores)
    std_z = np.std(z_scores, ddof=1) if len(z_scores) > 1 else 0
    
    mean_r = math.tanh(mean_z)
    ci_lower = math.tanh(mean_z - 1.96 * (std_z / np.sqrt(len(valid_corrs))))
    ci_upper = math.tanh(mean_z + 1.96 * (std_z / np.sqrt(len(valid_corrs))))
    
    return mean_r, ci_lower, ci_upper


def plot_offset_correlation_profile(
    offset_to_correlations: Dict[int, List[float]],
    output_path: Path,
    metric_type: str
) -> None:
    """Visualize mean lagged correlations across temporal offsets."""
    offsets = []
    means = []
    ci_lower_vals = []
    ci_upper_vals = []
    
    for offset in sorted(offset_to_correlations.keys()):
        correlations = offset_to_correlations[offset]
        mean_r, ci_lower, ci_upper = fisher_z_transform_correlations(correlations)
        
        if mean_r is not None:
            offsets.append(offset)
            means.append(mean_r)
            ci_lower_vals.append(ci_lower)
            ci_upper_vals.append(ci_upper)
    
    if len(offsets) == 0:
        logger.error("No valid correlations found for any offset")
        return

    offsets = np.array(offsets)
    means = np.array(means)
    ci_lower_vals = np.array(ci_lower_vals)
    ci_upper_vals = np.array(ci_upper_vals)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.errorbar(offsets, means, 
                yerr=[means - ci_lower_vals, ci_upper_vals - means],
                fmt='o-', capsize=5, capthick=2, elinewidth=1.5,
                markersize=6, color='#2E5090', ecolor='#D72638', 
                markerfacecolor='#2E5090', markeredgecolor='black', 
                markeredgewidth=0.5, label='Mean correlation')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Offset (turns)\n(Negative: Iceberg precedes Stance → H₂; Positive: Stance precedes Iceberg → H₁)', fontsize=12)
    ax.set_ylabel('Mean Lagged Correlation (Fisher Z)', fontsize=12)
    ax.set_title(f'Lagged Correlation vs Offset ({metric_type} metric)\nStance vs Iceberg', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(min(offsets), max(offsets)+1, 5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved offset correlation plot to {output_path}")


def perform_meta_analysis(cors: np.ndarray) -> Dict[str, Any]:
    """Conduct Fisher Z-transform meta-analysis across episodes."""
    cors = cors[np.isfinite(cors) & (np.abs(cors) < 0.999)]
    n = len(cors)
    
    if n < 5:
        return {
            "mean_r": None, "ci_lower": None, "ci_upper": None,
            "z_stat": None, "p_two_tailed": None, "p_one_tailed": None, "n": n
        }
    
    z_scores = np.arctanh(cors)
    mean_z = np.mean(z_scores)
    se_z = np.std(z_scores, ddof=1) / math.sqrt(n)
    
    ci_z_lower = mean_z - 1.96 * se_z
    ci_z_upper = mean_z + 1.96 * se_z
    ci_r_lower = math.tanh(ci_z_lower)
    ci_r_upper = math.tanh(ci_z_upper)
    mean_r = math.tanh(mean_z)
    
    z_stat = mean_z / se_z
    p_two_tailed = 1 - (math.erf(abs(z_stat) / math.sqrt(2)))
    p_one_tailed = 0.5 * (1 + math.erf(z_stat / math.sqrt(2)))
    
    return {
        "mean_r": float(mean_r),
        "ci_lower": float(ci_r_lower),
        "ci_upper": float(ci_r_upper),
        "z_stat": float(z_stat),
        "p_two_tailed": float(p_two_tailed),
        "p_one_tailed": float(p_one_tailed),
        "n": int(n)
    }


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gricean Repair Mechanism in Natural Dialogue: Temporal Causality Analysis"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="prop", choices=["prop", "ratio", "log_ratio"])
    parser.add_argument("--min_turns", type=int, default=50)
    parser.add_argument("--max_shift", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_iceberg/results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("GRICEAN REPAIR MECHANISM ANALYSIS")
    logger.info("="*80)
    logger.info(f"Data directory     : {args.data_dir}")
    logger.info(f"Iceberg metric     : {args.metric}")
    logger.info(f"Max temporal shift : ±{args.max_shift} turns")
    logger.info(f"Min turns/episode  : {args.min_turns}")
    logger.info(f"Output directory   : {args.output_dir}")
    logger.info(f"Start time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    episode_paths = sorted(Path(args.data_dir).glob("*.json"))
    logger.info(f"Found {len(episode_paths)} JSON files")
    
    results = []
    failures = []
    offset_to_correlations = {offset: [] for offset in range(-args.max_shift, args.max_shift + 1)}
    
    for path in tqdm(episode_paths, desc="Processing episodes"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                turns = json.load(f)
            
            df = process_episode_turns(
                turns, 
                metric_type=args.metric,
                min_turns=args.min_turns
            )
            
            if df is None or len(df) < args.min_turns:
                failures.append((path.stem, "quality_control_failed"))
                continue
            
            causality_result = analyze_temporal_causality(
                df,
                max_shift=args.max_shift
            )
            
            if causality_result["best_shift"] is None:
                failures.append((path.stem, "no_valid_shift_found"))
                continue
            
            results.append({
                "episode_id": path.stem,
                "n_turns": len(df),
                "best_shift": causality_result["best_shift"],
                "best_corr": causality_result["best_corr"],
                "causal_direction": causality_result["causal_direction"],
                "mean_iceberg": df["iceberg_norm"].mean(),
                "mean_stance": df["stance_5pt"].mean(),
            })
            
            profile = causality_result["profile"]
            for offset in range(-args.max_shift, args.max_shift + 1):
                corr = profile.get(offset)
                if corr is not None and not np.isnan(corr):
                    offset_to_correlations[offset].append(corr)
                    
        except Exception as e:
            failures.append((path.stem, f"exception: {str(e)[:50]}"))
            continue
    
    n_total = len(episode_paths)
    n_success = len(results)
    n_fail = len(failures)
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Successfully processed : {n_success} / {n_total} episodes ({n_success/n_total*100:.1f}%)")
    logger.info(f"  Failed (QC/filtering)  : {n_fail} episodes")
    
    if n_success < 20:
        raise RuntimeError(f"Insufficient valid episodes ({n_success}) for meta-analysis")
    
    df_agg = pd.DataFrame(results)
    meta_result = perform_meta_analysis(df_agg["best_corr"].to_numpy())
    
    plot_offset_correlation_profile(
        offset_to_correlations, 
        output_path / f"all_offsets_correlation_{args.metric}.png",
        args.metric
    )
    
    df_agg.to_csv(output_path / f"episode_results_{args.metric}.csv", index=False)
    
    correlations_df = pd.DataFrame.from_dict(offset_to_correlations, orient='index').T
    correlations_df.columns.name = 'offset'
    correlations_df.to_csv(output_path / f"raw_correlations_by_offset_{args.metric}.csv")
    
    report_data = {
        "meta_analysis": {
            "mean_correlation_r": round(meta_result['mean_r'], 4) if meta_result['mean_r'] is not None else None,
            "ci_95_lower": round(meta_result['ci_lower'], 4) if meta_result['ci_lower'] is not None else None,
            "ci_95_upper": round(meta_result['ci_upper'], 4) if meta_result['ci_upper'] is not None else None,
            "z_statistic": round(meta_result['z_stat'], 4) if meta_result['z_stat'] is not None else None,
            "p_value_two_tailed": meta_result['p_two_tailed'],
            "p_value_one_tailed": meta_result['p_one_tailed'],
            "n_episodes": meta_result['n'],
            "effect_size": (
                "large" if abs(meta_result['mean_r']) > 0.5 else 
                "medium" if abs(meta_result['mean_r']) > 0.3 else "small"
            ) if meta_result['mean_r'] is not None else "undefined"
        },
        "causal_direction_analysis": {
            "mean_optimal_shift": round(df_agg["best_shift"].mean(), 2),
            "median_optimal_shift": round(df_agg["best_shift"].median(), 2),
            "direction_distribution": {
                "H1_support": int((df_agg["causal_direction"] == "H1_support").sum()),
                "H2_support": int((df_agg["causal_direction"] == "H2_support").sum()),
                "synchronous": int((df_agg["causal_direction"] == "synchronous").sum()),
            },
            "percent_H1_support": round((df_agg["causal_direction"] == "H1_support").mean() * 100, 1),
            "percent_H2_support": round((df_agg["causal_direction"] == "H2_support").mean() * 100, 1),
        },
        "offset_correlation_analysis": {
            "best_offset": min(
                (offset for offset in offset_to_correlations if offset_to_correlations[offset]),
                key=lambda o: fisher_z_transform_correlations(offset_to_correlations[o])[0] or float('inf')
            ) if any(offset_to_correlations.values()) else None,
        },
        "methodology": {
            "metric_type": args.metric,
            "min_turns_per_episode": args.min_turns,
            "max_temporal_shift": args.max_shift,
            "processing_date": start_time.isoformat(),
        }
    }
    
    with open(output_path / f"meta_analysis_{args.metric}.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    p_val = meta_result['p_one_tailed'] if meta_result['p_one_tailed'] is not None else float('inf')
    p_display = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
    mean_shift = df_agg["best_shift"].mean()
    pct_h1 = (df_agg["causal_direction"] == "H1_support").mean() * 100
    pct_h2 = (df_agg["causal_direction"] == "H2_support").mean() * 100
    
    print("\n" + "="*80)
    print("GRICEAN CAUSALITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Episodes analyzed':.<35} {meta_result['n']:,}")
    print(f"{'Mean correlation (r)':.<35} {meta_result['mean_r']:.3f} [{meta_result['ci_lower']:.3f}, {meta_result['ci_upper']:.3f}]")
    print(f"{'Mean optimal shift':.<35} {mean_shift:.2f} turns")
    print(f"{'% supporting H1':.<35} {pct_h1:.1f}%  ← DISAGREEMENT → EXPLICIT REPAIR")
    print(f"{'% supporting H2':.<35} {pct_h2:.1f}%  ← EXPLICITNESS → DISAGREEMENT")
    print(f"{'One-tailed p-value':.<35} {p_display}")
    print("="*80)

if __name__ == "__main__":
    main()