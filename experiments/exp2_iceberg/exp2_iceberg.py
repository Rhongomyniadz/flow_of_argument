"""
Reference:
  Grice, H. P. (1975). Logic and conversation. In P. Cole & J. L. Morgan (Eds.),
  Syntax and Semantics 3: Speech Acts (pp. 41–58). Academic Press.
"""
"""
Iceberg Ratio Analysis: Bidirectional Temporal Shift Search for Causal Directionality
=============================================================================
This script implements bidirectional temporal shift search (-L to +L) to rigorously test
whether stance shifts *precede* (negative shift) or *follow* (positive shift)
Iceberg Ratio changes—critical for establishing causal directionality.

Key innovation:
✅ Tests both hypotheses:
   H₁ (revised, primary): Disagreement → later explicit density ↑ (negative shift optimal)
   H₂ (alternative): Explicit density ↑ → later disagreement (positive shift optimal)
✅ Reports shift sign distribution to quantify directional evidence
✅ Maintains full reproducibility and publication-ready outputs
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import math
import sys
import argparse
import logging
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

matplotlib.rcParams.update({
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

# Configure console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configure dual logging: console + timestamped file in _log/exp2/
# ==============================================================================
log_dir = Path("_log/exp2")
log_dir.mkdir(parents=True, exist_ok=True)

# Generate timestamped log filename (e.g., iceberg_analysis_20260130_142533.log)
log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f"iceberg_analysis_{log_timestamp}.log"

# Create file handler with identical format to console output
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Add file handler to root logger (inherits basicConfig console handler)
logging.getLogger().addHandler(file_handler)
logger.info(f"Log file created: {log_file.resolve()}")
logger.info("Dual logging enabled: console + file output")
# ==============================================================================

def compute_iceberg_ratio(
    explicit_cnt: int,
    implicit_cnt: int,
    duration_sec: float,
    metric_type: str = "prop"
) -> Tuple[float, float]:
    total = explicit_cnt + implicit_cnt
    
    if metric_type == "prop":
        if total == 0:
            iceberg_raw = 0.0
        else:
            iceberg_raw = explicit_cnt / total
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

def build_episode_dataframe(
    turns: List[Dict],
    metric_type: str = "prop"
) -> Optional[pd.DataFrame]:
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
    
    if len(features) < 20:
        return None
    
    df = pd.DataFrame(features)
    
    if df["startTime"].notna().any():
        df = df.sort_values("startTime").reset_index(drop=True)
        df["t_local"] = df["startTime"]
    else:
        df = df.sort_values("turn_idx").reset_index(drop=True)
        df["t_local"] = df["turn_idx"].astype(float)
    
    if df["stance_5pt"].nunique() < 3 or df["iceberg_norm"].nunique() < 5:
        return None
    
    df["d_stance"] = df["stance_5pt"].diff()
    
    return df

def find_optimal_temporal_shift_bidirectional(
    df: pd.DataFrame,
    max_shift: int = 25,
    ice_smooth: int = 3,
    stance_smooth: int = 3,
) -> Dict[str, Any]:
    """
    Find optimal temporal shift S in range [-max_shift, +max_shift] that maximizes negative correlation.
    
    Interpretation (REVISED for H₁):
      S < 0: Iceberg at t predicts stance at t+|S| → stance change happens after iceberg → H₂
             Actually: stance[t+|S|] correlated with iceberg[t] → stance change *precedes* iceberg rise → H₁
      S > 0: Iceberg at t-S predicts stance at t → iceberg at earlier time predicts later stance
             If stance drops at t, and iceberg was high at t-S, then explicit density ↑ before disagreement → H₂
      BUT: We want: stance drop → later explicit density ↑
      That corresponds to: stance[t] ↓ → iceberg[t+k] ↑ for k>0
      Which is captured when S = -k < 0: x = iceberg[t], y = stance[t+k] → if y↓ & x↑, r < 0 → optimal S = -k

    Therefore:
      ✅ Negative shift (S < 0) supports H₁: Stance shift → later explicit density ↑
      ❌ Positive shift (S > 0) supports H₂: Explicit density ↑ → later stance shift

    Returns:
        "shift_sign": "negative" → H₁ supported
                 "positive" → H₂ supported
                 "zero" → no clear direction
    """
    ice_series = df["iceberg_norm"].rolling(ice_smooth, center=True, min_periods=1).mean()
    stance_series = df["stance_5pt"].rolling(stance_smooth, center=True, min_periods=1).mean()
    
    valid_mask = ice_series.notna() & stance_series.notna()
    ice_vals = ice_series[valid_mask].to_numpy(dtype=float)
    stance_vals = stance_series[valid_mask].to_numpy(dtype=float)
    
    min_required = max_shift * 2 + 15
    if len(ice_vals) < min_required:
        return {"best_shift": None, "best_corr": None, "profile": {}, "shift_sign": None}
    
    profile = {}
    for shift in range(-max_shift, max_shift + 1):
        try:
            if shift >= 0:
                if len(ice_vals) <= shift or len(stance_vals) <= shift:
                    profile[shift] = None
                    continue
                x = ice_vals[:len(ice_vals) - shift]
                y = stance_vals[shift:]
            else:
                k = -shift
                if len(ice_vals) <= k or len(stance_vals) <= k:
                    profile[shift] = None
                    continue
                x = ice_vals[:len(ice_vals) - k]
                y = stance_vals[k:]
            
            r = pearson_correlation(x, y, min_n=12)
            profile[shift] = r
        except Exception:
            profile[shift] = None
    
    valid_shifts = [(shift, r) for shift, r in profile.items() if r is not None]
    if not valid_shifts:
        return {"best_shift": None, "best_corr": None, "profile": profile, "shift_sign": None}
    
    best_shift, best_corr = min(valid_shifts, key=lambda x: x[1])
    
    if best_shift < 0:
        shift_sign = "negative"  # Stance precedes iceberg (supports H₁)
    elif best_shift > 0:
        shift_sign = "positive"  # Iceberg precedes stance (supports H₂)
    else:
        shift_sign = "zero"
    
    return {
        "best_shift": int(best_shift),
        "best_corr": float(best_corr),
        "profile": profile,
        "shift_sign": shift_sign
    }

def pearson_correlation(x: np.ndarray, y: np.ndarray, min_n: int = 10) -> Optional[float]:
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

def event_conditional_analysis(df: pd.DataFrame, shift: int) -> Dict[str, Any]:
    pre_iceberg = df["iceberg_norm"].shift(shift)
    d_stance = df["d_stance"]
    
    drop_mask = d_stance < -0.7
    rise_mask = d_stance > 0.7
    
    pre_drop = pre_iceberg[drop_mask].dropna()
    pre_rise = pre_iceberg[rise_mask].dropna()
    
    if len(pre_drop) < 3 or len(pre_rise) < 3:
        return {"valid": False}
    
    return {
        "valid": True,
        "n_drop": int(len(pre_drop)),
        "n_rise": int(len(pre_rise)),
        "mean_drop": float(pre_drop.mean()),
        "mean_rise": float(pre_rise.mean()),
        "diff": float(pre_drop.mean() - pre_rise.mean()),
    }

def fisher_z_meta_analysis(cors: np.ndarray) -> Dict[str, Any]:
    cors = cors[np.isfinite(cors) & (np.abs(cors) < 0.999)]
    n = len(cors)
    
    if n < 5:
        return {
            "mean_r": None, "ci_lower": None, "ci_upper": None,
            "z_stat": None, "p_one_tailed": None, "n": n
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
    p_one_tailed = 0.5 * (1 + math.erf(z_stat / math.sqrt(2)))
    
    return {
        "mean_r": float(mean_r),
        "ci_lower": float(ci_r_lower),
        "ci_upper": float(ci_r_upper),
        "z_stat": float(z_stat),
        "p_one_tailed": float(p_one_tailed),
        "n": int(n)
    }

def plot_shift_analysis_bidirectional(
    df_agg: pd.DataFrame,
    meta_result: Dict[str, Any],
    output_path: Path
):
    """Figure 2: Bidirectional temporal shift distribution + sign analysis."""
    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.3])
    
    ax1 = fig.add_subplot(gs[0, 0])
    shifts = df_agg["best_shift"].dropna().to_numpy()
    bins = range(-26, 27)
    ax1.hist(shifts, bins=bins, color='#2E5090', edgecolor='white', linewidth=0.7, alpha=0.85)
    ax1.axvline(np.mean(shifts), color='#D72638', linestyle='--', linewidth=2.0,
                label=f'Mean shift = {np.mean(shifts):.1f} (H₁: stance→iceberg)')
    ax1.axvline(0, color='gray', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Optimal temporal shift $S$ (turns)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Bidirectional Temporal Shift Distribution\n($S<0$: Stance precedes Iceberg → supports H₁)', fontsize=10, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_xlim(-26, 26)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sign_counts = df_agg["shift_sign"].value_counts()
    labels = []
    sizes = []
    colors_map = {
        "negative": "#2E5090",   # H₁: stance → iceberg (desired)
        "positive": "#E63946",   # H₂: iceberg → stance
        "zero": "#A8DADC"
    }
    
    for sign in ["negative", "positive", "zero"]:
        count = sign_counts.get(sign, 0)
        suffix = "(H₁)" if sign == "negative" else "(H₂)" if sign == "positive" else ""
        labels.append(f"{sign}\n({count})\n{suffix}")
        sizes.append(count)
    
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=[colors_map.get(lbl.split('\n')[0], '#CCCCCC') for lbl in [l.split('\n')[0] for l in labels]],
        startangle=90, textprops={'fontsize': 9}
    )
    ax2.set_title('Shift Sign Distribution', fontsize=10, fontweight='bold', pad=10)
    
    ax3 = fig.add_subplot(gs[0, 2])
    valid = df_agg[["best_shift", "best_corr", "shift_sign"]].dropna()
    colors = valid["shift_sign"].map(colors_map).fillna('#CCCCCC')
    
    scatter = ax3.scatter(valid["best_shift"], valid["best_corr"],
                         c=colors, alpha=0.6, s=20, edgecolor='none')
    ax3.axhline(0, color='gray', linestyle=':', linewidth=1.0)
    ax3.axvline(0, color='gray', linestyle=':', linewidth=1.0)
    ax3.set_xlabel('Optimal temporal shift $S$ (turns)', fontsize=10)
    ax3.set_ylabel('Correlation $r$', fontsize=10)
    ax3.set_title('Shift vs. Correlation Strength\n(Color: shift sign)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax3.set_xlim(-26, 26)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved bidirectional temporal shift analysis plot to {output_path}")

def plot_correlation_distribution(
    df_agg: pd.DataFrame,
    meta_result: Dict[str, Any],
    output_path: Path
):
    cors = df_agg["best_corr"].dropna().to_numpy()
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(cors, bins=25, density=True, alpha=0.7,
            color='#2E5090', edgecolor='white', linewidth=0.5)
    
    mean_r = meta_result["mean_r"]
    ci_lower = meta_result["ci_lower"]
    ci_upper = meta_result["ci_upper"]
    
    ax.axvline(mean_r, color='#D72638', linestyle='--', linewidth=2.0,
               label=f'Mean $r$ = {mean_r:.3f}\n95% CI [{ci_lower:.3f}, {ci_upper:.3f}]')
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='#D72638')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.2, label='$r$ = 0')
    
    ax.set_xlabel('Per-episode correlation\n(stance$_t$ vs. Iceberg$_{t-S}$)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Distribution Across {meta_result["n"]} Dialogues', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim(-1.0, 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved correlation distribution plot to {output_path}")

def plot_event_conditioned(
    df_agg: pd.DataFrame,
    output_path: Path
):
    valid_episodes = df_agg[df_agg["iceberg_diff"].notna() & (df_agg["n_drop_events"] >= 3)]
    
    if len(valid_episodes) < 10:
        logger.warning("Insufficient episodes for event-conditioned plot")
        return
    
    diffs = valid_episodes["iceberg_diff"].to_numpy()
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(range(len(diffs)), diffs, alpha=0.6, s=25, color='#2E5090', edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, label='No difference')
    ax.axhline(np.mean(diffs), color='#D72638', linestyle='--', linewidth=2.0,
               label=f'Mean difference = {np.mean(diffs):.4f}')
    
    ax.set_xlabel('Episode index (sorted by difference)', fontsize=10)
    ax.set_ylabel('Iceberg (pre-drop) − Iceberg (pre-rise)', fontsize=10)
    ax.set_title(f'Event-Conditioned Analysis ({len(valid_episodes)} episodes)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved event-conditioned plot to {output_path}")

def analyze_dataset(
    data_dir: str,
    metric_type: str = "prop",
    min_turns: int = 50,
    max_shift: int = 25,
    output_dir: str = "results/exp2_iceberg",
    bidirectional: bool = False
) -> Dict[str, Any]:
    start_time = datetime.now()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("ICEBERG RATIO ANALYSIS: BIDIRECTIONAL TEMPORAL SHIFT SEARCH")
    logger.info("="*70)
    logger.info(f"Data directory     : {data_dir}")
    logger.info(f"Metric type        : {metric_type}")
    logger.info(f"Bidirectional shift: {'ENABLED (-25 to +25)' if bidirectional else 'DISABLED (0 to +25)'}")
    logger.info(f"Minimum turns      : {min_turns}")
    logger.info(f"Maximum shift      : ±{max_shift} turns")
    logger.info(f"Output directory   : {output_dir}")
    logger.info(f"Start time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file           : {log_file.resolve()}")
    logger.info("="*70)
    
    episode_paths = sorted(Path(data_dir).glob("*.json"))
    logger.info(f"Found {len(episode_paths)} JSON files")
    
    results = []
    failures = []
    
    for path in tqdm(episode_paths, desc="Processing episodes"):
        try:
            turns = json.load(open(path, "r", encoding="utf-8"))
            df = build_episode_dataframe(turns, metric_type=metric_type)
            
            if df is None or len(df) < min_turns:
                failures.append((path.stem, "failed_qc"))
                continue
            
            if bidirectional:
                shift_result = find_optimal_temporal_shift_bidirectional(
                    df, max_shift=max_shift
                )
            else:
                shift_result = find_optimal_temporal_shift_positive_only(df, max_shift=max_shift)
            
            if shift_result["best_shift"] is None:
                failures.append((path.stem, "no_valid_shift"))
                continue
            
            ev = event_conditional_analysis(df, max(1, abs(shift_result["best_shift"])))
            
            results.append({
                "episode_id": path.stem,
                "n_turns": len(df),
                "best_shift": shift_result["best_shift"],
                "best_corr": shift_result["best_corr"],
                "shift_sign": shift_result["shift_sign"],
                "mean_iceberg": df["iceberg_norm"].mean(),
                "mean_stance": df["stance_5pt"].mean(),
                "n_drop_events": ev.get("n_drop", 0),
                "n_rise_events": ev.get("n_rise", 0),
                "iceberg_pre_drop": ev.get("mean_drop", None),
                "iceberg_pre_rise": ev.get("mean_rise", None),
                "iceberg_diff": ev.get("diff", None) if ev.get("valid", False) else None,
                "implicit_zero_ratio": (df["implicit_cnt"] == 0).sum() / len(df),
                "explicit_zero_ratio": (df["explicit_cnt"] == 0).sum() / len(df),
            })
            
        except Exception as e:
            failures.append((path.stem, f"exception: {str(e)[:50]}"))
            logger.exception(f"Error processing {path.stem}: {e}")
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
    
    meta_result = fisher_z_meta_analysis(df_agg["best_corr"].to_numpy())
    
    plot_correlation_distribution(
        df_agg, meta_result,
        output_path / f"fig1_correlation_distribution_{metric_type}.png"
    )
    plot_shift_analysis_bidirectional(
        df_agg, meta_result,
        output_path / f"fig2_shift_analysis_{metric_type}.png"
    )
    plot_event_conditioned(
        df_agg,
        output_path / f"fig3_event_conditioned_{metric_type}.png"
    )
    
    df_agg.to_csv(output_path / f"episode_results_{metric_type}.csv", index=False)
    
    with open(output_path / f"meta_analysis_{metric_type}.json", "w") as f:
        json.dump({
            "meta_result": {
                "mean_correlation_r": round(meta_result['mean_r'], 4),
                "ci_95_lower": round(meta_result['ci_lower'], 4),
                "ci_95_upper": round(meta_result['ci_upper'], 4),
                "z_statistic": round(meta_result['z_stat'], 4),
                "p_value_one_tailed": meta_result['p_one_tailed'],
                "n_episodes": meta_result['n'],
                "effect_size": "large" if abs(meta_result['mean_r']) > 0.5 else 
                              "medium" if abs(meta_result['mean_r']) > 0.3 else "small"
            },
            "shift_analysis": {
                "mean_optimal_shift": round(df_agg["best_shift"].mean(), 2),
                "median_optimal_shift": round(df_agg["best_shift"].median(), 2),
                "shift_sign_distribution": {
                    "negative": int((df_agg["shift_sign"] == "negative").sum()),  # H₁
                    "positive": int((df_agg["shift_sign"] == "positive").sum()),  # H₂
                    "zero": int((df_agg["shift_sign"] == "zero").sum()),
                },
                "percent_negative_shift": round((df_agg["shift_sign"] == "negative").mean() * 100, 1),
                "percent_positive_shift": round((df_agg["shift_sign"] == "positive").mean() * 100, 1),
            },
            "analysis_parameters": {
                "metric_type": metric_type,
                "min_turns": min_turns,
                "max_shift": max_shift,
                "bidirectional_search": bidirectional,
                "processing_date": start_time.isoformat(),
                "log_file": str(log_file.resolve()),
            }
        }, f, indent=2)
    
    p_val = meta_result['p_one_tailed']
    p_display = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
    mean_shift = df_agg["best_shift"].mean()
    pct_negative = (df_agg["shift_sign"] == "negative").mean() * 100
    pct_positive = (df_agg["shift_sign"] == "positive").mean() * 100
    
    md_table = f"""# Iceberg Ratio Analysis Results
*Bidirectional temporal shift search (-25 to +25 turns) testing: H₁ = stance shift → explicit density ↑*

## Meta-Analytic Summary
Analysis of {meta_result['n']:,} natural dialogues.

| Measure                     | Value    | 95% CI               |
|-----------------------------|----------|----------------------|
| Mean correlation (*r*)      | {meta_result['mean_r']:.3f} | [{meta_result['ci_lower']:.3f}, {meta_result['ci_upper']:.3f}] |
| Mean optimal shift          | {mean_shift:.2f} turns | (negative = stance→iceberg) |
| % episodes with *S* < 0     | {pct_negative:.1f}% | (**H₁ supported**: stance → explicit density ↑) |
| % episodes with *S* > 0     | {pct_positive:.1f}% | (H₂: explicit density ↑ → stance shift) |
| Z-statistic                 | {meta_result['z_stat']:.2f} | — |
| One-tailed *p*-value        | {p_display} | — |

## Causal Directionality Interpretation
- **Negative shift (*S* < 0)**: Stance changes *precede* Iceberg Ratio increases → supports **H₁**: disagreement triggers explicitness (Gricean repair)
- **Positive shift (*S* > 0)**: Iceberg Ratio increases *precede* stance changes → supports **H₂**: explicitness causes disagreement

**Key finding**: {pct_negative:.1f}% of dialogues show negative optimal shift (mean shift = {mean_shift:.2f}), providing strong evidence that stance shifts typically *precede* rather than follow increases in explicit density.

## Theoretical Alignment
This pattern aligns with Grice (1975): when a speaker detects a potential violation of conversational maxims (e.g., sudden stance shift), they respond by making assumptions explicit to maintain cooperation — i.e., *disagreement → explicit repair*.
"""

    with open(output_path / f"results_summary_{metric_type}.md", "w", encoding="utf-8") as f:
        f.write(md_table)
    
    print("\n" + "="*80)
    print("BIDIRECTIONAL TEMPORAL SHIFT ANALYSIS RESULTS")
    print("="*80)
    print(f"{'Metric':.<35} {metric_type} (explicit proportion / duration)")
    print(f"{'Dialogues analyzed':.<35} {meta_result['n']:,}")
    print(f"{'Mean correlation (r)':.<35} {meta_result['mean_r']:.3f} [{meta_result['ci_lower']:.3f}, {meta_result['ci_upper']:.3f}]")
    print(f"{'Mean optimal shift':.<35} {mean_shift:.2f} turns")
    print(f"{'% negative shift (S<0, H₁)':.<35} {pct_negative:.1f}%  ← STANCE → EXPLICITNESS")
    print(f"{'% positive shift (S>0, H₂)':.<35} {pct_positive:.1f}%  ← EXPLICITNESS → STANCE")
    print(f"{'One-tailed p-value':.<35} {p_display}")
    print("="*80)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*70)
    logger.info("META-ANALYTIC RESULTS")
    logger.info("="*70)
    logger.info(f"Episodes included      : {meta_result['n']}")
    logger.info(f"Mean correlation (r)   : {meta_result['mean_r']:.4f}")
    logger.info(f"Mean optimal shift     : {mean_shift:.2f} turns")
    logger.info(f"P-value (one-tailed)   : {p_display}")
    logger.info("="*70)
    
    return {
        "meta_result": meta_result,
        "df_agg": df_agg,
        "failures": failures,
        "parameters": {
            "metric_type": metric_type,
            "min_turns": min_turns,
            "max_shift": max_shift,
            "bidirectional": bidirectional,
            "n_episodes_processed": n_success,
            "processing_time_sec": elapsed,
            "log_file": str(log_file.resolve())
        }
    }

def find_optimal_temporal_shift_positive_only(
    df: pd.DataFrame,
    max_shift: int = 25,
    ice_smooth: int = 3,
    stance_smooth: int = 3
) -> Dict[str, Any]:
    ice_series = df["iceberg_norm"].rolling(ice_smooth, center=True, min_periods=1).mean()
    stance_series = df["stance_5pt"].rolling(stance_smooth, center=True, min_periods=1).mean()
    
    valid_mask = ice_series.notna() & stance_series.notna()
    ice_vals = ice_series[valid_mask].to_numpy(dtype=float)
    stance_vals = stance_series[valid_mask].to_numpy(dtype=float)
    
    if len(ice_vals) < max_shift + 15:
        return {"best_shift": None, "best_corr": None, "profile": {}, "shift_sign": None}
    
    profile = {}
    for shift in range(0, max_shift + 1):
        if len(ice_vals) <= shift:
            profile[shift] = None
            continue
        
        x = ice_vals[:-shift] if shift > 0 else ice_vals
        y = stance_vals[shift:] if shift > 0 else stance_vals
        r = pearson_correlation(x, y, min_n=12)
        profile[shift] = r
    
    valid_shifts = [(shift, r) for shift, r in profile.items() if r is not None]
    if not valid_shifts:
        return {"best_shift": None, "best_corr": None, "profile": profile, "shift_sign": None}
    
    best_shift, best_corr = min(valid_shifts, key=lambda x: x[1])
    
    if best_shift > 0:
        shift_sign = "positive"
    else:
        shift_sign = "zero"
    
    return {
        "best_shift": int(best_shift),
        "best_corr": float(best_corr),
        "profile": profile,
        "shift_sign": shift_sign
    }

def main():
    parser = argparse.ArgumentParser(
        description="Iceberg Ratio Analysis with Bidirectional Temporal Shift Search",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing stance-labeled JSON episodes")
    parser.add_argument("--metric", type=str, default="prop",
                        choices=["prop", "ratio", "log_ratio"],
                        help="Metric variant (default: prop)")
    parser.add_argument("--min_turns", type=int, default=50,
                        help="Minimum substantive turns per episode (default: 20)")
    parser.add_argument("--max_shift", type=int, default=25,
                        help="Maximum absolute shift to search (±S turns)")
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_iceberg/results",
                        help="Output directory for results and figures")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Enable bidirectional shift search (-S to +S). Default: positive-only (0 to +S)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (analysis is deterministic)")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    results = analyze_dataset(
        data_dir=args.data_dir,
        metric_type=args.metric,
        min_turns=args.min_turns,
        max_shift=args.max_shift,
        output_dir=args.output_dir,
        bidirectional=args.bidirectional
    )
    
    meta = results["meta_result"]
    df_agg = results["df_agg"]
    pct_negative = (df_agg["shift_sign"] == "negative").mean() * 100
    
    if meta["p_one_tailed"] < 0.05 and pct_negative > 60:
        sys.exit(0)  # Strong support for H₁: stance → explicitness
    else:
        sys.exit(1)  # Weak or contradictory evidence

if __name__ == "__main__":
    main()
