import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ============================================================================
# Logging & Global Plot Settings
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard ACL-compliant plotting parameters
PLOT_PARAMS = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.fontsize": 9,
}
plt.rcParams.update(PLOT_PARAMS)


# ============================================================================
# Statistical Utilities (Fisher-Z)
# ============================================================================
def fisher_z_transform(r: float) -> float:
    """Applies the Fisher Z-transformation to a Pearson correlation coefficient."""
    eps = 1e-10
    # Clip r to avoid math domain errors
    r = max(min(r, 1 - eps), -1 + eps)
    return 0.5 * math.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """Applies the inverse Fisher Z-transformation."""
    return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)


def aggregate_correlations_fisher_z(
    corrs: List[float], 
    confidence_level: float = 0.95
) -> Tuple[float, Optional[float], Optional[float], int]:
    """
    Aggregates a list of correlation coefficients using the Fisher Z-transform.
    Returns: (r_aggregated, ci_lower, ci_upper, n_valid)
    """
    valid_corrs = [r for r in corrs if np.isfinite(r)]
    n_valid = len(valid_corrs)
    
    if n_valid == 0:
        return float("nan"), None, None, 0
    
    z_values = [fisher_z_transform(r) for r in valid_corrs]
    z_mean = np.mean(z_values)
    z_std = np.std(z_values, ddof=1) if n_valid > 1 else 0.0
    
    r_aggregated = inverse_fisher_z(z_mean)
    
    ci_lower, ci_upper = None, None
    if n_valid > 1 and confidence_level > 0:
        z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        z_se = z_std / math.sqrt(n_valid)
        ci_lower = inverse_fisher_z(z_mean - z_crit * z_se)
        ci_upper = inverse_fisher_z(z_mean + z_crit * z_se)
    
    return float(r_aggregated), ci_lower, ci_upper, n_valid


def safe_pearson_correlation(x: np.ndarray, y: np.ndarray, min_n: int = 12) -> Optional[float]:
    """Calculates Pearson correlation with safety checks for NaN and variance."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < min_n:
        return None
    if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
        return None
        
    return float(np.corrcoef(x_clean, y_clean)[0, 1])


# ============================================================================
# Data Processing
# ============================================================================
def compute_iceberg_ratio(explicit: int, implicit: int, duration: float, metric_type: str) -> float:
    """Computes the normalized 'Iceberg' ratio."""
    total = explicit + implicit
    duration = max(duration, 0.1)
    
    if total == 0:
        return 0.0
        
    if metric_type == "prop":
        return (explicit / total) / duration
    else:
        return (explicit / (implicit + 1e-6)) / duration


def preprocess_episode(data: List[Dict], metric_type: str, min_turns: int) -> Optional[pd.DataFrame]:
    """Parses raw JSON turn data into a structured DataFrame."""
    features = []
    
    for turn in data:
        if turn.get("turn_type_label") != "Substantive":
            continue
            
        stance = turn.get("stance_5pt")
        if stance is None:
            continue
        
        start_time = turn.get("startTime", 0)
        end_time = turn.get("endTime", 0.1)
        duration = max(float(end_time - start_time), 0.1)
        
        explicit_count = len(turn.get("explicit_propositions", []) or [])
        implicit_count = len(turn.get("assumptions", []) or [])
        
        iceberg_val = compute_iceberg_ratio(explicit_count, implicit_count, duration, metric_type)
        
        features.append({
            "speaker_id": turn.get("speaker_id"),
            "stance_5pt": float(stance),
            "iceberg_norm": iceberg_val,
            "startTime": start_time
        })
        
    if len(features) < min_turns:
        return None
        
    return pd.DataFrame(features).sort_values("startTime").reset_index(drop=True)


# ============================================================================
# Core Analysis Logic
# ============================================================================
def analyze_dyadic_interaction(
    df: pd.DataFrame, 
    max_shift: int,
    smooth_window: int,
    min_corr_samples: int
) -> Dict[int, Optional[float]]:
    """
    Calculates cross-correlation between speakers at various time shifts.
    """
    df = df.copy()
    # Apply smoothing
    df["iceberg_smooth"] = df["iceberg_norm"].rolling(smooth_window, center=True, min_periods=1).mean()
    df["stance_smooth"] = df["stance_5pt"].rolling(smooth_window, center=True, min_periods=1).mean()
    
    speakers = sorted(df["speaker_id"].unique())
    if len(speakers) != 2:
        return {}
        
    spk_a, spk_b = speakers[0], speakers[1]
    profile = {}
    
    # Calculate correlations for shifts [-max_shift, +max_shift]
    for shift in range(-max_shift, max_shift + 1):
        
        # 1. Forward: Speaker A Stance -> Speaker B Iceberg
        fwd_pairs = []
        for idx in range(len(df)):
            target_idx = idx + shift
            if 0 <= target_idx < len(df):
                curr = df.iloc[idx]
                target = df.iloc[target_idx]
                if curr["speaker_id"] == spk_a and target["speaker_id"] == spk_b:
                    fwd_pairs.append((curr["stance_smooth"], target["iceberg_smooth"]))
        
        # 2. Backward: Speaker B Stance -> Speaker A Iceberg
        bwd_pairs = []
        for idx in range(len(df)):
            target_idx = idx + shift
            if 0 <= target_idx < len(df):
                curr = df.iloc[idx]
                target = df.iloc[target_idx]
                if curr["speaker_id"] == spk_b and target["speaker_id"] == spk_a:
                    bwd_pairs.append((curr["stance_smooth"], target["iceberg_smooth"]))

        r_fwd = safe_pearson_correlation(
            np.array([p[0] for p in fwd_pairs]), np.array([p[1] for p in fwd_pairs]), min_n=min_corr_samples
        )
        r_bwd = safe_pearson_correlation(
            np.array([p[0] for p in bwd_pairs]), np.array([p[1] for p in bwd_pairs]), min_n=min_corr_samples
        )
        
        # Fisher-Z Averaging
        if r_fwd is not None and r_bwd is not None:
            z_avg = (fisher_z_transform(r_fwd) + fisher_z_transform(r_bwd)) / 2
            profile[shift] = inverse_fisher_z(z_avg)
        elif r_fwd is not None:
            profile[shift] = r_fwd
        elif r_bwd is not None:
            profile[shift] = r_bwd
        else:
            profile[shift] = None
            
    return profile


# ============================================================================
# Visualization
# ============================================================================
def plot_integrated_results(
    offset_data: Dict[int, List[float]],
    output_path: Path,
    metric_name: str,
    confidence_level: float = 0.95,
    plot_dpi: int = 300
) -> None:
    """Generates a publication-quality plot of the aggregated cross-correlations."""
    offsets = sorted(offset_data.keys())
    x_axis = np.array(offsets)
    
    means, ci_lows, ci_highs = [], [], []
    
    for shift in offsets:
        r, low, high, _ = aggregate_correlations_fisher_z(offset_data[shift], confidence_level)
        means.append(r)
        ci_lows.append(low if low is not None else np.nan)
        ci_highs.append(high if high is not None else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_axis, means, marker='o', markersize=5, linestyle='-', 
            linewidth=2, color='#1f77b4', label='Integrated Correlation')
    
    valid_mask = np.isfinite(ci_lows) & np.isfinite(ci_highs)
    if np.any(valid_mask):
        ax.fill_between(
            x_axis[valid_mask], 
            np.array(ci_lows)[valid_mask], 
            np.array(ci_highs)[valid_mask], 
            color='#1f77b4', alpha=0.2, 
            label=f'{int(confidence_level*100)}% Confidence Interval'
        )
    
    ax.axhline(0, color='black', linewidth=1, alpha=0.6)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
    ax.set_xlabel("Lag (Turns)\n(-) Other's Stance precedes Self Iceberg | (+) Self Stance precedes Other's Iceberg")
    ax.set_ylabel("Pearson Correlation (Fisher-Z Aggregated)")
    ax.set_title(f"Cross-Speaker Influence: Stance vs. Iceberg ({metric_name})")
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi)
    plt.close()
    logger.info(f"Generated plot at {output_path}")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Stance-Iceberg Cross-Correlation Analysis")
    
    # Input/Output
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing episode JSONs")
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_iceberg/results", help="Directory to save outputs")
    
    # Analysis Parameters
    parser.add_argument("--metric", type=str, default="prop", choices=["prop", "ratio"], help="Iceberg metric type")
    parser.add_argument("--max_shift", type=int, default=15, help="Maximum turn shift/lag for cross-correlation")
    parser.add_argument("--smooth_window", type=int, default=3, help="Smoothing window size for time series")
    parser.add_argument("--min_turns", type=int, default=30, help="Minimum turns required to process an episode")
    parser.add_argument("--min_corr_samples", type=int, default=12, help="Minimum samples required for Pearson correlation")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level for CI calculation")
    
    args = parser.parse_args()

    # Setup
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    episode_files = list(data_path.glob("*.json"))
    logger.info(f"Found {len(episode_files)} episode files in {data_path}")

    # Aggregators
    offset_correlations = {shift: [] for shift in range(-args.max_shift, args.max_shift + 1)}
    episode_records = []
    
    # Processing Loop
    for file_path in tqdm(episode_files, desc="Processing Episodes"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            df_episode = preprocess_episode(raw_data, args.metric, args.min_turns)
            
            if df_episode is not None:
                # Run Analysis
                corr_profile = analyze_dyadic_interaction(
                    df_episode, 
                    max_shift=args.max_shift,
                    smooth_window=args.smooth_window,
                    min_corr_samples=args.min_corr_samples
                )
                
                # Store Results
                record = {"episode_id": file_path.stem}
                has_data = False
                for shift, r_val in corr_profile.items():
                    record[f"shift_{shift}"] = r_val
                    if r_val is not None:
                        offset_correlations[shift].append(r_val)
                        has_data = True
                
                if has_data:
                    episode_records.append(record)
                    
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Save & Plot
    if episode_records:
        df_results = pd.DataFrame(episode_records)
        csv_path = out_path / f"episode_correlations_{args.metric}.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to {csv_path}")
        
        plot_integrated_results(
            offset_correlations, 
            out_path / f"integrated_analysis_{args.metric}.pdf",
            args.metric,
            confidence_level=args.confidence_level
        )
    else:
        logger.warning("No valid episodes found for analysis.")

if __name__ == "__main__":
    main()