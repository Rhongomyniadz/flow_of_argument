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
# Fisher-Z Utilities
# ============================================================================
def fisher_z_transform(r: float) -> float:
    eps = 1e-10
    r = max(min(r, 1 - eps), -1 + eps)
    return 0.5 * math.log((1 + r) / (1 - r))

def inverse_fisher_z(z: float) -> float:
    return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)

def safe_pearson_with_n(x: np.ndarray, y: np.ndarray, min_n: int = 12) -> Tuple[Optional[float], int]:
    """
    Returns (r, n_valid_pairs_used).
    n_valid_pairs_used counts finite pairs after masking.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    n = int(len(x_clean))

    if n < min_n:
        return None, n
    if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
        return None, n

    r = float(np.corrcoef(x_clean, y_clean)[0, 1])
    if not np.isfinite(r):
        return None, n
    return r, n

def _inv_var_weight_from_n(n_pairs: int) -> float:
    """
    For Fisher-z, Var(z) ≈ 1/(n-3). So inverse-variance weight w = (n-3).
    """
    return float(max(n_pairs - 3, 0))

def combine_two_directions_zavg_weight(
    r_fwd: float, n_fwd: int,
    r_bwd: float, n_bwd: int
) -> Tuple[float, float]:
    """
    z_avg = (z_fwd + z_bwd)/2
    Var(z_fwd)=1/(n_fwd-3), Var(z_bwd)=1/(n_bwd-3)
    Var(z_avg)=0.25*(Var_fwd+Var_bwd)
    weight = 1/Var(z_avg) = 4/(Var_fwd+Var_bwd)
    Returns (r_integrated, weight_for_zavg)
    """
    z_fwd = fisher_z_transform(r_fwd)
    z_bwd = fisher_z_transform(r_bwd)
    z_avg = 0.5 * (z_fwd + z_bwd)

    # Handle edge cases safely
    denom_f = _inv_var_weight_from_n(n_fwd)
    denom_b = _inv_var_weight_from_n(n_bwd)
    # denom_* here is (n-3) = 1/Var, so Var = 1/denom
    if denom_f <= 0 or denom_b <= 0:
        # fallback: if one side too small, treat as unweighted avg (very rare due to min_n)
        w = 0.0
    else:
        var_f = 1.0 / denom_f
        var_b = 1.0 / denom_b
        var_avg = 0.25 * (var_f + var_b)
        w = 1.0 / var_avg

    r_int = inverse_fisher_z(z_avg)
    return float(r_int), float(w)

def aggregate_correlations_fisher_z_weighted(
    rz_weight_list: List[Tuple[float, float]],
    confidence_level: float = 0.95
) -> Tuple[float, Optional[float], Optional[float], int, float]:
    """
    Weighted fixed-effect meta-analysis in Fisher-Z space.
    Input: list of (r, weight) where weight is inverse-variance weight in z-space.
    Returns: (r_agg, ci_lower, ci_upper, n_valid, sum_w)
    """
    # keep finite r and positive weights
    cleaned: List[Tuple[float, float]] = []
    for r, w in rz_weight_list:
        if r is None:
            continue
        if not (np.isfinite(r) and np.isfinite(w)):
            continue
        if w <= 0:
            continue
        cleaned.append((float(r), float(w)))

    n_valid = len(cleaned)
    if n_valid == 0:
        return float("nan"), None, None, 0, 0.0

    z = np.array([fisher_z_transform(r) for r, _ in cleaned], dtype=float)
    w = np.array([w for _, w in cleaned], dtype=float)

    sum_w = float(np.sum(w))
    z_hat = float(np.sum(w * z) / sum_w)
    r_hat = float(inverse_fisher_z(z_hat))

    ci_lower = ci_upper = None
    if confidence_level > 0:
        z_crit = float(stats.norm.ppf(1 - (1 - confidence_level) / 2))
        se = math.sqrt(1.0 / sum_w)
        z_lo = z_hat - z_crit * se
        z_hi = z_hat + z_crit * se
        ci_lower = float(inverse_fisher_z(z_lo))
        ci_upper = float(inverse_fisher_z(z_hi))

    return r_hat, ci_lower, ci_upper, n_valid, sum_w

# ============================================================================
# Data Processing
# ============================================================================
def compute_iceberg_ratio(explicit: int, implicit: int, duration: float, metric_type: str) -> float:
    total = explicit + implicit
    duration = max(duration, 0.1)
    if total == 0:
        return 0.0
    if metric_type == "prop":
        return (explicit / total) / duration
    else:
        return (explicit / (implicit + 1e-6)) / duration

def preprocess_episode(data: List[Dict], metric_type: str, min_turns: int) -> Optional[pd.DataFrame]:
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
            "iceberg_norm": float(iceberg_val),
            "startTime": float(start_time),
        })

    if len(features) < min_turns:
        return None
    return pd.DataFrame(features).sort_values("startTime").reset_index(drop=True)

# ============================================================================
# Core Analysis Logic (Now returns (r, weight, n_pairs_used))
# ============================================================================
def analyze_dyadic_interaction_weighted(
    df: pd.DataFrame,
    max_shift: int,
    smooth_window: int,
    min_corr_samples: int
) -> Dict[int, Tuple[Optional[float], float, int]]:
    """
    For each shift, returns:
      profile[shift] = (r_integrated_or_single, weight_in_z_space, n_pairs_used_effective)
    weight is inverse-variance weight suitable for Fisher-Z aggregation.
    """
    df = df.copy()
    df["iceberg_smooth"] = df["iceberg_norm"].rolling(smooth_window, center=True, min_periods=1).mean()
    df["stance_smooth"] = df["stance_5pt"].rolling(smooth_window, center=True, min_periods=1).mean()

    speakers = sorted(df["speaker_id"].unique())
    if len(speakers) != 2:
        return {}

    spk_a, spk_b = speakers[0], speakers[1]
    profile: Dict[int, Tuple[Optional[float], float, int]] = {}

    for shift in range(-max_shift, max_shift + 1):
        fwd_pairs = []
        bwd_pairs = []

        for idx in range(len(df)):
            target_idx = idx + shift
            if 0 <= target_idx < len(df):
                curr = df.iloc[idx]
                target = df.iloc[target_idx]

                # A stance -> B iceberg
                if curr["speaker_id"] == spk_a and target["speaker_id"] == spk_b:
                    fwd_pairs.append((curr["stance_smooth"], target["iceberg_smooth"]))

                # B stance -> A iceberg
                if curr["speaker_id"] == spk_b and target["speaker_id"] == spk_a:
                    bwd_pairs.append((curr["stance_smooth"], target["iceberg_smooth"]))

        x_fwd = np.array([p[0] for p in fwd_pairs], dtype=float)
        y_fwd = np.array([p[1] for p in fwd_pairs], dtype=float)
        x_bwd = np.array([p[0] for p in bwd_pairs], dtype=float)
        y_bwd = np.array([p[1] for p in bwd_pairs], dtype=float)

        r_fwd, n_fwd = safe_pearson_with_n(x_fwd, y_fwd, min_n=min_corr_samples)
        r_bwd, n_bwd = safe_pearson_with_n(x_bwd, y_bwd, min_n=min_corr_samples)

        if r_fwd is not None and r_bwd is not None:
            r_int, w_int = combine_two_directions_zavg_weight(r_fwd, n_fwd, r_bwd, n_bwd)
            # for reporting, an "effective n" is not strictly needed; keep min as a simple proxy
            n_eff = int(min(n_fwd, n_bwd))
            profile[shift] = (r_int, w_int, n_eff)

        elif r_fwd is not None:
            w = _inv_var_weight_from_n(n_fwd)
            profile[shift] = (float(r_fwd), float(w), int(n_fwd))

        elif r_bwd is not None:
            w = _inv_var_weight_from_n(n_bwd)
            profile[shift] = (float(r_bwd), float(w), int(n_bwd))

        else:
            profile[shift] = (None, 0.0, 0)

    return profile

# ============================================================================
# Visualization (Weighted aggregation)
# ============================================================================
def plot_integrated_results_weighted(
    offset_data: Dict[int, List[Tuple[float, float]]],  # shift -> list of (r, weight)
    output_path: Path,
    metric_name: str,
    confidence_level: float = 0.95,
    plot_dpi: int = 300
) -> None:
    offsets = sorted(offset_data.keys())
    x_axis = np.array(offsets, dtype=int)

    means, ci_lows, ci_highs = [], [], []

    for shift in offsets:
        r, low, high, n_valid, sum_w = aggregate_correlations_fisher_z_weighted(
            offset_data[shift],
            confidence_level=confidence_level
        )
        means.append(r)
        ci_lows.append(low if low is not None else np.nan)
        ci_highs.append(high if high is not None else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_axis, means,
        marker="o", markersize=5, linestyle="-", linewidth=2,
        label="Integrated Correlation (Weighted)"
    )

    valid_mask = np.isfinite(ci_lows) & np.isfinite(ci_highs)
    if np.any(valid_mask):
        ax.fill_between(
            x_axis[valid_mask],
            np.array(ci_lows)[valid_mask],
            np.array(ci_highs)[valid_mask],
            alpha=0.2,
            label=f"{int(confidence_level*100)}% CI (Weighted)"
        )

    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Lag (Turns)\n(-) Other's Stance precedes Self Iceberg | (+) Self Stance precedes Other's Iceberg")
    ax.set_ylabel("Pearson Correlation (Fisher-Z Aggregated)")
    ax.set_title(f"Cross-Speaker Influence: Stance vs. Iceberg ({metric_name})")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi)
    plt.close()
    logger.info(f"Generated plot at {output_path}")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Stance-Iceberg Cross-Correlation Analysis (Weighted)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing episode JSONs")
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_iceberg/results", help="Directory to save outputs")

    parser.add_argument("--metric", type=str, default="prop", choices=["prop", "ratio"], help="Iceberg metric type")
    parser.add_argument("--max_shift", type=int, default=15, help="Maximum turn shift/lag for cross-correlation")
    parser.add_argument("--smooth_window", type=int, default=3, help="Smoothing window size for time series")
    parser.add_argument("--min_turns", type=int, default=30, help="Minimum turns required to process an episode")
    parser.add_argument("--min_corr_samples", type=int, default=12, help="Minimum samples required for Pearson correlation")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level for CI calculation")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    episode_files = list(data_path.glob("*.json"))
    logger.info(f"Found {len(episode_files)} episode files in {data_path}")

    # shift -> list of (r, weight)
    offset_correlations_weighted: Dict[int, List[Tuple[float, float]]] = {
        shift: [] for shift in range(-args.max_shift, args.max_shift + 1)
    }
    episode_records = []

    for file_path in tqdm(episode_files, desc="Processing Episodes"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            df_episode = preprocess_episode(raw_data, args.metric, args.min_turns)
            if df_episode is None:
                continue

            corr_profile = analyze_dyadic_interaction_weighted(
                df_episode,
                max_shift=args.max_shift,
                smooth_window=args.smooth_window,
                min_corr_samples=args.min_corr_samples,
            )

            record = {"episode_id": file_path.stem}
            has_data = False

            for shift, (r_val, w_val, n_pairs) in corr_profile.items():
                record[f"shift_{shift}"] = r_val
                record[f"weight_{shift}"] = w_val
                record[f"nPairs_{shift}"] = n_pairs

                if r_val is not None and w_val > 0:
                    offset_correlations_weighted[shift].append((r_val, w_val))
                    has_data = True

            if has_data:
                episode_records.append(record)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    if episode_records:
        df_results = pd.DataFrame(episode_records)
        csv_path = out_path / f"episode_correlations_weighted_{args.metric}.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to {csv_path}")

        plot_integrated_results_weighted(
            offset_correlations_weighted,
            out_path / f"integrated_analysis_weighted_{args.metric}.pdf",
            args.metric,
            confidence_level=args.confidence_level,
        )
    else:
        logger.warning("No valid episodes found for analysis.")

if __name__ == "__main__":
    main()
