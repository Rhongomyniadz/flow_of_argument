import argparse
import contextlib
import io
import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

# ============================================================================
# Silence statsmodels FutureWarning about verbose parameter
# ============================================================================
warnings.filterwarnings(
    "ignore",
    message="verbose is deprecated since functions should not print results",
    category=FutureWarning,
)

# ============================================================================
# Logging (keep INFO so you can still see tqdm; we won't print Granger internals)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Processing
# ============================================================================
def compute_iceberg_ratio(explicit: int, implicit: int, duration: float, metric_type: str) -> float:
    """Computes the normalized 'Iceberg' metric."""
    total = explicit + implicit
    duration = max(duration, 0.1)

    if total == 0:
        return 0.0

    if metric_type == "prop":
        return (explicit / total) / duration
    else:
        return (explicit / (implicit + 1e-6)) / duration


def preprocess_episode(data: List[dict], metric_type: str, min_turns: int) -> Optional[pd.DataFrame]:
    """Parses raw JSON turn data into a structured DataFrame (Substantive turns only)."""
    features = []

    for turn in data:
        if turn.get("turn_type_label") != "Substantive":
            continue

        stance = turn.get("stance_5pt")
        if stance is None:
            continue

        start_time = float(turn.get("startTime", 0.0))
        end_time = float(turn.get("endTime", start_time + 0.1))
        duration = max(end_time - start_time, 0.1)

        explicit_count = len(turn.get("explicit_propositions", []) or [])
        implicit_count = len(turn.get("assumptions", []) or [])

        iceberg_val = compute_iceberg_ratio(explicit_count, implicit_count, duration, metric_type)

        features.append(
            {
                "speaker_id": turn.get("speaker_id"),
                "stance_5pt": float(stance),
                "iceberg_norm": float(iceberg_val),
                "startTime": start_time,
            }
        )

    if len(features) < min_turns:
        return None

    df = pd.DataFrame(features).sort_values("startTime").reset_index(drop=True)
    if df["speaker_id"].nunique() != 2:
        return None
    return df


# ============================================================================
# Granger Utilities
# ============================================================================
def safe_granger_f_stat(y: np.ndarray, x: np.ndarray, lag: int, min_n: int) -> Optional[float]:
    """
    Returns the SSR-based Granger F-statistic for testing whether x Granger-causes y.
    Silences all statsmodels stdout while running the test.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    # Need enough points for lagged regression.
    if len(y) < max(min_n, lag + 5):
        return None

    data = np.column_stack([y, x])

    try:
        # 🔇 Silence internal statsmodels printing
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data, maxlag=lag, verbose=False)

        f_stat, p_val, _, _ = res[lag][0]["ssr_ftest"]
        if not np.isfinite(f_stat):
            return None
        return float(f_stat)
    except Exception:
        return None


def analyze_dyadic_granger_longform(
    df: pd.DataFrame,
    episode_id: str,
    max_shift: int,
    smooth_window: int,
    granger_lag: int,
    min_granger_samples: int,
    odd_offsets_only: bool = True,
) -> List[dict]:
    """
    For each offset in [-max_shift, +max_shift], build matched (stance -> other iceberg) sequences
    for each direction (A->B and B->A), then compute Granger F-stat score.

    Output rows: {episode, speaker_id, offset, granger_score}
      - speaker_id = the source speaker whose stance is tested as causing the other's iceberg
      - granger_score = SSR F-statistic (higher -> stronger evidence, generally)

    If odd_offsets_only=True, only evaluates odd offsets (and skips 0). This is correct when you have
    merged consecutive same-speaker turns and the sequence alternates A,B,A,B,... so cross-speaker
    relations occur only at odd offsets.
    """
    df = df.copy()

    # smoothing
    df["iceberg_smooth"] = df["iceberg_norm"].rolling(smooth_window, center=True, min_periods=1).mean()
    df["stance_smooth"] = df["stance_5pt"].rolling(smooth_window, center=True, min_periods=1).mean()

    speakers = sorted(df["speaker_id"].unique())
    if len(speakers) != 2:
        return []

    spk_a, spk_b = speakers[0], speakers[1]
    rows: List[dict] = []

    # helper to append both directions for a given shift
    def process_shift(shift: int):
        a_stance_seq: List[float] = []
        b_iceberg_seq: List[float] = []
        b_stance_seq: List[float] = []
        a_iceberg_seq: List[float] = []

        n = len(df)
        for idx in range(n):
            target_idx = idx + shift
            if target_idx < 0 or target_idx >= n:
                continue

            curr = df.iloc[idx]
            target = df.iloc[target_idx]

            # A stance predicts B iceberg
            if curr["speaker_id"] == spk_a and target["speaker_id"] == spk_b:
                a_stance_seq.append(float(curr["stance_smooth"]))
                b_iceberg_seq.append(float(target["iceberg_smooth"]))

            # B stance predicts A iceberg
            if curr["speaker_id"] == spk_b and target["speaker_id"] == spk_a:
                b_stance_seq.append(float(curr["stance_smooth"]))
                a_iceberg_seq.append(float(target["iceberg_smooth"]))

        score_a_to_b = safe_granger_f_stat(
            y=np.array(b_iceberg_seq),
            x=np.array(a_stance_seq),
            lag=granger_lag,
            min_n=min_granger_samples,
        )
        rows.append(
            {"episode": episode_id, "speaker_id": spk_a, "offset": shift, "granger_score": score_a_to_b}
        )

        score_b_to_a = safe_granger_f_stat(
            y=np.array(a_iceberg_seq),
            x=np.array(b_stance_seq),
            lag=granger_lag,
            min_n=min_granger_samples,
        )
        rows.append(
            {"episode": episode_id, "speaker_id": spk_b, "offset": shift, "granger_score": score_b_to_a}
        )

    if odd_offsets_only:
        # Only evaluate odd offsets (skip 0 and even), both negative and positive
        for k in range(1, max_shift + 1, 2):
            process_shift(-k)
            process_shift(k)
    else:
        # Evaluate all offsets in the full grid
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0:
                process_shift(shift)  # still allowed
            else:
                process_shift(shift)

    return rows


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Stance->Other-Iceberg Granger Causality over Offset Grid")

    # Input/Output
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing episode JSONs")
    parser.add_argument(
        "--output_dir", type=str, default="experiments/exp2_iceberg/results", help="Directory to save outputs"
    )

    # Parameters
    parser.add_argument("--metric", type=str, default="prop", choices=["prop", "ratio"], help="Iceberg metric type")
    parser.add_argument("--max_shift", type=int, default=15, help="Maximum turn offset (both +/-)")
    parser.add_argument("--smooth_window", type=int, default=3, help="Rolling smoothing window")
    parser.add_argument("--min_turns", type=int, default=30, help="Minimum substantive turns required per episode")

    # Granger-specific
    parser.add_argument("--granger_lag", type=int, default=1, help="Lag (in steps) used in Granger test")
    parser.add_argument(
        "--min_granger_samples",
        type=int,
        default=20,
        help="Minimum matched samples required to run Granger test (per direction, per offset)",
    )

    # Offset mode
    parser.add_argument(
        "--odd_offsets_only",
        action="store_true",
        help="Only compute odd offsets (recommended if merged to strict A,B,A,B alternation).",
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(list(data_path.glob("*.json")))
    logger.info(f"Found {len(episode_files)} episode files in {data_path}")

    all_rows: List[dict] = []

    for file_path in tqdm(episode_files, desc="Processing Episodes"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            df_episode = preprocess_episode(raw_data, args.metric, args.min_turns)
            if df_episode is None:
                continue

            episode_id = file_path.stem
            rows = analyze_dyadic_granger_longform(
                df=df_episode,
                episode_id=episode_id,
                max_shift=args.max_shift,
                smooth_window=args.smooth_window,
                granger_lag=args.granger_lag,
                min_granger_samples=args.min_granger_samples,
                odd_offsets_only=args.odd_offsets_only,
            )
            all_rows.extend(rows)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    if not all_rows:
        logger.warning("No valid rows produced. Check min_turns/min_granger_samples or data format.")
        return

    df_long = pd.DataFrame(all_rows)
    df_long = df_long[["episode", "speaker_id", "offset", "granger_score"]]

    out_csv = out_path / f"granger_longform_{args.metric}.csv"
    df_long.to_csv(out_csv, index=False)
    logger.info(f"Saved long-form Granger output to {out_csv}")


if __name__ == "__main__":
    main()
