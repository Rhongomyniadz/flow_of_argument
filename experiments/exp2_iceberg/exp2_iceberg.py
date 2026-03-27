import argparse
import contextlib
import io
import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "data/stance_labeled/512"
DEFAULT_CATEGORY_DATA_SUBDIR = "parsed"

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

        raw_start_time = turn.get("start_time", turn.get("startTime", 0.0))
        start_time = float(raw_start_time if raw_start_time is not None else 0.0)

        raw_end_time = turn.get("end_time", turn.get("endTime"))
        raw_duration = turn.get("duration")
        if raw_end_time is not None:
            end_time = float(raw_end_time)
            duration = max(end_time - start_time, 0.1)
        elif raw_duration is not None:
            duration = max(float(raw_duration), 0.1)
            end_time = start_time + duration
        else:
            end_time = start_time + 0.1
            duration = 0.1

        explicit_count = len(turn.get("explicit_propositions", []) or [])
        implicit_count = len(turn.get("assumptions", []) or [])

        iceberg_val = compute_iceberg_ratio(explicit_count, implicit_count, duration, metric_type)

        features.append(
            {
                "speaker_id": turn.get("speaker_id"),
                "stance_5pt": float(stance),
                "iceberg_norm": float(iceberg_val),
                "start_time": start_time,
            }
        )

    if len(features) < min_turns:
        return None

    df = pd.DataFrame(features).sort_values("start_time").reset_index(drop=True)
    if df["speaker_id"].nunique() != 2:
        return None
    return df


def discover_category_names(data_path: Path, category_data_subdir: str) -> List[str]:
    """Returns category names for immediate subdirectories that contain the requested data subdir."""
    categories: List[str] = []
    for child in sorted(data_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / category_data_subdir).is_dir():
            categories.append(child.name)
    return categories


def normalize_categories(categories: Optional[Sequence[str]]) -> List[str]:
    """Deduplicates requested categories while preserving user order."""
    if not categories:
        return []

    normalized: List[str] = []
    seen = set()
    for category in categories:
        category_name = str(category).strip()
        if not category_name:
            continue
        key = category_name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(category_name)
    return normalized


def resolve_requested_categories(requested: Sequence[str], available: Sequence[str]) -> List[str]:
    """Resolves a user request against available category names, supporting 'all'."""
    available_lookup = {category.lower(): category for category in available}

    if not requested or any(category.lower() == "all" for category in requested):
        return list(available)

    resolved: List[str] = []
    missing: List[str] = []
    for category in requested:
        match = available_lookup.get(category.lower())
        if match is None:
            missing.append(category)
            continue
        resolved.append(match)

    if missing:
        raise ValueError(
            f"Unknown categories: {', '.join(sorted(missing))}. "
            f"Available categories: {', '.join(available)}."
        )

    return resolved


def find_category_lookup_root(data_path: Path, category_data_subdir: str) -> Optional[Path]:
    """Finds a nearby category-root directory that can be used to map flat files back to categories."""
    for candidate in [data_path.parent.parent, data_path.parent]:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if discover_category_names(candidate, category_data_subdir):
            return candidate
    return None


def collect_episode_files(
    data_path: Path, categories: Optional[Sequence[str]], category_data_subdir: str
) -> List[Tuple[Optional[str], Path]]:
    """
    Collects episode files from one of three layouts:
      1. Category root: data/{category}/{category_data_subdir}/*.json
      2. Single category root: data/{category}/{category_data_subdir}
      3. Direct JSON folder: some/path/*.json
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {data_path}")

    requested_categories = normalize_categories(categories)
    available_categories = discover_category_names(data_path, category_data_subdir)

    if available_categories:
        selected_categories = resolve_requested_categories(requested_categories, available_categories)
        logger.info("Running categories: %s", ", ".join(selected_categories))

        episode_files: List[Tuple[Optional[str], Path]] = []
        for category in selected_categories:
            category_path = data_path / category / category_data_subdir
            category_files = sorted(category_path.glob("*.json"))
            logger.info("Category=%s | files=%d | path=%s", category, len(category_files), category_path)
            episode_files.extend((category, file_path) for file_path in category_files)
        return episode_files

    single_category_path = data_path / category_data_subdir
    if single_category_path.is_dir():
        category_name = data_path.name
        if requested_categories and not any(
            category.lower() in {"all", category_name.lower()} for category in requested_categories
        ):
            logger.warning(
                "Ignoring --categories because %s already points to a single category directory.", data_path
            )
        category_files = sorted(single_category_path.glob("*.json"))
        logger.info("Single category=%s | files=%d | path=%s", category_name, len(category_files), single_category_path)
        return [(category_name, file_path) for file_path in category_files]

    direct_json_files = sorted(data_path.glob("*.json"))
    if direct_json_files:
        inferred_category = data_path.parent.name if data_path.name == category_data_subdir else None
        category_lookup_root = find_category_lookup_root(data_path, category_data_subdir)

        if requested_categories and inferred_category is None and category_lookup_root is not None:
            available_categories = discover_category_names(category_lookup_root, category_data_subdir)
            selected_categories = resolve_requested_categories(requested_categories, available_categories)

            filtered_files: List[Tuple[Optional[str], Path]] = []
            missing_files = 0
            for category in selected_categories:
                category_files = sorted((category_lookup_root / category / category_data_subdir).glob("*.json"))
                for category_file in category_files:
                    candidate = data_path / category_file.name
                    if candidate.exists():
                        filtered_files.append((category, candidate))
                    else:
                        missing_files += 1

            logger.info(
                "Direct JSON category selection | categories=%s | matched_files=%d | missing_matches=%d | lookup_root=%s",
                ", ".join(selected_categories),
                len(filtered_files),
                missing_files,
                category_lookup_root,
            )
            return filtered_files

        if requested_categories and inferred_category is None:
            logger.info("Applying category filtering from each episode's embedded 'category' field.")
        logger.info(
            "Direct JSON input | inferred_category=%s | files=%d | path=%s",
            inferred_category or "none",
            len(direct_json_files),
            data_path,
        )
        return [(inferred_category, file_path) for file_path in direct_json_files]

    raise FileNotFoundError(
        f"No episode JSON files found under {data_path}. Expected either JSON files directly, "
        f"a single '{category_data_subdir}' subdirectory, or category directories containing "
        f"'{category_data_subdir}'."
    )


def build_episode_id(category: Optional[str], file_path: Path) -> str:
    """Builds a unique episode identifier for combined multi-category runs."""
    if category:
        return f"{category}/{file_path.stem}"
    return file_path.stem


def infer_category_from_episode(raw_data: List[dict]) -> Optional[str]:
    """Infers the category from episode payload metadata when files are stored in a flat directory."""
    if not raw_data:
        return None

    for turn in raw_data:
        category = str(turn.get("category") or "").strip()
        if category:
            return category
    return None


# ============================================================================
# Granger Utilities
# ============================================================================
def safe_granger_p_value(y: np.ndarray, x: np.ndarray, lag: int, min_n: int) -> Optional[float]:
    """
    Returns the p-value from the SSR-based Granger F-test.
    Lower p-value indicates stronger evidence of Granger causality.
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
        # Silence internal statsmodels printing
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data, maxlag=lag, verbose=False)

        # Extract p-value from the ssr_ftest tuple: (f_stat, p_val, df_denom, df_num)
        _, p_val, _, _ = res[lag][0]["ssr_ftest"]
        
        if not np.isfinite(p_val):
            return None
        return float(p_val)
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
    for each direction (A->B and B->A), then compute Granger p-value.

    Output rows: {episode, speaker_id, offset, granger_p_value}
      - speaker_id = the source speaker whose stance is tested as causing the other's iceberg
      - granger_p_value = p-value from the F-test (lower means more significant)
    """
    df = df.copy()

    df["iceberg_smooth"] = df["iceberg_norm"]
    df["stance_smooth"] = df["stance_5pt"]
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

        # Compute p-value for A -> B
        p_val_a_to_b = safe_granger_p_value(
            y=np.array(b_iceberg_seq),
            x=np.array(a_stance_seq),
            lag=granger_lag,
            min_n=min_granger_samples,
        )
        rows.append(
            {"episode": episode_id, "speaker_id": spk_a, "offset": shift, "granger_p_value": p_val_a_to_b}
        )

        # Compute p-value for B -> A
        p_val_b_to_a = safe_granger_p_value(
            y=np.array(a_iceberg_seq),
            x=np.array(b_stance_seq),
            lag=granger_lag,
            min_n=min_granger_samples,
        )
        rows.append(
            {"episode": episode_id, "speaker_id": spk_b, "offset": shift, "granger_p_value": p_val_b_to_a}
        )

    if odd_offsets_only:
        # Only evaluate odd offsets (skip 0 and even), both negative and positive
        for k in range(1, max_shift + 1, 2):
            process_shift(-k)
            process_shift(k)
    else:
        # Evaluate all offsets in the full grid
        for shift in range(-max_shift, max_shift + 1):
            process_shift(shift)

    return rows


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Stance->Other-Iceberg Granger Causality over Offset Grid")

    # Input/Output
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=(
            "Path to either a flat stance-labeled directory (default), a category root "
            "(for example data/), a single category (for example data/commentary), "
            "or a directory containing episode JSONs."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category names to run when --data_dir is a category root. Omit or pass 'all' to run every category.",
    )
    parser.add_argument(
        "--category_data_subdir",
        type=str,
        default=DEFAULT_CATEGORY_DATA_SUBDIR,
        help="Per-category subdirectory that contains episode JSONs when using category-root mode.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/exp2_iceberg/results", help="Directory to save outputs"
    )

    # Parameters
    parser.add_argument("--metric", type=str, default="prop", choices=["prop", "ratio"], help="Iceberg metric type")
    parser.add_argument("--max_shift", type=int, default=15, help="Maximum turn offset (both +/-)")
    parser.add_argument("--smooth_window", type=int, default=3, help="Rolling smoothing window")
    parser.add_argument("--min_turns", type=int, default=30, help="Minimum substantive turns required per episode")

    # Granger-specific
    parser.add_argument("--granger_lag", type=int, default=2, help="Lag (in steps) used in Granger test")
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

    try:
        episode_files = collect_episode_files(
            data_path=data_path,
            categories=args.categories,
            category_data_subdir=args.category_data_subdir,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info("Found %d episode files to process from %s", len(episode_files), data_path)

    all_rows: List[dict] = []
    requested_categories = normalize_categories(args.categories)
    requested_category_keys = {category.lower() for category in requested_categories if category.lower() != "all"}

    for category, file_path in tqdm(episode_files, desc="Processing Episodes"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            file_category = category or infer_category_from_episode(raw_data)
            if requested_category_keys and (file_category or "").lower() not in requested_category_keys:
                continue

            df_episode = preprocess_episode(raw_data, args.metric, args.min_turns)
            if df_episode is None:
                continue

            episode_id = build_episode_id(file_category, file_path)
            rows = analyze_dyadic_granger_longform(
                df=df_episode,
                episode_id=episode_id,
                max_shift=args.max_shift,
                smooth_window=args.smooth_window,
                granger_lag=args.granger_lag,
                min_granger_samples=args.min_granger_samples,
                odd_offsets_only=args.odd_offsets_only,
            )
            for row in rows:
                row["category"] = file_category or ""
            all_rows.extend(rows)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    if not all_rows:
        logger.warning("No valid rows produced. Check min_turns/min_granger_samples or data format.")
        return

    df_long = pd.DataFrame(all_rows)
    df_long = df_long[["category", "episode", "speaker_id", "offset", "granger_p_value"]]

    out_csv = out_path / f"granger_longform_{args.metric}.csv"
    df_long.to_csv(out_csv, index=False)
    logger.info(f"Saved long-form Granger output to {out_csv}")


if __name__ == "__main__":
    main()
