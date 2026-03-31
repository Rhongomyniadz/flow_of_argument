import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from exp2_iceberg import (
    build_episode_id,
    collect_episode_files,
    infer_category_from_episode,
    normalize_categories,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = "data/stance_labeled/512"
DEFAULT_CATEGORY_DATA_SUBDIR = "parsed"
DEFAULT_FEATURE_CACHE_PATH = "experiments/exp2_iceberg/cache/turn_level_features.csv"
DEFAULT_OUTPUT_DIR = "experiments/exp2_iceberg/results_iceberg_from_stance"
DEFAULT_EVENT_PRE_TURNS = 1
DEFAULT_EVENT_POST_TURNS = 3
PRIMARY_OUTCOMES = ["explicit_count", "implicit_count", "explicit_share"]
STORED_OUTCOMES = ["explicit_count", "implicit_count", "explicit_share", "total_content", "iceberg_log_ratio"]
REQUIRED_CACHE_COLUMNS = [
    "speaker_id",
    "stance_5pt",
    "agreement_binary",
    "explicit_count",
    "implicit_count",
    "total_content",
    "explicit_share",
    "duration",
    "start_time",
    "end_time",
    "iceberg_log_ratio",
    "episode",
    "category",
    "turn_idx",
]
OUTCOME_LABELS = {
    "explicit_count": "Explicit claims",
    "implicit_count": "Implicit assumptions",
    "explicit_share": "Explicit share",
    "total_content": "Total content",
    "iceberg_log_ratio": "Log explicit/implicit ratio",
}
ONSET_COLORS = {
    "agreement": "#2a9d8f",
    "disagreement": "#bc4749",
}
OUTCOME_COLORS = {
    "explicit_count": "#2a9d8f",
    "implicit_count": "#bc4749",
    "explicit_share": "#264653",
}


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def uses_signed_stance_scale(turn: dict) -> bool:
    stance_scheme = str(turn.get("stance_scheme") or "").lower()
    if stance_scheme.startswith("signed_5pt"):
        return True
    stance_value = float(turn.get("stance_5pt"))
    return stance_value <= 0.0


def compute_agreement_binary(stance_value: float, signed_scale: bool) -> float:
    if signed_scale:
        if stance_value > 0.0:
            return 1.0
        if stance_value < 0.0:
            return 0.0
        return math.nan

    if stance_value >= 4.0:
        return 1.0
    if stance_value <= 2.0:
        return 0.0
    return math.nan


def infer_signed_stance_scale(df_input: pd.DataFrame) -> bool:
    if "stance_scheme" in df_input.columns:
        scheme_series = df_input["stance_scheme"].dropna().astype(str).str.lower()
        if scheme_series.str.startswith("signed_5pt").any():
            return True
    stance_series = pd.to_numeric(df_input["stance_5pt"], errors="coerce").dropna()
    return bool((stance_series <= 0.0).any())


def compute_turn_features(turn: dict) -> Optional[dict]:
    if turn.get("turn_type_label") != "Substantive":
        return None

    stance = turn.get("stance_5pt")
    if stance is None:
        return None

    raw_start = turn.get("start_time", turn.get("startTime", 0.0))
    start_time = float(raw_start if raw_start is not None else 0.0)

    raw_end = turn.get("end_time", turn.get("endTime"))
    raw_duration = turn.get("duration")
    if raw_end is not None:
        end_time = float(raw_end)
        duration = max(end_time - start_time, 0.1)
    elif raw_duration is not None:
        duration = max(float(raw_duration), 0.1)
        end_time = start_time + duration
    else:
        duration = 0.1
        end_time = start_time + duration

    explicit_count = int(len(turn.get("explicit_propositions", []) or []))
    implicit_count = int(len(turn.get("assumptions", []) or []))
    total_content = explicit_count + implicit_count
    explicit_share = float(explicit_count / total_content) if total_content > 0 else math.nan
    stance_value = float(stance)

    signed_scale = uses_signed_stance_scale(turn)

    return {
        "speaker_id": str(turn.get("speaker_id") or ""),
        "stance_5pt": stance_value,
        "agreement_binary": compute_agreement_binary(stance_value, signed_scale),
        "explicit_count": explicit_count,
        "implicit_count": implicit_count,
        "total_content": total_content,
        "explicit_share": explicit_share,
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "iceberg_log_ratio": math.log((explicit_count + 1.0) / (implicit_count + 1.0)),
        "stance_scheme": str(turn.get("stance_scheme") or ""),
    }


def upgrade_cached_feature_frame(df_cached: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    upgraded = df_cached.copy()
    required_base_columns = {
        "speaker_id",
        "stance_5pt",
        "explicit_count",
        "implicit_count",
        "duration",
        "start_time",
        "end_time",
        "episode",
        "category",
        "turn_idx",
    }
    missing_base_columns = sorted(required_base_columns - set(upgraded.columns))
    if missing_base_columns:
        return upgraded, missing_base_columns

    upgraded["speaker_id"] = upgraded["speaker_id"].fillna("").astype(str)
    upgraded["explicit_count"] = upgraded["explicit_count"].astype(int)
    upgraded["implicit_count"] = upgraded["implicit_count"].astype(int)

    if "agreement_binary" not in upgraded.columns:
        signed_scale = infer_signed_stance_scale(upgraded)
        upgraded["agreement_binary"] = upgraded["stance_5pt"].astype(float).map(
            lambda value: compute_agreement_binary(float(value), signed_scale)
        )
    if "total_content" not in upgraded.columns:
        upgraded["total_content"] = upgraded["explicit_count"] + upgraded["implicit_count"]
    if "explicit_share" not in upgraded.columns:
        total_content = upgraded["total_content"].astype(float)
        upgraded["explicit_share"] = np.where(
            total_content > 0.0,
            upgraded["explicit_count"].astype(float) / total_content,
            np.nan,
        )
    if "iceberg_log_ratio" not in upgraded.columns:
        upgraded["iceberg_log_ratio"] = np.log(
            (upgraded["explicit_count"].astype(float) + 1.0) / (upgraded["implicit_count"].astype(float) + 1.0)
        )

    return upgraded, []


def build_turn_level_features(
    data_dir: Path,
    categories: Optional[Sequence[str]],
    category_data_subdir: str,
    min_turns: int,
    require_two_speakers: bool,
    max_episodes: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    episode_files = collect_episode_files(
        data_path=data_dir,
        categories=categories,
        category_data_subdir=category_data_subdir,
    )
    if max_episodes is not None and max_episodes > 0:
        episode_files = episode_files[:max_episodes]

    requested_categories = normalize_categories(categories)
    requested_category_keys = {category.lower() for category in requested_categories if category.lower() != "all"}

    episode_frames: List[pd.DataFrame] = []
    files_seen = 0
    files_with_turns = 0
    files_kept = 0
    skipped_short = 0
    skipped_speaker_count = 0

    for category, file_path in tqdm(episode_files, desc="Building turn-level features"):
        files_seen += 1
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            continue

        file_category = category or infer_category_from_episode(raw_data) or ""
        if requested_category_keys and file_category.lower() not in requested_category_keys:
            continue

        turn_rows: List[dict] = []
        for turn in raw_data:
            turn_features = compute_turn_features(turn)
            if turn_features is None:
                continue
            turn_rows.append(turn_features)

        if not turn_rows:
            continue

        files_with_turns += 1
        if len(turn_rows) < min_turns:
            skipped_short += 1
            continue

        df_episode = pd.DataFrame(turn_rows)
        if require_two_speakers and df_episode["speaker_id"].nunique() != 2:
            skipped_speaker_count += 1
            continue

        episode_id = build_episode_id(file_category or None, file_path)
        df_episode["episode"] = episode_id
        df_episode["category"] = file_category or "unknown"
        df_episode["turn_idx"] = np.arange(len(df_episode), dtype=int)
        episode_frames.append(df_episode)
        files_kept += 1

    if not episode_frames:
        raise RuntimeError("No eligible episodes produced turn-level features.")

    df_all = pd.concat(episode_frames, ignore_index=True)
    summary = {
        "files_seen": files_seen,
        "episode_file_limit": int(max_episodes) if max_episodes is not None else None,
        "files_with_substantive_turns": files_with_turns,
        "episodes_kept": files_kept,
        "episodes_skipped_short": skipped_short,
        "episodes_skipped_speaker_count": skipped_speaker_count,
        "turn_rows": int(len(df_all)),
        "binary_rows": int(df_all["agreement_binary"].notna().sum()),
        "categories": df_all["category"].value_counts().sort_index().to_dict(),
        "stance_distribution": df_all["stance_5pt"].value_counts().sort_index().to_dict(),
    }
    return df_all, summary


def restrict_cached_features(
    df_cached: pd.DataFrame,
    categories: Optional[Sequence[str]],
    max_episodes: Optional[int],
) -> pd.DataFrame:
    df_filtered = df_cached.copy()
    requested_categories = normalize_categories(categories)
    requested_category_keys = {category.lower() for category in requested_categories if category.lower() != "all"}
    if requested_category_keys:
        df_filtered = df_filtered[df_filtered["category"].astype(str).str.lower().isin(requested_category_keys)].copy()

    if max_episodes is not None and max_episodes > 0:
        selected_episodes = sorted(df_filtered["episode"].astype(str).unique())[:max_episodes]
        df_filtered = df_filtered[df_filtered["episode"].astype(str).isin(selected_episodes)].copy()

    return df_filtered.reset_index(drop=True)


def attach_event_context_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df_model = df_input.sort_values(["episode", "start_time", "turn_idx"]).reset_index(drop=True).copy()
    df_model["speaker_id"] = df_model["speaker_id"].fillna("").astype(str)
    df_model["agreement_binary"] = pd.to_numeric(df_model["agreement_binary"], errors="coerce")

    episode_groups = df_model.groupby("episode", sort=False)
    df_model["prev_turn_speaker_id"] = episode_groups["speaker_id"].shift(1)
    df_model["prev_turn_agreement_binary"] = episode_groups["agreement_binary"].shift(1)
    df_model["speaker_switch"] = np.where(
        df_model["prev_turn_speaker_id"].isna(),
        np.nan,
        (df_model["speaker_id"] != df_model["prev_turn_speaker_id"]).astype(float),
    )
    return df_model


def build_onset_mask(df_model: pd.DataFrame, onset_type: str) -> pd.Series:
    if onset_type == "disagreement":
        current_value = 0.0
    elif onset_type == "agreement":
        current_value = 1.0
    else:
        raise ValueError(f"Unknown onset_type={onset_type}")

    return (
        df_model["agreement_binary"].eq(current_value)
        & df_model["prev_turn_agreement_binary"].eq(1.0)
        & df_model["speaker_switch"].eq(1.0)
    )


def coerce_optional_float(value: object) -> float:
    if pd.isna(value):
        return math.nan
    return float(value)


def compute_delta(current_value: float, reference_value: float) -> float:
    if math.isnan(current_value) or math.isnan(reference_value):
        return math.nan
    return float(current_value - reference_value)


def build_empty_event_frame() -> pd.DataFrame:
    base_columns = [
        "event_id",
        "onset_type",
        "episode",
        "category",
        "onset_row_idx",
        "pre_row_idx",
        "response_row_idx",
        "onset_turn_idx",
    ]
    outcome_columns: List[str] = []
    for outcome in STORED_OUTCOMES:
        outcome_columns.extend(
            [
                f"pre_{outcome}",
                f"onset_{outcome}",
                f"response_{outcome}",
                f"response_minus_onset_{outcome}",
                f"response_minus_pre_{outcome}",
            ]
        )
    return pd.DataFrame(columns=base_columns + outcome_columns)


def collect_response_valid_events(df_model: pd.DataFrame, onset_type: str) -> Tuple[pd.DataFrame, int]:
    onset_mask = build_onset_mask(df_model, onset_type)
    onset_indices = list(df_model.index[onset_mask])
    rows: List[dict] = []

    for onset_idx in onset_indices:
        pre_idx = onset_idx - 1
        response_idx = onset_idx + 1
        if pre_idx < 0 or response_idx >= len(df_model):
            continue

        episode = str(df_model.at[onset_idx, "episode"])
        if str(df_model.at[pre_idx, "episode"]) != episode or str(df_model.at[response_idx, "episode"]) != episode:
            continue

        onset_speaker_id = str(df_model.at[onset_idx, "speaker_id"])
        response_speaker_id = str(df_model.at[response_idx, "speaker_id"])
        if response_speaker_id == onset_speaker_id:
            continue

        row = {
            "event_id": f"{episode}::{onset_type}::turn_{int(df_model.at[onset_idx, 'turn_idx'])}",
            "onset_type": onset_type,
            "episode": episode,
            "category": str(df_model.at[onset_idx, "category"]),
            "onset_row_idx": int(onset_idx),
            "pre_row_idx": int(pre_idx),
            "response_row_idx": int(response_idx),
            "onset_turn_idx": int(df_model.at[onset_idx, "turn_idx"]),
        }

        for outcome in STORED_OUTCOMES:
            pre_value = coerce_optional_float(df_model.at[pre_idx, outcome])
            onset_value = coerce_optional_float(df_model.at[onset_idx, outcome])
            response_value = coerce_optional_float(df_model.at[response_idx, outcome])
            row[f"pre_{outcome}"] = pre_value
            row[f"onset_{outcome}"] = onset_value
            row[f"response_{outcome}"] = response_value
            row[f"response_minus_onset_{outcome}"] = compute_delta(response_value, onset_value)
            row[f"response_minus_pre_{outcome}"] = compute_delta(response_value, pre_value)

        rows.append(row)

    if not rows:
        return build_empty_event_frame(), int(onset_mask.sum())

    return pd.DataFrame(rows), int(onset_mask.sum())


def compute_summary_stats(values: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se_value = float(std_value / math.sqrt(len(values)))
    return {
        "n": int(len(values)),
        "mean": mean_value,
        "std": std_value,
        "se": se_value,
        "ci_low": float(mean_value - 1.96 * se_value),
        "ci_high": float(mean_value + 1.96 * se_value),
    }


def compute_difference_stats(left_values: np.ndarray, right_values: np.ndarray) -> Dict[str, float]:
    if len(left_values) == 0 or len(right_values) == 0:
        return {
            "n": 0,
            "reference_n": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    left_var = float(np.var(left_values, ddof=1)) if len(left_values) > 1 else 0.0
    right_var = float(np.var(right_values, ddof=1)) if len(right_values) > 1 else 0.0
    se_value = float(math.sqrt((left_var / len(left_values)) + (right_var / len(right_values))))
    diff_mean = float(left_mean - right_mean)
    return {
        "n": int(len(left_values)),
        "reference_n": int(len(right_values)),
        "mean": diff_mean,
        "std": math.nan,
        "se": se_value,
        "ci_low": float(diff_mean - 1.96 * se_value),
        "ci_high": float(diff_mean + 1.96 * se_value),
    }


def extract_numeric_values(df_input: pd.DataFrame, column_name: str) -> np.ndarray:
    return pd.to_numeric(df_input[column_name], errors="coerce").dropna().astype(float).to_numpy()


def build_group_summary_row(event_df: pd.DataFrame, onset_type: str, outcome: str) -> dict:
    pre_values = extract_numeric_values(event_df, f"pre_{outcome}")
    onset_values = extract_numeric_values(event_df, f"onset_{outcome}")
    response_values = extract_numeric_values(event_df, f"response_{outcome}")
    delta_onset_values = extract_numeric_values(event_df, f"response_minus_onset_{outcome}")
    delta_pre_values = extract_numeric_values(event_df, f"response_minus_pre_{outcome}")

    response_stats = compute_summary_stats(response_values)
    delta_onset_stats = compute_summary_stats(delta_onset_values)
    delta_pre_stats = compute_summary_stats(delta_pre_values)
    pre_mean = float(np.mean(pre_values)) if len(pre_values) else math.nan
    onset_mean = float(np.mean(onset_values)) if len(onset_values) else math.nan

    return {
        "row_type": "group_mean",
        "onset_type": onset_type,
        "reference_onset_type": None,
        "outcome": outcome,
        "n_events": int(event_df["event_id"].nunique()),
        "reference_n_events": math.nan,
        "pre_turn_mean": pre_mean,
        "onset_mean": onset_mean,
        "response_mean": response_stats["mean"],
        "response_sd": response_stats["std"],
        "response_se": response_stats["se"],
        "response_ci_low": response_stats["ci_low"],
        "response_ci_high": response_stats["ci_high"],
        "response_minus_onset_mean": delta_onset_stats["mean"],
        "response_minus_onset_se": delta_onset_stats["se"],
        "response_minus_onset_ci_low": delta_onset_stats["ci_low"],
        "response_minus_onset_ci_high": delta_onset_stats["ci_high"],
        "response_minus_pre_turn_mean": delta_pre_stats["mean"],
        "response_minus_pre_turn_se": delta_pre_stats["se"],
        "response_minus_pre_turn_ci_low": delta_pre_stats["ci_low"],
        "response_minus_pre_turn_ci_high": delta_pre_stats["ci_high"],
    }


def build_contrast_summary_row(
    disagreement_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    outcome: str,
) -> dict:
    disagreement_pre = extract_numeric_values(disagreement_df, f"pre_{outcome}")
    agreement_pre = extract_numeric_values(agreement_df, f"pre_{outcome}")
    disagreement_onset = extract_numeric_values(disagreement_df, f"onset_{outcome}")
    agreement_onset = extract_numeric_values(agreement_df, f"onset_{outcome}")
    disagreement_response = extract_numeric_values(disagreement_df, f"response_{outcome}")
    agreement_response = extract_numeric_values(agreement_df, f"response_{outcome}")
    disagreement_delta_onset = extract_numeric_values(disagreement_df, f"response_minus_onset_{outcome}")
    agreement_delta_onset = extract_numeric_values(agreement_df, f"response_minus_onset_{outcome}")
    disagreement_delta_pre = extract_numeric_values(disagreement_df, f"response_minus_pre_{outcome}")
    agreement_delta_pre = extract_numeric_values(agreement_df, f"response_minus_pre_{outcome}")

    response_stats = compute_difference_stats(disagreement_response, agreement_response)
    delta_onset_stats = compute_difference_stats(disagreement_delta_onset, agreement_delta_onset)
    delta_pre_stats = compute_difference_stats(disagreement_delta_pre, agreement_delta_pre)
    pre_mean_diff = float(np.mean(disagreement_pre) - np.mean(agreement_pre)) if len(disagreement_pre) and len(agreement_pre) else math.nan
    onset_mean_diff = float(np.mean(disagreement_onset) - np.mean(agreement_onset)) if len(disagreement_onset) and len(agreement_onset) else math.nan

    return {
        "row_type": "contrast",
        "onset_type": "disagreement",
        "reference_onset_type": "agreement",
        "outcome": outcome,
        "n_events": int(disagreement_df["event_id"].nunique()),
        "reference_n_events": int(agreement_df["event_id"].nunique()),
        "pre_turn_mean": pre_mean_diff,
        "onset_mean": onset_mean_diff,
        "response_mean": response_stats["mean"],
        "response_sd": math.nan,
        "response_se": response_stats["se"],
        "response_ci_low": response_stats["ci_low"],
        "response_ci_high": response_stats["ci_high"],
        "response_minus_onset_mean": delta_onset_stats["mean"],
        "response_minus_onset_se": delta_onset_stats["se"],
        "response_minus_onset_ci_low": delta_onset_stats["ci_low"],
        "response_minus_onset_ci_high": delta_onset_stats["ci_high"],
        "response_minus_pre_turn_mean": delta_pre_stats["mean"],
        "response_minus_pre_turn_se": delta_pre_stats["se"],
        "response_minus_pre_turn_ci_low": delta_pre_stats["ci_low"],
        "response_minus_pre_turn_ci_high": delta_pre_stats["ci_high"],
    }


def build_event_response_comparison(
    disagreement_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []
    for outcome in STORED_OUTCOMES:
        rows.append(build_group_summary_row(disagreement_df, "disagreement", outcome))
        rows.append(build_group_summary_row(agreement_df, "agreement", outcome))
        rows.append(build_contrast_summary_row(disagreement_df, agreement_df, outcome))
    return pd.DataFrame(rows)


def build_disagreement_event_window(
    disagreement_df: pd.DataFrame,
    df_model: pd.DataFrame,
    pre_event_turns: int,
    post_event_turns: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, event_row in disagreement_df.iterrows():
        onset_row_idx = int(event_row["onset_row_idx"])
        episode = str(event_row["episode"])
        event_id = str(event_row["event_id"])

        for event_time in range(-int(pre_event_turns), int(post_event_turns) + 1):
            row_idx = onset_row_idx + event_time
            if row_idx < 0 or row_idx >= len(df_model):
                continue
            if str(df_model.at[row_idx, "episode"]) != episode:
                continue

            for outcome in PRIMARY_OUTCOMES:
                value = coerce_optional_float(df_model.at[row_idx, outcome])
                onset_value = coerce_optional_float(df_model.at[onset_row_idx, outcome])
                if math.isnan(value):
                    continue
                rows.append(
                    {
                        "event_id": event_id,
                        "event_time": int(event_time),
                        "outcome": outcome,
                        "value": value,
                        "delta_from_onset": compute_delta(value, onset_value),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "outcome",
                "event_time",
                "mean_value",
                "std_value",
                "se_value",
                "ci_low_value",
                "ci_high_value",
                "mean_delta_from_onset",
                "se_delta_from_onset",
                "ci_low_delta_from_onset",
                "ci_high_delta_from_onset",
                "n_rows",
                "n_events",
            ]
        )

    event_window_df = pd.DataFrame(rows)
    summary_rows: List[dict] = []
    for (outcome, event_time), group_df in event_window_df.groupby(["outcome", "event_time"], observed=False):
        value_stats = compute_summary_stats(extract_numeric_values(group_df, "value"))
        delta_stats = compute_summary_stats(extract_numeric_values(group_df, "delta_from_onset"))
        summary_rows.append(
            {
                "outcome": outcome,
                "event_time": int(event_time),
                "mean_value": value_stats["mean"],
                "std_value": value_stats["std"],
                "se_value": value_stats["se"],
                "ci_low_value": value_stats["ci_low"],
                "ci_high_value": value_stats["ci_high"],
                "mean_delta_from_onset": delta_stats["mean"],
                "se_delta_from_onset": delta_stats["se"],
                "ci_low_delta_from_onset": delta_stats["ci_low"],
                "ci_high_delta_from_onset": delta_stats["ci_high"],
                "n_rows": int(len(group_df)),
                "n_events": int(group_df["event_id"].nunique()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["outcome", "event_time"]).reset_index(drop=True)


def save_response_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = comparison_df[
        comparison_df["row_type"].eq("group_mean") & comparison_df["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()
    if plot_df.empty:
        raise RuntimeError("Response comparison plot requires non-empty group summaries.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))
    onset_order = ["agreement", "disagreement"]
    x_positions = np.arange(len(onset_order), dtype=float)

    for axis, outcome in zip(axes, PRIMARY_OUTCOMES):
        outcome_df = plot_df[plot_df["outcome"] == outcome].set_index("onset_type").reindex(onset_order).reset_index()
        means = outcome_df["response_mean"].astype(float).to_numpy()
        ci_low = outcome_df["response_ci_low"].astype(float).to_numpy()
        ci_high = outcome_df["response_ci_high"].astype(float).to_numpy()
        lower_err = means - ci_low
        upper_err = ci_high - means
        colors = [ONSET_COLORS[onset_type] for onset_type in onset_order]

        axis.bar(
            x_positions,
            means,
            color=colors,
            width=0.62,
            alpha=0.92,
            yerr=np.vstack([lower_err, upper_err]),
            capsize=4,
            edgecolor="#1f2933",
            linewidth=0.5,
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(["Agreement", "Disagreement"])
        axis.set_title(f"Immediate Reply: {OUTCOME_LABELS[outcome]}", fontsize=11, pad=10)
        axis.grid(axis="y", alpha=0.16, linewidth=0.6)
        if outcome == "explicit_share":
            axis.set_ylim(bottom=0.0, top=min(1.0, float(np.nanmax(ci_high)) * 1.08 if len(ci_high) else 1.0))
        axis.set_ylabel(OUTCOME_LABELS[outcome])

    fig.suptitle("Experiment 2: Immediate Reply After Agreement vs Disagreement", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.80, left=0.07, right=0.98, bottom=0.14, wspace=0.28)
    fig.savefig(output_dir / "exp2_response_comparison.png", dpi=200)
    plt.close(fig)


def save_event_window_plot(
    event_window_df: pd.DataFrame,
    output_dir: Path,
    pre_event_turns: int,
    post_event_turns: int,
) -> None:
    if event_window_df.empty:
        raise RuntimeError("Event-window plot requires non-empty disagreement event summaries.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True)

    for axis, outcome in zip(axes, PRIMARY_OUTCOMES):
        outcome_df = event_window_df[event_window_df["outcome"] == outcome].sort_values("event_time").reset_index(drop=True)
        axis.plot(
            outcome_df["event_time"],
            outcome_df["mean_value"],
            color=OUTCOME_COLORS[outcome],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )
        axis.fill_between(
            outcome_df["event_time"],
            outcome_df["ci_low_value"],
            outcome_df["ci_high_value"],
            color=OUTCOME_COLORS[outcome],
            alpha=0.18,
        )
        axis.axvline(0.0, color="#495057", linewidth=1.0, linestyle="--", alpha=0.85)
        axis.axvline(1.0, color="#0f4c5c", linewidth=1.0, linestyle=":", alpha=0.85)
        axis.set_xticks(list(range(-int(pre_event_turns), int(post_event_turns) + 1)))
        axis.set_title(OUTCOME_LABELS[outcome], fontsize=11, pad=10)
        axis.set_xlabel("Turns from disagreement onset")
        axis.set_ylabel(OUTCOME_LABELS[outcome])
        axis.grid(alpha=0.16, linewidth=0.6)
        if outcome == "explicit_share":
            axis.set_ylim(bottom=0.0, top=min(1.0, float(outcome_df["ci_high_value"].max()) * 1.08))

    fig.suptitle("Experiment 2: Disagreement Event Window", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.80, left=0.07, right=0.98, bottom=0.14, wspace=0.30)
    fig.savefig(output_dir / "exp2_event_window.png", dpi=200)
    plt.close(fig)


def build_verdict(comparison_df: pd.DataFrame) -> Dict[str, object]:
    contrast_df = comparison_df[
        comparison_df["row_type"].eq("contrast") & comparison_df["reference_onset_type"].eq("agreement")
    ].copy()
    contrast_lookup = contrast_df.set_index("outcome")

    explicit_count_row = contrast_lookup.loc["explicit_count"]
    explicit_share_row = contrast_lookup.loc["explicit_share"]
    implicit_count_row = contrast_lookup.loc["implicit_count"]

    supported_on_explicit_count = bool(float(explicit_count_row["response_ci_low"]) > 0.0)
    supported_on_explicit_share = bool(float(explicit_share_row["response_ci_low"]) > 0.0)

    implicit_ci_low = float(implicit_count_row["response_ci_low"])
    implicit_ci_high = float(implicit_count_row["response_ci_high"])
    if implicit_ci_low > 0.0:
        direction_on_implicit_count = "increase"
    elif implicit_ci_high < 0.0:
        direction_on_implicit_count = "decrease"
    else:
        direction_on_implicit_count = "no_clear_change"

    if supported_on_explicit_count and supported_on_explicit_share:
        interpretation = (
            "The immediate reply after disagreement is more explicit than the matched agreement-control reply: "
            "it contains more explicit claims and a higher explicit share."
        )
    elif supported_on_explicit_count or supported_on_explicit_share:
        interpretation = (
            "Disagreement provides partial support for explicitization in the immediate reply, but the evidence is "
            "not uniform across the explicitness outcomes."
        )
    else:
        interpretation = (
            "The immediate reply after disagreement is not clearly more explicit than the matched agreement-control "
            "reply on the primary explicitness outcomes."
        )

    if direction_on_implicit_count == "decrease":
        interpretation += " Implicit content decreases at the same time, which strengthens the explicitization reading."
    elif direction_on_implicit_count == "increase":
        interpretation += " Implicit content also increases, suggesting disagreement may elicit more content overall rather than only surfacing explicit claims."
    else:
        interpretation += " Implicit content shows no clear change."

    return {
        "hypothesis_tested": "disagreement should make the later turn more explicit",
        "primary_outcome": "immediate other-speaker reply at turn +1",
        "supported_on_explicit_count": supported_on_explicit_count,
        "supported_on_explicit_share": supported_on_explicit_share,
        "direction_on_implicit_count": direction_on_implicit_count,
        "interpretation": interpretation,
    }


def row_to_summary_payload(row: pd.Series) -> Dict[str, object]:
    return {
        "onset_type": row["onset_type"],
        "reference_onset_type": row["reference_onset_type"],
        "n_events": int(row["n_events"]) if not pd.isna(row["n_events"]) else 0,
        "reference_n_events": int(row["reference_n_events"]) if not pd.isna(row["reference_n_events"]) else None,
        "pre_turn_mean": coerce_optional_float(row["pre_turn_mean"]),
        "onset_mean": coerce_optional_float(row["onset_mean"]),
        "response_mean": coerce_optional_float(row["response_mean"]),
        "response_ci_low": coerce_optional_float(row["response_ci_low"]),
        "response_ci_high": coerce_optional_float(row["response_ci_high"]),
        "response_minus_onset_mean": coerce_optional_float(row["response_minus_onset_mean"]),
        "response_minus_onset_ci_low": coerce_optional_float(row["response_minus_onset_ci_low"]),
        "response_minus_onset_ci_high": coerce_optional_float(row["response_minus_onset_ci_high"]),
        "response_minus_pre_turn_mean": coerce_optional_float(row["response_minus_pre_turn_mean"]),
        "response_minus_pre_turn_ci_low": coerce_optional_float(row["response_minus_pre_turn_ci_low"]),
        "response_minus_pre_turn_ci_high": coerce_optional_float(row["response_minus_pre_turn_ci_high"]),
    }


def extract_group_mean_rows(comparison_df: pd.DataFrame, outcomes: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    subset = comparison_df[
        comparison_df["row_type"].eq("group_mean") & comparison_df["outcome"].isin(list(outcomes))
    ].copy()
    rows: Dict[str, Dict[str, dict]] = {}
    for _, row in subset.iterrows():
        outcome = str(row["outcome"])
        onset_type = str(row["onset_type"])
        rows.setdefault(outcome, {})
        rows[outcome][onset_type] = row_to_summary_payload(row)
    return rows


def extract_contrast_rows(comparison_df: pd.DataFrame, outcomes: Sequence[str]) -> Dict[str, dict]:
    subset = comparison_df[
        comparison_df["row_type"].eq("contrast") & comparison_df["outcome"].isin(list(outcomes))
    ].copy()
    rows: Dict[str, dict] = {}
    for _, row in subset.iterrows():
        rows[str(row["outcome"])] = row_to_summary_payload(row)
    return rows


def build_dataset_summary(
    df_all: pd.DataFrame,
    feature_cache_path: Path,
    loaded_from_cache: bool,
    requested_categories: Optional[Sequence[str]],
    max_episodes: Optional[int],
    min_turns: int,
    require_two_speakers: bool,
    pre_event_turns: int,
    post_event_turns: int,
) -> Dict[str, object]:
    return {
        "loaded_from_cache": loaded_from_cache,
        "turn_rows": int(len(df_all)),
        "binary_rows": int(df_all["agreement_binary"].notna().sum()),
        "categories": df_all["category"].value_counts().sort_index().to_dict(),
        "stance_distribution": df_all["stance_5pt"].value_counts().sort_index().to_dict(),
        "feature_cache_path": str(feature_cache_path),
        "requested_categories": list(normalize_categories(requested_categories)),
        "max_episodes": int(max_episodes) if max_episodes is not None else None,
        "min_turns": int(min_turns),
        "require_two_speakers": bool(require_two_speakers),
        "event_study_pre_turns": int(pre_event_turns),
        "event_study_post_turns": int(post_event_turns),
        "hypothesis_alignment": "higher explicit_count and explicit_share after disagreement indicate more explicit replies",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the event-centered Experiment 2 analysis that tests whether disagreement makes the immediate "
            "later turn more explicit."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--category_data_subdir", type=str, default=DEFAULT_CATEGORY_DATA_SUBDIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_turns", type=int, default=12)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--rebuild_features", action="store_true")
    parser.add_argument("--allow_non_dyadic", action="store_true")
    parser.add_argument("--event_pre_turns", type=int, default=DEFAULT_EVENT_PRE_TURNS)
    parser.add_argument("--event_post_turns", type=int, default=DEFAULT_EVENT_POST_TURNS)
    parser.add_argument(
        "--feature_cache_path",
        type=str,
        default=DEFAULT_FEATURE_CACHE_PATH,
        help="Path for the cached turn-level feature table. Defaults to the shared Experiment 2 cache directory.",
    )
    parser.add_argument("--feature_cache_name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cv_splits", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--target_metric_candidates", nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model_candidates", nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--high_conflict_min_disagreement_turns", type=int, default=6, help=argparse.SUPPRESS)
    parser.add_argument("--high_conflict_min_disagreement_rate", type=float, default=0.10, help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_path = output_dir / args.feature_cache_name if args.feature_cache_name else Path(args.feature_cache_path)
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)

    use_cache = feature_cache_path.exists() and not args.rebuild_features
    if use_cache:
        logger.info("Loading cached features from %s", feature_cache_path)
        df_all = pd.read_csv(feature_cache_path)
        df_all, missing_base_columns = upgrade_cached_feature_frame(df_all)
        if missing_base_columns:
            logger.info(
                "Cached features missing required columns %s; rebuilding feature cache.",
                missing_base_columns,
            )
            use_cache = False
        else:
            df_all = restrict_cached_features(
                df_cached=df_all,
                categories=args.categories,
                max_episodes=args.max_episodes,
            )
            missing_final_columns = sorted(set(REQUIRED_CACHE_COLUMNS) - set(df_all.columns))
            if missing_final_columns:
                raise RuntimeError(
                    f"Cached features are missing required final columns after upgrade: {missing_final_columns}"
                )
            logger.info("Using cached features after request filtering | rows=%d | episodes=%d", len(df_all), df_all["episode"].nunique())

    if not use_cache:
        df_all, build_summary = build_turn_level_features(
            data_dir=Path(args.data_dir),
            categories=args.categories,
            category_data_subdir=args.category_data_subdir,
            min_turns=args.min_turns,
            require_two_speakers=not args.allow_non_dyadic,
            max_episodes=args.max_episodes,
        )
        df_all.to_csv(feature_cache_path, index=False)
        logger.info("Saved feature cache to %s", feature_cache_path)
    else:
        build_summary = {
            "files_seen": None,
            "episode_file_limit": int(args.max_episodes) if args.max_episodes is not None else None,
            "files_with_substantive_turns": None,
            "episodes_kept": int(df_all["episode"].nunique()),
            "episodes_skipped_short": None,
            "episodes_skipped_speaker_count": None,
            "turn_rows": int(len(df_all)),
            "binary_rows": int(df_all["agreement_binary"].notna().sum()),
            "categories": df_all["category"].value_counts().sort_index().to_dict(),
            "stance_distribution": df_all["stance_5pt"].value_counts().sort_index().to_dict(),
        }

    if df_all.empty:
        raise RuntimeError("No turn-level features are available after loading or building the dataset.")

    df_model = attach_event_context_columns(df_all)
    disagreement_events, candidate_disagreement_onsets = collect_response_valid_events(df_model, "disagreement")
    agreement_events, candidate_agreement_onsets = collect_response_valid_events(df_model, "agreement")

    if disagreement_events.empty:
        raise RuntimeError("No response-valid disagreement onsets were found. Experiment 2 cannot be evaluated.")
    if agreement_events.empty:
        raise RuntimeError("No response-valid agreement-control onsets were found. Experiment 2 cannot be evaluated.")

    comparison_df = build_event_response_comparison(disagreement_events, agreement_events)
    event_window_df = build_disagreement_event_window(
        disagreement_df=disagreement_events,
        df_model=df_model,
        pre_event_turns=args.event_pre_turns,
        post_event_turns=args.event_post_turns,
    )

    comparison_df.to_csv(output_dir / "exp2_event_response_comparison.csv", index=False)
    event_window_df.to_csv(output_dir / "exp2_disagreement_event_window.csv", index=False)
    save_response_comparison_plot(comparison_df=comparison_df, output_dir=output_dir)
    save_event_window_plot(
        event_window_df=event_window_df,
        output_dir=output_dir,
        pre_event_turns=args.event_pre_turns,
        post_event_turns=args.event_post_turns,
    )

    dataset_summary = build_dataset_summary(
        df_all=df_all,
        feature_cache_path=feature_cache_path,
        loaded_from_cache=use_cache,
        requested_categories=args.categories,
        max_episodes=args.max_episodes,
        min_turns=args.min_turns,
        require_two_speakers=not args.allow_non_dyadic,
        pre_event_turns=args.event_pre_turns,
        post_event_turns=args.event_post_turns,
    )
    dataset_summary.update(build_summary)

    verdict = build_verdict(comparison_df)
    summary_payload = {
        "dataset_summary": dataset_summary,
        "analysis_design": {
            "framing": "event_centered_disagreement_analysis",
            "clean_disagreement_onset_definition": "current disagreement turn after agreement turn with speaker switch",
            "agreement_control_definition": "current agreement turn after agreement turn with speaker switch",
            "primary_response_requirement": "turn +1 must stay in the same episode and be spoken by the other speaker",
            "primary_outcomes": list(PRIMARY_OUTCOMES),
            "secondary_diagnostics": ["total_content", "iceberg_log_ratio"],
        },
        "onset_counts": {
            "candidate_disagreement_onsets": int(candidate_disagreement_onsets),
            "response_valid_disagreement_onsets": int(disagreement_events["event_id"].nunique()),
            "candidate_agreement_onsets": int(candidate_agreement_onsets),
            "response_valid_agreement_onsets": int(agreement_events["event_id"].nunique()),
        },
        "primary_response_group_means": extract_group_mean_rows(
            comparison_df=comparison_df,
            outcomes=PRIMARY_OUTCOMES,
        ),
        "primary_contrasts": extract_contrast_rows(
            comparison_df=comparison_df,
            outcomes=PRIMARY_OUTCOMES,
        ),
        "secondary_diagnostics": extract_contrast_rows(
            comparison_df=comparison_df,
            outcomes=["total_content", "iceberg_log_ratio"],
        ),
        "verdict": verdict,
    }

    save_json(output_dir / "exp2_summary.json", summary_payload)
    save_json(output_dir / "dataset_summary.json", dataset_summary)

    logger.info(
        "Experiment 2 completed | disagreement_events=%d | agreement_events=%d | explicit_count_supported=%s | explicit_share_supported=%s",
        disagreement_events["event_id"].nunique(),
        agreement_events["event_id"].nunique(),
        verdict["supported_on_explicit_count"],
        verdict["supported_on_explicit_share"],
    )


if __name__ == "__main__":
    main()
