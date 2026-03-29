import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
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
DEFAULT_FEATURE_CACHE = "turn_level_features.csv"
DEFAULT_FEATURE_CACHE_PATH = "experiments/exp2_iceberg/cache/turn_level_features.csv"
DEFAULT_OUTPUT_DIR = "experiments/exp2_iceberg/results_iceberg_from_stance"
DEFAULT_TARGET_METRIC_CANDIDATES = [
    "iceberg_log_ratio",
]
DEFAULT_MODEL_CANDIDATES = [
    "baseline_temporal_control",
    "linear_current_plus_history",
    "linear_temporal_history",
    "linear_temporal_history_interactions",
    "spline_temporal_history",
    "hist_gradient_boosting_temporal",
]
DEFAULT_EVENT_PRE_TURNS = 4
DEFAULT_EVENT_POST_TURNS = 6
DEFAULT_HIGH_CONFLICT_MIN_DISAGREEMENT_TURNS = 6
DEFAULT_HIGH_CONFLICT_MIN_DISAGREEMENT_RATE = 0.10


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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

    explicit_count = len(turn.get("explicit_propositions", []) or [])
    implicit_count = len(turn.get("assumptions", []) or [])
    total = explicit_count + implicit_count

    if total > 0:
        visible_prop = explicit_count / total
        context_prop = implicit_count / total
        iceberg_prop_per_sec = (explicit_count / total) / duration
        iceberg_ratio_per_sec = (explicit_count / (implicit_count + 1e-6)) / duration
    else:
        visible_prop = 0.0
        context_prop = 0.0
        iceberg_prop_per_sec = 0.0
        iceberg_ratio_per_sec = 0.0

    iceberg_log_ratio = math.log((explicit_count + 1.0) / (implicit_count + 1.0))
    iceberg_log_ratio_per_sec = iceberg_log_ratio - math.log(duration)
    iceberg_context_log_ratio = math.log((implicit_count + 1.0) / (explicit_count + 1.0))

    stance_value = float(stance)
    if stance_value >= 4.0:
        agreement_binary = 1.0
    elif stance_value <= 2.0:
        agreement_binary = 0.0
    else:
        agreement_binary = math.nan

    return {
        "speaker_id": str(turn.get("speaker_id") or ""),
        "stance_5pt": stance_value,
        "agreement_binary": agreement_binary,
        "explicit_count": explicit_count,
        "implicit_count": implicit_count,
        "duration": duration,
        "log_duration": math.log(duration),
        "start_time": start_time,
        "end_time": end_time,
        "iceberg_visible_prop": visible_prop,
        "iceberg_context_prop": context_prop,
        "iceberg_prop_per_sec": iceberg_prop_per_sec,
        "iceberg_ratio_per_sec": iceberg_ratio_per_sec,
        "iceberg_log_ratio": iceberg_log_ratio,
        "iceberg_log_ratio_per_sec": iceberg_log_ratio_per_sec,
        "iceberg_context_log_ratio": iceberg_context_log_ratio,
    }


def upgrade_cached_feature_frame(df_cached: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    upgraded = df_cached.copy()
    required_base_columns = {"explicit_count", "implicit_count", "duration"}
    missing_base_columns = sorted(required_base_columns - set(upgraded.columns))
    if missing_base_columns:
        return upgraded, missing_base_columns

    total = upgraded["explicit_count"].astype(float) + upgraded["implicit_count"].astype(float)
    safe_duration = upgraded["duration"].astype(float).clip(lower=0.1)

    if "log_duration" not in upgraded.columns:
        upgraded["log_duration"] = np.log(safe_duration)
    if "iceberg_visible_prop" not in upgraded.columns:
        upgraded["iceberg_visible_prop"] = np.where(total > 0.0, upgraded["explicit_count"].astype(float) / total, 0.0)
    if "iceberg_context_prop" not in upgraded.columns:
        upgraded["iceberg_context_prop"] = np.where(total > 0.0, upgraded["implicit_count"].astype(float) / total, 0.0)
    if "iceberg_context_log_ratio" not in upgraded.columns:
        upgraded["iceberg_context_log_ratio"] = np.log(
            (upgraded["implicit_count"].astype(float) + 1.0) / (upgraded["explicit_count"].astype(float) + 1.0)
        )

    return upgraded, []


def enrich_episode_frame(df_episode: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    df_episode = df_episode.sort_values("start_time").reset_index(drop=True).copy()

    max_end_time = float(df_episode["end_time"].max()) if len(df_episode) else 0.0
    if max_end_time > 0.0:
        df_episode["timeline_progress"] = df_episode["start_time"] / max_end_time
    else:
        df_episode["timeline_progress"] = 0.0
    df_episode["turn_progress"] = (
        np.arange(len(df_episode), dtype=float) / max(len(df_episode) - 1, 1) if len(df_episode) else 0.0
    )

    for metric_name in metric_names:
        speaker_mean_col = f"{metric_name}_speaker_mean"
        within_col = f"{metric_name}_within_speaker"
        prev_col = f"{metric_name}_same_speaker_prev"
        delta_col = f"{metric_name}_same_speaker_delta"
        episode_mean_col = f"{metric_name}_episode_mean"
        within_episode_col = f"{metric_name}_within_episode"

        df_episode[speaker_mean_col] = df_episode.groupby("speaker_id")[metric_name].transform("mean")
        df_episode[within_col] = df_episode[metric_name] - df_episode[speaker_mean_col]
        df_episode[prev_col] = df_episode.groupby("speaker_id")[metric_name].shift(1)
        df_episode[delta_col] = df_episode[metric_name] - df_episode[prev_col]
        df_episode[episode_mean_col] = float(df_episode[metric_name].mean())
        df_episode[within_episode_col] = df_episode[metric_name] - df_episode[episode_mean_col]

    return df_episode


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
    metric_names = [
        "iceberg_visible_prop",
        "iceberg_context_prop",
        "iceberg_prop_per_sec",
        "iceberg_ratio_per_sec",
        "iceberg_log_ratio",
        "iceberg_log_ratio_per_sec",
        "iceberg_context_log_ratio",
    ]

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
        df_episode = enrich_episode_frame(df_episode, metric_names)

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


def attach_stance_predictor_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df_model = df_input.sort_values(["episode", "start_time", "turn_idx"]).reset_index(drop=True).copy()
    df_model["stance_raw"] = df_model["stance_5pt"].astype(float)
    df_model["log_duration"] = np.log(df_model["duration"].clip(lower=0.1))
    df_model["stance_speaker_mean"] = df_model.groupby(["episode", "speaker_id"])["stance_raw"].transform("mean")
    df_model["stance_within_speaker"] = df_model["stance_raw"] - df_model["stance_speaker_mean"]
    df_model["stance_same_speaker_prev"] = df_model.groupby(["episode", "speaker_id"])["stance_raw"].shift(1)

    episode_groups = df_model.groupby("episode", sort=False)
    df_model["prev_turn_speaker_id"] = episode_groups["speaker_id"].shift(1)
    df_model["prev_turn_stance_raw"] = episode_groups["stance_raw"].shift(1)
    df_model["prev_prev_turn_stance_raw"] = episode_groups["stance_raw"].shift(2)
    df_model["prev_turn_agreement_binary"] = episode_groups["agreement_binary"].shift(1)
    df_model["prev_turn_iceberg_log_ratio"] = episode_groups["iceberg_log_ratio"].shift(1)
    df_model["prev_prev_turn_iceberg_log_ratio"] = episode_groups["iceberg_log_ratio"].shift(2)
    df_model["prev_turn_log_duration"] = episode_groups["log_duration"].shift(1)
    df_model["prev_turn_end_time"] = episode_groups["end_time"].shift(1)
    df_model["speaker_switch"] = np.where(
        df_model["prev_turn_speaker_id"].isna(),
        np.nan,
        (df_model["speaker_id"] != df_model["prev_turn_speaker_id"]).astype(float),
    )
    df_model["turn_gap_sec"] = (df_model["start_time"] - df_model["prev_turn_end_time"]).clip(lower=0.0)
    df_model["log_turn_gap_sec"] = np.log1p(df_model["turn_gap_sec"])
    df_model["stance_change_from_prev_turn"] = df_model["stance_raw"] - df_model["prev_turn_stance_raw"]
    df_model["prev_turn_iceberg_delta"] = (
        df_model["prev_turn_iceberg_log_ratio"] - df_model["prev_prev_turn_iceberg_log_ratio"]
    )
    df_model["prev_turn_stance_delta"] = df_model["prev_turn_stance_raw"] - df_model["prev_prev_turn_stance_raw"]
    df_model["speaker_switch_x_stance"] = df_model["speaker_switch"] * df_model["stance_raw"]
    df_model["prev_turn_target_x_stance"] = df_model["prev_turn_iceberg_log_ratio"] * df_model["stance_raw"]
    df_model["prev_turn_target_x_speaker_switch"] = (
        df_model["prev_turn_iceberg_log_ratio"] * df_model["speaker_switch"]
    )
    df_model["prev_same_speaker_iceberg_log_ratio"] = df_model["iceberg_log_ratio_same_speaker_prev"]

    prev_other_target = np.full(len(df_model), np.nan, dtype=float)
    prev_other_stance = np.full(len(df_model), np.nan, dtype=float)
    prev_other_log_duration = np.full(len(df_model), np.nan, dtype=float)

    for _, episode_index in df_model.groupby("episode", sort=False).groups.items():
        history_by_speaker: Dict[str, Dict[str, float]] = {}
        for order, row_idx in enumerate(episode_index):
            speaker_id = str(df_model.at[row_idx, "speaker_id"])
            other_candidates = [record for spk, record in history_by_speaker.items() if spk != speaker_id]
            if other_candidates:
                latest_other = max(other_candidates, key=lambda record: record["order"])
                prev_other_target[row_idx] = latest_other["target"]
                prev_other_stance[row_idx] = latest_other["stance"]
                prev_other_log_duration[row_idx] = latest_other["log_duration"]
            history_by_speaker[speaker_id] = {
                "order": float(order),
                "target": float(df_model.at[row_idx, "iceberg_log_ratio"]),
                "stance": float(df_model.at[row_idx, "stance_raw"]),
                "log_duration": float(df_model.at[row_idx, "log_duration"]),
            }

    df_model["prev_other_speaker_iceberg_log_ratio"] = prev_other_target
    df_model["prev_other_speaker_stance_raw"] = prev_other_stance
    df_model["prev_other_speaker_log_duration"] = prev_other_log_duration
    df_model["prev_other_target_gap"] = (
        df_model["prev_turn_iceberg_log_ratio"] - df_model["prev_other_speaker_iceberg_log_ratio"]
    )
    df_model["prev_other_stance_gap"] = df_model["stance_raw"] - df_model["prev_other_speaker_stance_raw"]
    return df_model


def build_regression_pipeline(model_name: str) -> Tuple[Pipeline, List[str], List[str]]:
    categorical_features = ["category"]
    estimator = Ridge(alpha=1.0)

    if model_name == "baseline_temporal_control":
        numeric_features = ["log_duration", "log_turn_gap_sec", "speaker_switch"]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                ("cat", make_one_hot_encoder(), categorical_features),
            ]
        )
    elif model_name == "linear_current_plus_history":
        numeric_features = [
            "log_duration",
            "log_turn_gap_sec",
            "speaker_switch",
            "stance_raw",
            "prev_turn_stance_raw",
            "prev_turn_iceberg_log_ratio",
            "prev_turn_iceberg_delta",
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                ("cat", make_one_hot_encoder(), categorical_features),
            ]
        )
    elif model_name == "linear_temporal_history":
        numeric_features = [
            "log_duration",
            "log_turn_gap_sec",
            "speaker_switch",
            "stance_raw",
            "prev_turn_stance_raw",
            "stance_speaker_mean",
            "stance_within_speaker",
            "prev_turn_iceberg_log_ratio",
            "prev_turn_iceberg_delta",
            "prev_same_speaker_iceberg_log_ratio",
            "prev_other_speaker_iceberg_log_ratio",
            "prev_other_speaker_stance_raw",
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                ("cat", make_one_hot_encoder(), categorical_features),
            ]
        )
    elif model_name == "linear_temporal_history_interactions":
        numeric_features = [
            "log_duration",
            "log_turn_gap_sec",
            "speaker_switch",
            "stance_raw",
            "prev_turn_stance_raw",
            "stance_speaker_mean",
            "stance_within_speaker",
            "prev_turn_iceberg_log_ratio",
            "prev_turn_iceberg_delta",
            "prev_same_speaker_iceberg_log_ratio",
            "prev_other_speaker_iceberg_log_ratio",
            "prev_other_speaker_stance_raw",
            "stance_change_from_prev_turn",
            "prev_turn_stance_delta",
            "speaker_switch_x_stance",
            "prev_turn_target_x_stance",
            "prev_turn_target_x_speaker_switch",
            "prev_other_target_gap",
            "prev_other_stance_gap",
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                ("cat", make_one_hot_encoder(), categorical_features),
            ]
        )
    elif model_name == "spline_temporal_history":
        numeric_features = [
            "log_duration",
            "log_turn_gap_sec",
            "speaker_switch",
            "prev_turn_stance_raw",
            "stance_speaker_mean",
            "prev_turn_iceberg_delta",
            "prev_other_speaker_iceberg_log_ratio",
            "prev_other_speaker_stance_raw",
            "prev_turn_target_x_stance",
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "stance_spline",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    ["stance_raw"],
                ),
                (
                    "history_spline",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    ["prev_turn_iceberg_log_ratio"],
                ),
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                ("cat", make_one_hot_encoder(), categorical_features),
            ]
        )
    elif model_name == "hist_gradient_boosting_temporal":
        numeric_features = [
            "log_duration",
            "log_turn_gap_sec",
            "speaker_switch",
            "stance_raw",
            "prev_turn_stance_raw",
            "stance_speaker_mean",
            "stance_within_speaker",
            "prev_turn_iceberg_log_ratio",
            "prev_turn_iceberg_delta",
            "prev_same_speaker_iceberg_log_ratio",
            "prev_other_speaker_iceberg_log_ratio",
            "prev_other_speaker_stance_raw",
            "prev_other_speaker_log_duration",
            "stance_change_from_prev_turn",
            "prev_turn_stance_delta",
            "speaker_switch_x_stance",
            "prev_turn_target_x_stance",
            "prev_turn_target_x_speaker_switch",
            "prev_other_target_gap",
            "prev_other_stance_gap",
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                ),
                ("cat", make_one_hot_encoder(), categorical_features),
            ]
        )
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    return pipeline, numeric_features, categorical_features


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    y_true_std = float(np.std(y_true))
    y_pred_std = float(np.std(y_pred))
    if np.isclose(y_true_std, 0.0) or np.isclose(y_pred_std, 0.0):
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_grouped_cv(
    df_model: pd.DataFrame,
    pipeline: Pipeline,
    group_col: str,
    target_col: str,
    n_splits: int,
) -> Tuple[dict, List[dict]]:
    groups = df_model[group_col].astype(str)
    X = df_model.copy()
    y = df_model[target_col].astype(float).to_numpy()

    unique_groups = groups.nunique()
    split_count = max(2, min(n_splits, unique_groups))
    splitter = GroupKFold(n_splits=split_count)

    fold_rows: List[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        x_train = X.iloc[train_idx]
        x_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        target_std = float(np.std(y_test))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
        fold_row = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "target_mean_test": float(np.mean(y_test)),
            "target_std_test": target_std,
            "r2": float(r2_score(y_test, y_pred)),
            "pearson_r": compute_pearson_r(y_test, y_pred),
            "mae": mae,
            "rmse": rmse,
            "mae_over_std": float(mae / target_std) if not np.isclose(target_std, 0.0) else math.nan,
        }
        fold_rows.append(fold_row)

    metric_keys = ["r2", "pearson_r", "mae", "rmse", "mae_over_std"]
    metrics = {f"{key}_mean": float(np.nanmean([row[key] for row in fold_rows])) for key in metric_keys}
    metrics.update({f"{key}_std": float(np.nanstd([row[key] for row in fold_rows])) for key in metric_keys})
    metrics["fold_count"] = int(split_count)
    return metrics, fold_rows


def compare_target_metric_candidates(
    df_target: pd.DataFrame,
    metric_candidates: Sequence[str],
    n_splits: int,
) -> Tuple[pd.DataFrame, str]:
    rows: List[dict] = []
    for metric_name in metric_candidates:
        pipeline, _, _ = build_regression_pipeline("linear_temporal_history")
        metrics, _ = evaluate_grouped_cv(
            df_model=df_target,
            pipeline=pipeline,
            group_col="episode",
            target_col=metric_name,
            n_splits=n_splits,
        )
        row = {"metric_name": metric_name, **metrics}
        rows.append(row)
        logger.info(
            "Metric comparison | metric=%s | r2=%.4f | pearson=%.4f | rmse=%.4f",
            metric_name,
            metrics["r2_mean"],
            metrics["pearson_r_mean"],
            metrics["rmse_mean"],
        )

    df_results = pd.DataFrame(rows).sort_values(
        by=["r2_mean", "pearson_r_mean", "mae_over_std_mean"],
        ascending=[False, False, True],
    )
    best_metric = str(df_results.iloc[0]["metric_name"])
    return df_results, best_metric


def compare_model_candidates(
    df_target: pd.DataFrame,
    target_metric: str,
    model_candidates: Sequence[str],
    n_splits: int,
) -> Tuple[pd.DataFrame, str]:
    rows: List[dict] = []

    for model_name in model_candidates:
        pipeline, _, _ = build_regression_pipeline(model_name)
        metrics, _ = evaluate_grouped_cv(
            df_model=df_target,
            pipeline=pipeline,
            group_col="episode",
            target_col=target_metric,
            n_splits=n_splits,
        )
        row = {"model_name": model_name, "target_metric": target_metric, **metrics}
        rows.append(row)
        logger.info(
            "Model comparison | model=%s | r2=%.4f | pearson=%.4f | rmse=%.4f",
            model_name,
            metrics["r2_mean"],
            metrics["pearson_r_mean"],
            metrics["rmse_mean"],
        )

    df_results = pd.DataFrame(rows).sort_values(
        by=["r2_mean", "pearson_r_mean", "rmse_mean", "mae_mean"],
        ascending=[False, False, True, True],
    )
    best_model = str(df_results.iloc[0]["model_name"])
    return df_results, best_model


def fit_full_model(
    df_target: pd.DataFrame,
    target_metric: str,
    model_name: str,
) -> Pipeline:
    pipeline, _, _ = build_regression_pipeline(model_name)
    pipeline.fit(df_target, df_target[target_metric].astype(float).to_numpy())
    return pipeline


def extract_coefficient_table(
    pipeline: Pipeline,
    df_target: Optional[pd.DataFrame] = None,
    target_metric: Optional[str] = None,
) -> pd.DataFrame:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        feature_names = preprocess.get_feature_names_out()
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df["importance_source"] = "model_coefficient"
        return coef_df.sort_values(["abs_coefficient", "feature"], ascending=[False, True]).reset_index(drop=True)

    if df_target is None or target_metric is None:
        raise ValueError("df_target and target_metric are required for model families without coefficients.")

    predictor_cols = [
        "category",
        "log_duration",
        "log_turn_gap_sec",
        "speaker_switch",
        "stance_raw",
        "prev_turn_stance_raw",
        "stance_speaker_mean",
        "stance_within_speaker",
        "prev_turn_iceberg_log_ratio",
        "prev_turn_iceberg_delta",
        "prev_same_speaker_iceberg_log_ratio",
        "prev_other_speaker_iceberg_log_ratio",
        "prev_other_speaker_stance_raw",
        "prev_other_speaker_log_duration",
        "stance_change_from_prev_turn",
        "prev_turn_stance_delta",
        "speaker_switch_x_stance",
        "prev_turn_target_x_stance",
        "prev_turn_target_x_speaker_switch",
        "prev_other_target_gap",
        "prev_other_stance_gap",
    ]
    available_predictors = [col for col in predictor_cols if col in df_target.columns]
    sample_size = min(12000, len(df_target))
    df_sample = df_target.sample(n=sample_size, random_state=42) if len(df_target) > sample_size else df_target.copy()
    perm_result = permutation_importance(
        estimator=pipeline,
        X=df_sample[available_predictors],
        y=df_sample[target_metric].astype(float).to_numpy(),
        n_repeats=3,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    coef_df = pd.DataFrame(
        {
            "feature": available_predictors,
            "coefficient": perm_result.importances_mean,
            "coefficient_std": perm_result.importances_std,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["importance_source"] = "permutation_importance"
    return coef_df.sort_values(["abs_coefficient", "feature"], ascending=[False, True]).reset_index(drop=True)


def summarize_group_curves(group_df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if group_df.empty:
        return summary

    group_df = group_df.copy()
    group_df["timeline_progress"] = pd.to_numeric(group_df["timeline_progress"], errors="coerce")
    group_df = group_df.dropna(subset=["timeline_progress"])

    for group_name in ["agreement", "disagreement"]:
        subset = group_df[group_df["stance_group"] == group_name].sort_values("timeline_progress").reset_index(drop=True)
        if subset.empty:
            continue
        summary[f"{group_name}_progress_delta"] = float(subset[value_col].iloc[-1] - subset[value_col].iloc[0])

    for progress in [0.1, 0.5, 0.9]:
        agreement_subset = group_df[group_df["stance_group"] == "agreement"].copy()
        disagreement_subset = group_df[group_df["stance_group"] == "disagreement"].copy()
        if agreement_subset.empty or disagreement_subset.empty:
            continue
        agreement_idx = (agreement_subset["timeline_progress"] - progress).abs().idxmin()
        disagreement_idx = (disagreement_subset["timeline_progress"] - progress).abs().idxmin()
        summary[f"agreement_minus_disagreement_progress_{progress:.1f}"] = float(
            agreement_subset.loc[agreement_idx, value_col] - disagreement_subset.loc[disagreement_idx, value_col]
        )
    return summary


def build_predicted_progress_curves(
    pipeline: Pipeline,
    df_model: pd.DataFrame,
    output_dir: Path,
    target_metric: str,
    progress_bins: int = 10,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    df_group_prediction = df_model[df_model["agreement_binary"].notna()].copy()
    if df_group_prediction.empty:
        empty_df = pd.DataFrame(
            columns=[
                "progress_bin",
                "agreement_binary",
                "predicted_iceberg",
                "timeline_progress",
                "stance_group",
                "predicted_explicitness",
                "predicted_context",
            ]
        )
        return {}, empty_df
    df_group_prediction["predicted_iceberg"] = pipeline.predict(df_group_prediction)
    bin_edges = np.linspace(0.0, 1.0, progress_bins + 1)
    df_group_prediction["progress_bin"] = pd.cut(
        df_group_prediction["timeline_progress"], bins=bin_edges, include_lowest=True
    )
    group_df = (
        df_group_prediction.groupby(["progress_bin", "agreement_binary"], observed=False)["predicted_iceberg"]
        .mean()
        .reset_index()
    )
    group_df["timeline_progress"] = group_df["progress_bin"].apply(
        lambda interval: float((interval.left + interval.right) / 2.0) if pd.notna(interval) else math.nan
    )
    group_df["timeline_progress"] = pd.to_numeric(group_df["timeline_progress"], errors="coerce")
    group_df["stance_group"] = group_df["agreement_binary"].map({0.0: "disagreement", 1.0: "agreement"})
    group_df = group_df.dropna(subset=["timeline_progress", "stance_group"]).reset_index(drop=True)
    group_df["predicted_explicitness"] = group_df["predicted_iceberg"]
    group_df["predicted_context"] = group_df["predicted_iceberg"]
    summary = summarize_group_curves(group_df=group_df, value_col="predicted_iceberg")
    return summary, group_df


def build_observed_progress_curves(
    df_model: pd.DataFrame,
    output_dir: Path,
    target_metric: str,
    progress_bins: int = 10,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    df_observed = df_model[df_model["agreement_binary"].notna()].copy()
    if df_observed.empty:
        empty_df = pd.DataFrame(columns=["timeline_progress", target_metric, "n", "stance_group"])
        return {}, empty_df

    bin_edges = np.linspace(0.0, 1.0, progress_bins + 1)
    df_observed["progress_bin"] = pd.cut(df_observed["timeline_progress"], bins=bin_edges, include_lowest=True)

    grouped = (
        df_observed.groupby(["progress_bin", "agreement_binary"], observed=False)[target_metric]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "observed_target", "size": "n"})
    )
    grouped["timeline_progress"] = grouped["progress_bin"].apply(
        lambda interval: float((interval.left + interval.right) / 2.0) if pd.notna(interval) else math.nan
    )
    grouped["timeline_progress"] = pd.to_numeric(grouped["timeline_progress"], errors="coerce")
    grouped["stance_group"] = grouped["agreement_binary"].map({0.0: "disagreement", 1.0: "agreement"})
    grouped["observed_explicitness"] = grouped["observed_target"]
    grouped["observed_context"] = grouped["observed_target"]
    grouped = grouped.dropna(subset=["timeline_progress", "stance_group"]).reset_index(drop=True)
    summary = summarize_group_curves(group_df=grouped, value_col="observed_target")
    return summary, grouped


def save_curve_comparison(
    predicted_group_df: pd.DataFrame,
    observed_group_df: pd.DataFrame,
    output_dir: Path,
    target_metric: str,
) -> pd.DataFrame:
    comparison_frames: List[pd.DataFrame] = []

    if not predicted_group_df.empty:
        comparison_frames.append(
            predicted_group_df.rename(columns={"predicted_iceberg": "target_value"}).assign(curve_source="predicted")
        )
    if not observed_group_df.empty:
        comparison_frames.append(
            observed_group_df.rename(columns={"observed_target": "target_value"}).assign(curve_source="observed")
        )

    if comparison_frames:
        comparison_df = pd.concat(comparison_frames, ignore_index=True)
    else:
        comparison_df = pd.DataFrame(
            columns=[
                "progress_bin",
                "agreement_binary",
                "target_value",
                "timeline_progress",
                "stance_group",
                "curve_source",
            ]
        )
    comparison_df.to_csv(output_dir / "agreement_disagreement_progress_curve_comparison.csv", index=False)
    return comparison_df


def identify_high_conflict_episodes(
    df_model: pd.DataFrame,
    min_disagreement_turns: int,
    min_disagreement_rate: float,
) -> Tuple[Set[str], pd.DataFrame]:
    df_labeled = df_model[df_model["agreement_binary"].notna()].copy()
    if df_labeled.empty:
        empty_df = pd.DataFrame(
            columns=[
                "episode",
                "category",
                "labeled_turns",
                "agreement_turns",
                "disagreement_turns",
                "disagreement_rate",
                "high_conflict",
            ]
        )
        return set(), empty_df

    episode_stats = (
        df_labeled.groupby("episode", observed=False)
        .agg(
            category=("category", lambda s: str(pd.Series(s).mode().iloc[0]) if not pd.Series(s).mode().empty else "unknown"),
            labeled_turns=("agreement_binary", "size"),
            agreement_turns=("agreement_binary", lambda s: int((s == 1.0).sum())),
            disagreement_turns=("agreement_binary", lambda s: int((s == 0.0).sum())),
        )
        .reset_index()
    )
    episode_stats["disagreement_rate"] = np.where(
        episode_stats["labeled_turns"] > 0,
        episode_stats["disagreement_turns"] / episode_stats["labeled_turns"],
        0.0,
    )
    episode_stats["high_conflict"] = (
        (episode_stats["disagreement_turns"] >= int(min_disagreement_turns))
        & (episode_stats["disagreement_rate"] >= float(min_disagreement_rate))
    )
    high_conflict_episodes = set(episode_stats.loc[episode_stats["high_conflict"], "episode"].astype(str))
    return high_conflict_episodes, episode_stats.sort_values(
        ["high_conflict", "disagreement_turns", "disagreement_rate", "episode"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_disagreement_onset_event_study(
    df_model: pd.DataFrame,
    output_dir: Path,
    target_metric: str,
    pre_event_turns: int,
    post_event_turns: int,
    high_conflict_min_disagreement_turns: int,
    high_conflict_min_disagreement_rate: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    high_conflict_episodes, episode_stats_df = identify_high_conflict_episodes(
        df_model=df_model,
        min_disagreement_turns=high_conflict_min_disagreement_turns,
        min_disagreement_rate=high_conflict_min_disagreement_rate,
    )

    onset_mask = (
        df_model["agreement_binary"].eq(0.0)
        & df_model["prev_turn_agreement_binary"].eq(1.0)
        & df_model["speaker_switch"].eq(1.0)
        & df_model[target_metric].notna()
    )
    onset_indices = list(df_model.index[onset_mask])

    event_rows: List[dict] = []
    high_conflict_event_count = 0
    for onset_idx in onset_indices:
        episode = str(df_model.at[onset_idx, "episode"])
        prev_row_idx = onset_idx - 1
        if prev_row_idx < 0 or str(df_model.at[prev_row_idx, "episode"]) != episode:
            continue
        next_row_idx = onset_idx + 1
        if next_row_idx >= len(df_model) or str(df_model.at[next_row_idx, "episode"]) != episode:
            continue
        onset_speaker_id = str(df_model.at[onset_idx, "speaker_id"])
        response_speaker_id = str(df_model.at[next_row_idx, "speaker_id"])
        if response_speaker_id == onset_speaker_id:
            continue
        baseline_value = df_model.at[prev_row_idx, target_metric]
        if pd.isna(baseline_value):
            continue
        baseline_value = float(baseline_value)
        onset_value = df_model.at[onset_idx, target_metric]
        response_value = df_model.at[next_row_idx, target_metric]
        if pd.isna(onset_value) or pd.isna(response_value):
            continue
        onset_value = float(onset_value)
        response_value = float(response_value)
        onset_turn_idx = int(df_model.at[onset_idx, "turn_idx"])
        onset_category = str(df_model.at[onset_idx, "category"])
        event_id = f"{episode}::turn_{onset_turn_idx}"
        is_high_conflict = episode in high_conflict_episodes
        if is_high_conflict:
            high_conflict_event_count += 1

        for event_time in range(-int(pre_event_turns), int(post_event_turns) + 1):
            row_idx = onset_idx + event_time
            if row_idx < 0 or row_idx >= len(df_model):
                continue
            if str(df_model.at[row_idx, "episode"]) != episode:
                continue

            target_value = df_model.at[row_idx, target_metric]
            if pd.isna(target_value):
                continue

            event_rows.append(
                {
                    "event_id": event_id,
                    "episode": episode,
                    "category": onset_category,
                    "onset_turn_idx": onset_turn_idx,
                    "event_time": int(event_time),
                    "target_value": float(target_value),
                    "baseline_prev_turn_value": baseline_value,
                    "onset_turn_value": onset_value,
                    "response_turn_value": response_value,
                    "delta_from_prev_turn": float(target_value - baseline_value),
                    "delta_from_onset_turn": float(target_value - onset_value),
                    "stance_at_offset": float(df_model.at[row_idx, "stance_raw"]),
                    "agreement_binary_at_offset": df_model.at[row_idx, "agreement_binary"],
                    "high_conflict": bool(is_high_conflict),
                }
            )

    if not event_rows:
        empty_df = pd.DataFrame(
            columns=[
                "event_time",
                "mean_target",
                "mean_delta_from_prev_turn",
                "mean_delta_from_onset_turn",
                "se_target",
                "se_delta_from_prev_turn",
                "se_delta_from_onset_turn",
                "ci_low_target",
                "ci_high_target",
                "ci_low_prev_turn",
                "ci_high_prev_turn",
                "ci_low_onset_turn",
                "ci_high_onset_turn",
                "n_rows",
                "n_events",
            ]
        )
        empty_df.to_csv(output_dir / "disagreement_onset_event_study.csv", index=False)
        summary = {
            "target_metric": target_metric,
            "clean_onset_definition": "current disagreement turn after agreement turn with speaker switch",
            "response_turn_requirement": "turn +1 must stay in the same episode and be spoken by the other speaker",
            "pre_event_turns": int(pre_event_turns),
            "post_event_turns": int(post_event_turns),
            "high_conflict_min_disagreement_turns": int(high_conflict_min_disagreement_turns),
            "high_conflict_min_disagreement_rate": float(high_conflict_min_disagreement_rate),
            "response_valid_clean_onsets": {"n_events": 0},
        }
        return summary, empty_df

    event_df = pd.DataFrame(event_rows)
    summary_df = (
        event_df.groupby("event_time", observed=False)
        .agg(
            mean_target=("target_value", "mean"),
            std_target=("target_value", "std"),
            mean_delta_from_prev_turn=("delta_from_prev_turn", "mean"),
            std_delta_from_prev_turn=("delta_from_prev_turn", "std"),
            mean_delta_from_onset_turn=("delta_from_onset_turn", "mean"),
            std_delta_from_onset_turn=("delta_from_onset_turn", "std"),
            n_rows=("target_value", "size"),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    summary_df["se_target"] = np.where(
        summary_df["n_rows"] > 1,
        summary_df["std_target"].fillna(0.0) / np.sqrt(summary_df["n_rows"]),
        0.0,
    )
    summary_df["se_delta_from_prev_turn"] = np.where(
        summary_df["n_rows"] > 1,
        summary_df["std_delta_from_prev_turn"].fillna(0.0) / np.sqrt(summary_df["n_rows"]),
        0.0,
    )
    summary_df["se_delta_from_onset_turn"] = np.where(
        summary_df["n_rows"] > 1,
        summary_df["std_delta_from_onset_turn"].fillna(0.0) / np.sqrt(summary_df["n_rows"]),
        0.0,
    )
    summary_df["ci_low_target"] = summary_df["mean_target"] - 1.96 * summary_df["se_target"]
    summary_df["ci_high_target"] = summary_df["mean_target"] + 1.96 * summary_df["se_target"]
    summary_df["ci_low_prev_turn"] = summary_df["mean_delta_from_prev_turn"] - 1.96 * summary_df["se_delta_from_prev_turn"]
    summary_df["ci_high_prev_turn"] = summary_df["mean_delta_from_prev_turn"] + 1.96 * summary_df["se_delta_from_prev_turn"]
    summary_df["ci_low_onset_turn"] = summary_df["mean_delta_from_onset_turn"] - 1.96 * summary_df["se_delta_from_onset_turn"]
    summary_df["ci_high_onset_turn"] = summary_df["mean_delta_from_onset_turn"] + 1.96 * summary_df["se_delta_from_onset_turn"]
    summary_df.to_csv(output_dir / "disagreement_onset_event_study.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharex=True)
    level_color = "#bc4749"
    response_color = "#2a9d8f"
    for axis in axes:
        axis.axvline(0.0, color="#495057", linewidth=1.1, linestyle="--", alpha=0.85)
        axis.axvline(1.0, color="#0f4c5c", linewidth=1.1, linestyle=":", alpha=0.85)
        axis.grid(alpha=0.16, linewidth=0.6)
        axis.set_xticks(list(range(-int(pre_event_turns), int(post_event_turns) + 1)))
        axis.set_xlabel("Turns from disagreement onset")

    axes[0].plot(summary_df["event_time"], summary_df["mean_target"], color=level_color, linewidth=2.2, marker="o", markersize=4)
    axes[0].fill_between(
        summary_df["event_time"],
        summary_df["ci_low_target"],
        summary_df["ci_high_target"],
        color=level_color,
        alpha=0.18,
    )
    axes[0].set_title("Observed Explicit/Implicit Level", fontsize=11, pad=10)
    axes[0].set_ylabel(target_metric)

    axes[1].axhline(0.0, color="#6c757d", linewidth=1.0, alpha=0.8)
    axes[1].plot(
        summary_df["event_time"],
        summary_df["mean_delta_from_onset_turn"],
        color=response_color,
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    axes[1].fill_between(
        summary_df["event_time"],
        summary_df["ci_low_onset_turn"],
        summary_df["ci_high_onset_turn"],
        color=response_color,
        alpha=0.18,
    )
    axes[1].set_title("Change Relative to Disagreement Turn (t=0)", fontsize=11, pad=10)
    axes[1].set_ylabel(f"{target_metric} - value at turn 0")

    fig.suptitle("Experiment 2: Response-Validated Disagreement Onsets", fontsize=15, y=0.95)
    fig.subplots_adjust(top=0.84, left=0.08, right=0.98, bottom=0.12, wspace=0.18)
    fig.savefig(output_dir / "experiment2_disagreement_onset_event_study.png", dpi=200)
    plt.close(fig)

    onset_row = summary_df[summary_df["event_time"] == 0]
    response_row = summary_df[summary_df["event_time"] == 1]
    pre_row = summary_df[summary_df["event_time"] == -1]
    min_row = summary_df.loc[summary_df["mean_target"].idxmin()]
    summary = {
        "target_metric": target_metric,
        "clean_onset_definition": "current disagreement turn after agreement turn with speaker switch",
        "response_turn_requirement": "turn +1 must stay in the same episode and be spoken by the other speaker",
        "pre_event_turns": int(pre_event_turns),
        "post_event_turns": int(post_event_turns),
        "high_conflict_min_disagreement_turns": int(high_conflict_min_disagreement_turns),
        "high_conflict_min_disagreement_rate": float(high_conflict_min_disagreement_rate),
        "high_conflict_episode_count": int(len(high_conflict_episodes)),
        "high_conflict_response_valid_onset_count": int(high_conflict_event_count),
        "response_valid_clean_onsets": {
            "n_events": int(event_df["event_id"].nunique()),
            "mean_target_at_pre_turn": float(pre_row["mean_target"].iloc[0]) if not pre_row.empty else math.nan,
            "mean_target_at_onset": float(onset_row["mean_target"].iloc[0]) if not onset_row.empty else math.nan,
            "mean_target_at_response_turn": float(response_row["mean_target"].iloc[0]) if not response_row.empty else math.nan,
            "onset_minus_pre_turn": float(onset_row["mean_delta_from_prev_turn"].iloc[0]) if not onset_row.empty else math.nan,
            "response_minus_pre_turn": float(response_row["mean_delta_from_prev_turn"].iloc[0]) if not response_row.empty else math.nan,
            "response_minus_onset_turn": float(response_row["mean_delta_from_onset_turn"].iloc[0]) if not response_row.empty else math.nan,
            "lowest_mean_target_event_time": int(min_row["event_time"]),
            "lowest_mean_target_value": float(min_row["mean_target"]),
        },
    }
    return summary, summary_df


def build_transition_heatmap_comparison(
    pipeline: Pipeline,
    df_model: pd.DataFrame,
    output_dir: Path,
    target_metric: str,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    response_df = df_model[
        (df_model["speaker_switch"] >= 0.5)
        & df_model["prev_turn_stance_raw"].notna()
        & df_model["stance_raw"].notna()
    ].copy()
    if response_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "previous_stance_level",
                "current_stance_level",
                "observed_target",
                "predicted_iceberg",
                "n",
                "prediction_error",
            ]
        )
        empty_df.to_csv(output_dir / "temporal_transition_heatmap_comparison.csv", index=False)
        return {"response_turn_rows": 0, "target_metric": target_metric}, empty_df

    response_df["predicted_iceberg"] = pipeline.predict(response_df)

    observed_grouped = (
        response_df.groupby(["prev_turn_stance_raw", "stance_raw"], observed=False)[target_metric]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "observed_target", "size": "n"})
    )
    predicted_grouped = (
        response_df.groupby(["prev_turn_stance_raw", "stance_raw"], observed=False)["predicted_iceberg"]
        .mean()
        .reset_index()
    )
    comparison_df = observed_grouped.merge(
        predicted_grouped,
        on=["prev_turn_stance_raw", "stance_raw"],
        how="outer",
    )
    comparison_df = comparison_df.rename(
        columns={
            "prev_turn_stance_raw": "previous_stance_level",
            "stance_raw": "current_stance_level",
        }
    )
    comparison_df["previous_stance_level"] = comparison_df["previous_stance_level"].astype(int)
    comparison_df["current_stance_level"] = comparison_df["current_stance_level"].astype(int)
    comparison_df["prediction_error"] = comparison_df["predicted_iceberg"] - comparison_df["observed_target"]
    comparison_df = comparison_df.sort_values(
        ["current_stance_level", "previous_stance_level"]
    ).reset_index(drop=True)
    comparison_df.to_csv(output_dir / "temporal_transition_heatmap_comparison.csv", index=False)

    summary: Dict[str, object] = {
        "response_turn_rows": int(len(response_df)),
        "transition_cell_count": int(len(comparison_df)),
        "target_metric": target_metric,
    }

    for prefix, value_col in [("observed", "observed_target"), ("predicted", "predicted_iceberg")]:
        subset = comparison_df.dropna(subset=[value_col])
        if subset.empty:
            continue
        max_row = subset.loc[subset[value_col].idxmax()]
        min_row = subset.loc[subset[value_col].idxmin()]
        summary[f"{prefix}_max_transition"] = {
            "previous_stance_level": int(max_row["previous_stance_level"]),
            "current_stance_level": int(max_row["current_stance_level"]),
            "value": float(max_row[value_col]),
        }
        summary[f"{prefix}_min_transition"] = {
            "previous_stance_level": int(min_row["previous_stance_level"]),
            "current_stance_level": int(min_row["current_stance_level"]),
            "value": float(min_row[value_col]),
        }
    return summary, comparison_df


def save_temporal_summary_figure(
    transition_comparison_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    output_dir: Path,
    target_metric: str,
    best_model: str,
    best_model_cv: Dict[str, float],
) -> None:
    fig = plt.figure(figsize=(14.5, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.35, wspace=0.28)
    ax_observed = fig.add_subplot(grid[0, 0])
    ax_predicted = fig.add_subplot(grid[0, 1])
    ax_importance = fig.add_subplot(grid[1, :])

    levels = [1, 2, 3, 4, 5]
    observed_matrix = (
        transition_comparison_df.pivot(
            index="current_stance_level", columns="previous_stance_level", values="observed_target"
        )
        .reindex(index=levels, columns=levels)
        .astype(float)
    )
    predicted_matrix = (
        transition_comparison_df.pivot(
            index="current_stance_level", columns="previous_stance_level", values="predicted_iceberg"
        )
        .reindex(index=levels, columns=levels)
        .astype(float)
    )

    valid_values = np.concatenate(
        [
            observed_matrix.to_numpy(dtype=float).ravel(),
            predicted_matrix.to_numpy(dtype=float).ravel(),
        ]
    )
    valid_values = valid_values[~np.isnan(valid_values)]
    max_abs_value = float(np.abs(valid_values).max()) if len(valid_values) else 0.1
    max_abs_value = max(max_abs_value, 0.1)
    color_norm = TwoSlopeNorm(vmin=-max_abs_value, vcenter=0.0, vmax=max_abs_value)

    heatmap_artist = None
    for axis, matrix, panel_title in [
        (ax_observed, observed_matrix, "Observed Response-Turn Means"),
        (ax_predicted, predicted_matrix, "Model-Predicted Response-Turn Means"),
    ]:
        heatmap_artist = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", norm=color_norm, origin="lower", aspect="auto")
        axis.set_title(panel_title, fontsize=12, pad=10)
        axis.set_xlabel("Previous turn stance")
        axis.set_ylabel("Current turn stance")
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels(levels)
        axis.set_yticks(range(len(levels)))
        axis.set_yticklabels(levels)
        for row_idx, current_level in enumerate(levels):
            for col_idx, previous_level in enumerate(levels):
                value = matrix.loc[current_level, previous_level]
                if pd.isna(value):
                    continue
                text_color = "white" if abs(value) > (0.45 * max_abs_value) else "#1f2933"
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

    if heatmap_artist is not None:
        colorbar = fig.colorbar(heatmap_artist, ax=[ax_observed, ax_predicted], shrink=0.86, pad=0.02)
        colorbar.set_label(target_metric)

    top_features = coef_df.sort_values("abs_coefficient", ascending=False).head(10).copy()
    top_features = top_features.sort_values("abs_coefficient", ascending=True)
    control_features = {"log_duration", "category", "prev_other_speaker_log_duration"}
    bar_colors = ["#8d99ae" if feature in control_features else "#2a9d8f" for feature in top_features["feature"]]
    ax_importance.barh(top_features["feature"], top_features["abs_coefficient"], color=bar_colors, alpha=0.92)
    ax_importance.set_title("Top Predictors (Permutation Importance)", fontsize=12, pad=10)
    ax_importance.set_xlabel("Held-out importance")
    text_offset = max(float(top_features["abs_coefficient"].max()) * 0.03, 0.0002) if not top_features.empty else 0.0002
    for value, feature in zip(top_features["abs_coefficient"], top_features["feature"]):
        ax_importance.text(value + text_offset, feature, f"{value:.4f}", va="center", fontsize=8.5)
    ax_importance.grid(axis="x", alpha=0.18, linewidth=0.6)

    fig.suptitle("Experiment 2: Temporal Explicitness Summary", fontsize=16, y=0.98)
    fig.text(
        0.015,
        0.942,
        (
            f"Best model: {best_model} | Held-out R^2={best_model_cv['r2_mean']:.3f} | "
            f"Pearson r={best_model_cv['pearson_r_mean']:.3f} | RMSE={best_model_cv['rmse_mean']:.3f}"
        ),
        fontsize=11,
    )
    fig.text(
        0.015,
        0.918,
        "Heatmaps use response turns only (speaker switches). Conversation progress is not used as a predictor or shown in this figure.",
        fontsize=9.5,
        color="#52606d",
    )
    fig.subplots_adjust(top=0.86, left=0.07, right=0.98, bottom=0.07, hspace=0.38, wspace=0.28)
    fig.savefig(output_dir / "experiment2_temporal_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare iceberg metrics and regression model families for Experiment 2, then fit a grouped "
            "model that predicts explicitness from lagged stance and turn history."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--category_data_subdir", type=str, default=DEFAULT_CATEGORY_DATA_SUBDIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_turns", type=int, default=12)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--rebuild_features", action="store_true")
    parser.add_argument("--allow_non_dyadic", action="store_true")
    parser.add_argument("--event_pre_turns", type=int, default=DEFAULT_EVENT_PRE_TURNS)
    parser.add_argument("--event_post_turns", type=int, default=DEFAULT_EVENT_POST_TURNS)
    parser.add_argument(
        "--high_conflict_min_disagreement_turns",
        type=int,
        default=DEFAULT_HIGH_CONFLICT_MIN_DISAGREEMENT_TURNS,
    )
    parser.add_argument(
        "--high_conflict_min_disagreement_rate",
        type=float,
        default=DEFAULT_HIGH_CONFLICT_MIN_DISAGREEMENT_RATE,
    )
    parser.add_argument(
        "--feature_cache_path",
        type=str,
        default=DEFAULT_FEATURE_CACHE_PATH,
        help="Path for the cached turn-level feature table. Defaults to the shared Experiment 2 cache directory.",
    )
    parser.add_argument("--feature_cache_name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--target_metric_candidates",
        nargs="+",
        default=DEFAULT_TARGET_METRIC_CANDIDATES,
        help="Explicit-over-implicit targets to compare before selecting the best one.",
    )
    parser.add_argument(
        "--model_candidates",
        nargs="+",
        default=DEFAULT_MODEL_CANDIDATES,
        help="Regression model families to compare after selecting the best explicitness target.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_path = output_dir / args.feature_cache_name if args.feature_cache_name else Path(args.feature_cache_path)
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)

    use_cache = feature_cache_path.exists() and not args.rebuild_features
    if use_cache:
        logger.info("Loading cached features from %s", feature_cache_path)
        df_all = pd.read_csv(feature_cache_path)
        missing_metric_columns = [metric for metric in args.target_metric_candidates if metric not in df_all.columns]
        if missing_metric_columns:
            logger.info("Cached features missing requested target columns %s; upgrading cache in place.", missing_metric_columns)
            df_all, missing_base_columns = upgrade_cached_feature_frame(df_all)
            if missing_base_columns:
                logger.info(
                    "Cached features missing base columns %s required for upgrade; rebuilding feature cache.",
                    missing_base_columns,
                )
                use_cache = False
            else:
                df_all.to_csv(feature_cache_path, index=False)
                logger.info("Updated cached features at %s", feature_cache_path)
    if not use_cache:
        df_all, dataset_summary = build_turn_level_features(
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
        dataset_summary = {
            "loaded_from_cache": True,
            "turn_rows": int(len(df_all)),
            "binary_rows": int(df_all["agreement_binary"].notna().sum()),
            "categories": df_all["category"].value_counts().sort_index().to_dict(),
            "stance_distribution": df_all["stance_5pt"].value_counts().sort_index().to_dict(),
        }

        dataset_summary["target_metric_candidates"] = list(args.target_metric_candidates)
    dataset_summary["model_candidates"] = list(args.model_candidates)
    dataset_summary["predictor_family"] = "lagged_stance_and_turn_history_to_explicitness"
    dataset_summary["hypothesis_alignment"] = "higher target values mean more explicit relative to implicit content"
    dataset_summary["progress_used_as_predictor"] = False
    dataset_summary["progress_curve_role"] = "diagnostic_only"
    dataset_summary["feature_cache_path"] = str(feature_cache_path)
    dataset_summary["event_study_pre_turns"] = int(args.event_pre_turns)
    dataset_summary["event_study_post_turns"] = int(args.event_post_turns)
    dataset_summary["high_conflict_min_disagreement_turns"] = int(args.high_conflict_min_disagreement_turns)
    dataset_summary["high_conflict_min_disagreement_rate"] = float(args.high_conflict_min_disagreement_rate)
    save_json(output_dir / "dataset_summary.json", dataset_summary)

    df_model = attach_stance_predictor_columns(df_all)

    metric_results, best_metric = compare_target_metric_candidates(
        df_target=df_model,
        metric_candidates=args.target_metric_candidates,
        n_splits=args.cv_splits,
    )

    model_results, best_model = compare_model_candidates(
        df_target=df_model,
        target_metric=best_metric,
        model_candidates=args.model_candidates,
        n_splits=args.cv_splits,
    )

    best_pipeline = fit_full_model(
        df_target=df_model,
        target_metric=best_metric,
        model_name=best_model,
    )
    coef_df = extract_coefficient_table(
        pipeline=best_pipeline,
        df_target=df_model,
        target_metric=best_metric,
    )
    coef_df.to_csv(output_dir / "best_model_coefficients.csv", index=False)

    predicted_curve_summary, predicted_group_df = build_predicted_progress_curves(
        pipeline=best_pipeline,
        df_model=df_model,
        output_dir=output_dir,
        target_metric=best_metric,
    )
    observed_curve_summary, observed_group_df = build_observed_progress_curves(
        df_model=df_model,
        output_dir=output_dir,
        target_metric=best_metric,
    )
    disagreement_onset_event_study_summary, _ = build_disagreement_onset_event_study(
        df_model=df_model,
        output_dir=output_dir,
        target_metric=best_metric,
        pre_event_turns=args.event_pre_turns,
        post_event_turns=args.event_post_turns,
        high_conflict_min_disagreement_turns=args.high_conflict_min_disagreement_turns,
        high_conflict_min_disagreement_rate=args.high_conflict_min_disagreement_rate,
    )

    summary_payload = {
        "best_target_metric": best_metric,
        "best_model": best_model,
        "dataset_summary": dataset_summary,
        "best_metric_cv": metric_results[metric_results["metric_name"] == best_metric].iloc[0].to_dict(),
        "best_model_cv": model_results[model_results["model_name"] == best_model].iloc[0].to_dict(),
        "predicted_curve_summary": predicted_curve_summary,
        "observed_curve_summary": observed_curve_summary,
        "disagreement_onset_event_study_summary": disagreement_onset_event_study_summary,
    }
    save_json(output_dir / "best_model_summary.json", summary_payload)

    logger.info("Best target metric: %s", best_metric)
    logger.info("Best model: %s", best_model)
    logger.info(
        "Best model CV | r2=%.4f | pearson=%.4f | rmse=%.4f",
        summary_payload["best_model_cv"]["r2_mean"],
        summary_payload["best_model_cv"]["pearson_r_mean"],
        summary_payload["best_model_cv"]["rmse_mean"],
    )


if __name__ == "__main__":
    main()
