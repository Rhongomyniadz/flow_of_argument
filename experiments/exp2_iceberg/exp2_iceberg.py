import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pandas.api.types import is_string_dtype
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = "data/stance_labeled/1024"
DEFAULT_CATEGORY_DATA_SUBDIR = "parsed"
DEFAULT_OUTPUT_DIR = "experiments/exp2_iceberg/results"
DEFAULT_MIN_TURNS = 12
WORD_PATTERN: re.Pattern[str] = re.compile(r"\b\w+\b")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def stringify_count_map(value_counts: pd.Series) -> Dict[str, int]:
    return {str(index): int(value) for index, value in value_counts.sort_index().items()}


def discover_category_names(data_path: Path, category_data_subdir: str) -> List[str]:
    categories: List[str] = []
    for child in sorted(data_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / category_data_subdir).is_dir():
            categories.append(child.name)
    return categories


def normalize_categories(categories: Optional[Sequence[str]]) -> List[str]:
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
    for candidate in [data_path.parent.parent, data_path.parent]:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if discover_category_names(candidate, category_data_subdir):
            return candidate
    return None


def collect_episode_files(
    data_path: Path,
    categories: Optional[Sequence[str]],
    category_data_subdir: str,
) -> List[Tuple[Optional[str], Path]]:
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
        logger.info(
            "Single category=%s | files=%d | path=%s",
            category_name,
            len(category_files),
            single_category_path,
        )
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
    if category:
        return f"{category}/{file_path.stem}"
    return file_path.stem


def infer_category_from_episode(raw_data: List[dict]) -> Optional[str]:
    if not raw_data:
        return None

    for turn in raw_data:
        category = str(turn.get("category") or "").strip()
        if category:
            return category
    return None


def count_words(text: str) -> int:
    return int(len(WORD_PATTERN.findall(text)))


def compute_duration_seconds(turn: Dict[str, Any]) -> Tuple[float, float, float]:
    raw_start = turn.get("start_time", turn.get("startTime", 0.0))
    start_time = float(raw_start if raw_start is not None else 0.0)

    raw_end = turn.get("end_time", turn.get("endTime"))
    raw_duration = turn.get("duration")
    if raw_end is not None:
        end_time = float(raw_end)
        duration_seconds = max(end_time - start_time, 0.1)
        return start_time, end_time, duration_seconds
    if raw_duration is not None:
        duration_seconds = max(float(raw_duration), 0.1)
        end_time = start_time + duration_seconds
        return start_time, end_time, duration_seconds

    end_time = start_time + 0.1
    return start_time, end_time, 0.1


def compute_turn_features(turn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if turn.get("turn_type_label") != "Substantive":
        return None

    raw_stance = turn.get("stance_pt")
    if raw_stance is None:
        return None

    stance_pt = float(raw_stance)
    start_time, end_time, duration_seconds = compute_duration_seconds(turn)
    midpoint_time = float((start_time + end_time) / 2.0)
    raw_turn_text: Any = turn.get("turn_text")
    if not isinstance(raw_turn_text, str):
        raise TypeError(
            "Missing or non-string turn_text for substantive stance-labeled turn: "
            f"speaker_id={turn.get('speaker_id')!r}, start_time={start_time!r}."
        )
    num_words: int = count_words(raw_turn_text)
    words_per_second: float = float(float(num_words) / max(duration_seconds, 0.1))
    log_words_per_second: float = float(np.log1p(words_per_second))

    explicit_count = int(len(turn.get("explicit_propositions", []) or []))
    implicit_count = int(len(turn.get("assumptions", []) or []))
    iceberg_density = float((float(explicit_count) / float(implicit_count + 1)) / max(duration_seconds, 0.1))
    log_iceberg_density = float(np.log1p(iceberg_density))

    return {
        "speaker_id": str(turn.get("speaker_id") or ""),
        "stance_pt": stance_pt,
        "stance_strength": float(stance_pt / 5.0),
        "explicit_count": explicit_count,
        "implicit_count": implicit_count,
        "duration_seconds": duration_seconds,
        "num_words": num_words,
        "words_per_second": words_per_second,
        "log_words_per_second": log_words_per_second,
        "start_time": start_time,
        "end_time": end_time,
        "midpoint_time": midpoint_time,
        "iceberg_density": iceberg_density,
        "log_iceberg_density": log_iceberg_density,
    }


def build_turn_level_features(
    data_dir: Path,
    categories: Optional[Sequence[str]],
    category_data_subdir: str,
    min_turns: int,
    require_two_speakers: bool,
    max_episodes: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    for category, file_path in tqdm(episode_files, desc="Building regression turn features"):
        files_seen += 1
        with open(file_path, "r", encoding="utf-8") as handle:
            raw_data = json.load(handle)

        file_category = category or infer_category_from_episode(raw_data) or ""
        if requested_category_keys and file_category.lower() not in requested_category_keys:
            continue

        turn_rows: List[Dict[str, Any]] = []
        for turn in raw_data:
            turn_features = compute_turn_features(turn)
            if turn_features is not None:
                turn_rows.append(turn_features)

        if not turn_rows:
            continue

        files_with_turns += 1
        if len(turn_rows) < min_turns:
            skipped_short += 1
            continue

        df_episode = pd.DataFrame(turn_rows).sort_values(["midpoint_time", "start_time", "end_time"]).reset_index(drop=True)
        if require_two_speakers and df_episode["speaker_id"].nunique() != 2:
            skipped_speaker_count += 1
            continue

        episode_id = build_episode_id(file_category or None, file_path)
        df_episode["episode"] = episode_id
        df_episode["category"] = file_category or "unknown"
        df_episode["turn_idx"] = np.arange(len(df_episode), dtype=int)

        episode_start = float(df_episode["start_time"].min())
        episode_end = float(df_episode["end_time"].max())
        episode_duration = episode_end - episode_start
        if episode_duration > 0.0:
            timeline_position = (df_episode["midpoint_time"].astype(float) - episode_start) / episode_duration
        elif len(df_episode) > 1:
            timeline_position = np.linspace(0.0, 1.0, len(df_episode))
        else:
            timeline_position = np.array([0.0], dtype=float)
        df_episode["timeline_position"] = np.clip(timeline_position, 0.0, 1.0)

        episode_frames.append(df_episode)
        files_kept += 1

    if not episode_frames:
        raise RuntimeError("No eligible episodes produced turn-level features.")

    df_all = pd.concat(episode_frames, ignore_index=True)
    build_summary = {
        "files_seen": files_seen,
        "episode_file_limit": int(max_episodes) if max_episodes is not None else None,
        "files_with_substantive_turns": files_with_turns,
        "episodes_kept": files_kept,
        "episodes_skipped_short": skipped_short,
        "episodes_skipped_speaker_count": skipped_speaker_count,
        "turn_rows": int(len(df_all)),
        "categories": stringify_count_map(df_all["category"].value_counts()),
        "stance_pt_distribution": stringify_count_map(df_all["stance_pt"].value_counts()),
    }
    return df_all, build_summary


def build_transition_frame(turn_df: pd.DataFrame) -> pd.DataFrame:
    transition_frames: List[pd.DataFrame] = []
    for _, group_df in turn_df.groupby("episode", sort=False):
        ordered_df = group_df.sort_values(["midpoint_time", "turn_idx"]).reset_index(drop=True).copy()
        if len(ordered_df) < 2:
            continue

        current_df = ordered_df.iloc[1:].copy().reset_index(drop=True)
        previous_df = ordered_df.iloc[:-1].copy().reset_index(drop=True)

        current_df["previous_turn_idx"] = previous_df["turn_idx"].astype(int)
        current_df["delta_stance_pt"] = current_df["stance_pt"].astype(float) - previous_df["stance_pt"].astype(float)
        current_df["delta_stance_strength"] = current_df["delta_stance_pt"].astype(float) / 5.0
        current_df["delta_log_iceberg_density"] = (
            current_df["log_iceberg_density"].astype(float) - previous_df["log_iceberg_density"].astype(float)
        )
        current_df["previous_log_iceberg_density"] = previous_df["log_iceberg_density"].astype(float)
        current_df["delta_log_words_per_second"] = (
            current_df["log_words_per_second"].astype(float) - previous_df["log_words_per_second"].astype(float)
        )
        current_df["shift_toward_agreement"] = np.maximum(current_df["delta_stance_strength"].astype(float), 0.0)
        current_df["shift_toward_disagreement"] = np.maximum(-current_df["delta_stance_strength"].astype(float), 0.0)
        current_df["lag1_delta_stance_strength"] = current_df["delta_stance_strength"].shift(1).fillna(0.0)
        current_df["lag1_shift_toward_agreement"] = current_df["shift_toward_agreement"].shift(1).fillna(0.0)
        current_df["lag1_shift_toward_disagreement"] = current_df["shift_toward_disagreement"].shift(1).fillna(0.0)
        transition_frames.append(current_df)

    if not transition_frames:
        raise RuntimeError("No transition rows could be constructed from the stance-labeled data.")

    transition_df = pd.concat(transition_frames, ignore_index=True)
    transition_df = transition_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "delta_stance_pt",
            "delta_stance_strength",
            "delta_log_iceberg_density",
            "previous_log_iceberg_density",
            "timeline_position",
            "words_per_second",
            "delta_log_words_per_second",
        ]
    )
    if transition_df.empty:
        raise RuntimeError("All transition rows were removed after finite-value filtering.")
    transition_df["delta_stance_pt"] = transition_df["delta_stance_pt"].astype(int)
    return transition_df.reset_index(drop=True)


def compute_mean_summary(values: np.ndarray) -> Dict[str, float]:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
    se_value = float(std_value / math.sqrt(len(finite_values)))
    return {
        "n": int(len(finite_values)),
        "mean": mean_value,
        "std": std_value,
        "se": se_value,
        "ci_low": float(mean_value - 1.96 * se_value),
        "ci_high": float(mean_value + 1.96 * se_value),
    }


def fit_clustered_ols(formula: str, transition_df: pd.DataFrame) -> Any:
    model_df = transition_df.copy()
    # Patsy does not reliably accept pandas nullable StringDtype columns.
    for column_name in ["category", "episode"]:
        if column_name in model_df.columns and is_string_dtype(model_df[column_name].dtype):
            model_df[column_name] = model_df[column_name].astype(object)

    ols_model = smf.ols(formula=formula, data=model_df)
    return ols_model.fit(cov_type="cluster", cov_kwds={"groups": model_df["episode"]})


def build_coefficient_frame(model_result: Any, model_name: str) -> pd.DataFrame:
    confidence_interval = model_result.conf_int()
    rows: List[Dict[str, Any]] = []
    for term_name in model_result.params.index:
        rows.append(
            {
                "model_name": model_name,
                "term": str(term_name),
                "coefficient": float(model_result.params[term_name]),
                "clustered_se": float(model_result.bse[term_name]),
                "t": float(model_result.tvalues[term_name]),
                "p": float(model_result.pvalues[term_name]),
                "ci_low": float(confidence_interval.loc[term_name, 0]),
                "ci_high": float(confidence_interval.loc[term_name, 1]),
            }
        )
    return pd.DataFrame(rows)


def build_model_comparison_frame(model_results: Dict[str, Any], transition_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_rows = int(len(transition_df))
    n_episodes = int(transition_df["episode"].nunique())
    for model_name, model_result in model_results.items():
        rows.append(
            {
                "model_name": model_name,
                "n_rows": n_rows,
                "n_episodes": n_episodes,
                "rsquared": float(model_result.rsquared),
                "adjusted_rsquared": float(model_result.rsquared_adj),
                "aic": float(model_result.aic),
                "bic": float(model_result.bic),
            }
        )
    return pd.DataFrame(rows)


def build_local_effect_bins_for_response(
    transition_df: pd.DataFrame,
    response_column: str,
    response_label: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = transition_df.groupby("delta_stance_pt", observed=False, sort=True)
    for delta_stance_pt, group_df in grouped:
        stats = compute_mean_summary(group_df[response_column].astype(float).to_numpy())
        rows.append(
            {
                "delta_stance_pt": int(delta_stance_pt),
                "n_transition_rows": stats["n"],
                f"mean_{response_label}": stats["mean"],
                f"std_{response_label}": stats["std"],
                f"se_{response_label}": stats["se"],
                f"ci_low_{response_label}": stats["ci_low"],
                f"ci_high_{response_label}": stats["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def build_local_effect_bins(transition_df: pd.DataFrame) -> pd.DataFrame:
    return build_local_effect_bins_for_response(
        transition_df,
        "delta_log_iceberg_density",
        "delta_log_iceberg_density",
    )


def extract_term_row(coefficient_df: pd.DataFrame, model_name: str, term_name: str) -> Dict[str, Any]:
    matching_rows = coefficient_df[
        (coefficient_df["model_name"] == model_name) & (coefficient_df["term"] == term_name)
    ]
    if matching_rows.empty:
        raise KeyError(f"Missing coefficient term: model={model_name}, term={term_name}")
    return matching_rows.iloc[0].to_dict()


def build_verdict(coefficient_df: pd.DataFrame) -> Dict[str, Any]:
    primary_row = extract_term_row(coefficient_df, "primary_signed_change", "delta_stance_strength")
    directional_agreement_row = extract_term_row(
        coefficient_df, "secondary_directional_change", "shift_toward_agreement"
    )
    directional_disagreement_row = extract_term_row(
        coefficient_df, "secondary_directional_change", "shift_toward_disagreement"
    )

    primary_coefficient = float(primary_row["coefficient"])
    primary_ci_low = float(primary_row["ci_low"])
    primary_ci_high = float(primary_row["ci_high"])
    local_relationship_detected = bool(primary_ci_low > 0.0 or primary_ci_high < 0.0)
    agreement_deepening_supported = bool(primary_ci_high < 0.0)
    disagreement_surfacing_supported = bool(primary_ci_high < 0.0)
    directional_agreement_supported = bool(float(directional_agreement_row["ci_high"]) < 0.0)
    directional_disagreement_supported = bool(float(directional_disagreement_row["ci_low"]) > 0.0)

    if local_relationship_detected and agreement_deepening_supported:
        interpretation = (
            "Local movement toward agreement predicts lower subsequent iceberg density, which is consistent with "
            "context deepening; the same signed effect implies that movement toward disagreement predicts local "
            "iceberg-density increases."
        )
    elif local_relationship_detected:
        interpretation = (
            "The regression detects a local stance-density relationship, but the sign does not match the expected "
            "agreement-deepening and disagreement-surfacing pattern."
        )
    else:
        interpretation = (
            "The regression does not detect a clear local relationship between stance change and iceberg-density change."
        )

    return {
        "primary_relationship_detected": local_relationship_detected,
        "agreement_deepening_supported": agreement_deepening_supported,
        "disagreement_surfacing_supported": disagreement_surfacing_supported,
        "directional_agreement_supported": directional_agreement_supported,
        "directional_disagreement_supported": directional_disagreement_supported,
        "primary_delta_stance_strength_coefficient": primary_coefficient,
        "primary_delta_stance_strength_ci_low": primary_ci_low,
        "primary_delta_stance_strength_ci_high": primary_ci_high,
        "interpretation": interpretation,
    }


def build_dataset_summary(
    turn_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    build_summary: Dict[str, Any],
    requested_categories: Optional[Sequence[str]],
    max_episodes: Optional[int],
    min_turns: int,
    require_two_speakers: bool,
) -> Dict[str, Any]:
    dataset_summary = {
        "requested_categories": list(normalize_categories(requested_categories)),
        "max_episodes": int(max_episodes) if max_episodes is not None else None,
        "min_turns": int(min_turns),
        "require_two_speakers": bool(require_two_speakers),
        "turn_rows": int(len(turn_df)),
        "transition_rows": int(len(transition_df)),
        "episodes_in_turn_data": int(turn_df["episode"].nunique()),
        "episodes_in_transition_data": int(transition_df["episode"].nunique()),
        "categories": stringify_count_map(turn_df["category"].value_counts()),
        "stance_pt_distribution": stringify_count_map(turn_df["stance_pt"].value_counts()),
        "delta_stance_pt_distribution": stringify_count_map(transition_df["delta_stance_pt"].value_counts()),
    }
    dataset_summary.update(build_summary)
    return dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 2 as a local regression analysis of stance change and iceberg-density change."
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--category_data_subdir", type=str, default=DEFAULT_CATEGORY_DATA_SUBDIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_turns", type=int, default=DEFAULT_MIN_TURNS)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--allow_non_dyadic", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    turn_df, build_summary = build_turn_level_features(
        data_dir=Path(args.data_dir),
        categories=args.categories,
        category_data_subdir=args.category_data_subdir,
        min_turns=args.min_turns,
        require_two_speakers=not args.allow_non_dyadic,
        max_episodes=args.max_episodes,
    )
    transition_df = build_transition_frame(turn_df)

    primary_formula = (
        "delta_log_iceberg_density ~ delta_stance_strength + lag1_delta_stance_strength + "
        "previous_log_iceberg_density + timeline_position + I(timeline_position ** 2) + C(category)"
    )
    secondary_formula = (
        "delta_log_iceberg_density ~ shift_toward_agreement + shift_toward_disagreement + "
        "lag1_shift_toward_agreement + lag1_shift_toward_disagreement + previous_log_iceberg_density + "
        "timeline_position + I(timeline_position ** 2) + C(category)"
    )
    lexical_rate_baseline_formula = (
        "delta_log_iceberg_density ~ words_per_second + previous_log_iceberg_density + "
        "timeline_position + I(timeline_position ** 2) + C(category)"
    )

    primary_model = fit_clustered_ols(primary_formula, transition_df)
    secondary_model = fit_clustered_ols(secondary_formula, transition_df)
    lexical_rate_baseline_model = fit_clustered_ols(lexical_rate_baseline_formula, transition_df)

    coefficient_df = pd.concat(
        [
            build_coefficient_frame(primary_model, "primary_signed_change"),
            build_coefficient_frame(secondary_model, "secondary_directional_change"),
            build_coefficient_frame(lexical_rate_baseline_model, "lexical_rate_baseline"),
        ],
        ignore_index=True,
    )
    model_comparison_df = build_model_comparison_frame(
        {
            "primary_signed_change": primary_model,
            "secondary_directional_change": secondary_model,
            "lexical_rate_baseline": lexical_rate_baseline_model,
        },
        transition_df,
    )
    local_effect_bins_df = build_local_effect_bins(transition_df)
    heuristic_effect_bins_df = build_local_effect_bins_for_response(
        transition_df,
        "delta_log_words_per_second",
        "delta_log_words_per_second",
    )

    coefficient_df.to_csv(output_dir / "exp2_regression_coefficients.csv", index=False)
    model_comparison_df.to_csv(output_dir / "exp2_regression_model_comparison.csv", index=False)
    local_effect_bins_df.to_csv(output_dir / "exp2_local_effect_bins.csv", index=False)
    heuristic_effect_bins_df.to_csv(output_dir / "exp2_heuristic_local_effect_bins.csv", index=False)

    dataset_summary = build_dataset_summary(
        turn_df=turn_df,
        transition_df=transition_df,
        build_summary=build_summary,
        requested_categories=args.categories,
        max_episodes=args.max_episodes,
        min_turns=args.min_turns,
        require_two_speakers=not args.allow_non_dyadic,
    )
    exp2_summary = {
        "analysis_design": {
            "framing": "local_stance_change_regression",
            "stance_field": "stance_pt",
            "stance_mapping": "signed scale from -5 disagreement to +5 agreement, with 0 treated as neutral or no stance",
            "iceberg_density_definition": "(explicit_count / (implicit_count + 1)) / duration_seconds",
            "lexical_rate_baseline": "words_per_second = num_words / duration_seconds, fit as a separate baseline model",
            "response_variable": "delta_log_iceberg_density",
            "heuristic_response_variable": "delta_log_words_per_second",
            "authoritative_outputs": [
                "exp2_regression_coefficients.csv",
                "exp2_regression_model_comparison.csv",
                "exp2_local_effect_bins.csv",
                "exp2_heuristic_local_effect_bins.csv",
                "exp2_local_relationship.pdf",
                "exp2_local_relationship.png",
                "exp2_heuristic_relationship.pdf",
                "exp2_heuristic_relationship.png",
                "exp2_summary.json",
                "dataset_summary.json",
            ],
        },
        "formulas": {
            "primary_signed_change": primary_formula,
            "secondary_directional_change": secondary_formula,
            "lexical_rate_baseline": lexical_rate_baseline_formula,
        },
        "dataset_summary": dataset_summary,
        "model_comparison": model_comparison_df.to_dict(orient="records"),
        "headline_coefficients": {
            "primary_delta_stance_strength": extract_term_row(
                coefficient_df, "primary_signed_change", "delta_stance_strength"
            ),
            "secondary_shift_toward_agreement": extract_term_row(
                coefficient_df, "secondary_directional_change", "shift_toward_agreement"
            ),
            "secondary_shift_toward_disagreement": extract_term_row(
                coefficient_df, "secondary_directional_change", "shift_toward_disagreement"
            ),
            "lexical_rate_baseline_words_per_second": extract_term_row(
                coefficient_df, "lexical_rate_baseline", "words_per_second"
            ),
        },
        "verdict": build_verdict(coefficient_df),
    }

    save_json(output_dir / "dataset_summary.json", dataset_summary)
    save_json(output_dir / "exp2_summary.json", exp2_summary)

    logger.info(
        "Experiment 2 regression completed | transition_rows=%d | episodes=%d | primary_coef=%.6f",
        int(len(transition_df)),
        int(transition_df["episode"].nunique()),
        float(extract_term_row(coefficient_df, "primary_signed_change", "delta_stance_strength")["coefficient"]),
    )


if __name__ == "__main__":
    main()
