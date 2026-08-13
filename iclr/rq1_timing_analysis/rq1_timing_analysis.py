from __future__ import annotations

"""Timing-aware, duration-free robustness analysis for RQ1."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
DEFAULT_DATA_DIR = Path("data/stance_labeled/1024")
DEFAULT_OUTPUT_DIR = Path("iclr/rq1_timing_analysis/results")
DEFAULT_CATEGORY_DATA_SUBDIR = "parsed"
DEFAULT_PLOT_DPI = 300
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

PER_SECOND_MODEL = "per_second"
TOKEN_MODEL = "token_normalized"
TIMING_MODEL = "token_timing_adjusted"
PREVIOUS_DURATION_MODEL = "token_timing_previous_duration"
MODEL_ORDER = (
    PER_SECOND_MODEL,
    TOKEN_MODEL,
    TIMING_MODEL,
    PREVIOUS_DURATION_MODEL,
)
HEADLINE_MODELS = (PER_SECOND_MODEL, TOKEN_MODEL, TIMING_MODEL)
STANCE_TERMS = ("agree_move", "disagree_move")
MANUSCRIPT_REFERENCE_COEFFICIENTS = {
    "agree_move": -0.0056,
    "disagree_move": 0.0049,
}

FORMULAS = {
    PER_SECOND_MODEL: (
        "delta_log_density_per_second ~ agree_move + disagree_move + "
        "lag_delta_stance + previous_log_density_per_second + timeline_position + "
        "I(timeline_position ** 2) + C(category)"
    ),
    TOKEN_MODEL: (
        "delta_log_density_per_token ~ agree_move + disagree_move + "
        "lag_delta_stance + previous_log_density_per_token + timeline_position + "
        "I(timeline_position ** 2) + C(category)"
    ),
    TIMING_MODEL: (
        "delta_log_density_per_token ~ agree_move + disagree_move + "
        "lag_delta_stance + previous_log_density_per_token + timeline_position + "
        "I(timeline_position ** 2) + C(category) + log_duration + log_gap + overlap"
    ),
    PREVIOUS_DURATION_MODEL: (
        "delta_log_density_per_token ~ agree_move + disagree_move + "
        "lag_delta_stance + previous_log_density_per_token + timeline_position + "
        "I(timeline_position ** 2) + C(category) + log_duration + log_gap + overlap + "
        "previous_log_duration"
    ),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TimingRecord(TypedDict):
    valid: bool
    source: str | None
    error: str | None
    start_time: float | None
    end_time: float | None
    duration: float | None


class TurnRecord(TypedDict):
    raw_index: int
    turn_idx: int
    speaker_id: str | None
    turn_type_label: str | None
    turn_text: str | None
    word_count: int
    stance: float | None
    explicit_count: int | None
    assumption_count: int | None
    timing: TimingRecord


class Observation(TypedDict):
    observation_id: str
    category: str
    episode: str
    source_path: str
    lag_turn_raw_index: int
    previous_turn_raw_index: int
    current_turn_raw_index: int
    previous_turn_idx: int
    current_turn_idx: int
    previous_speaker_id: str
    current_speaker_id: str
    previous_stance: float
    current_stance: float
    delta_stance: float
    lag_delta_stance: float
    agree_move: float
    disagree_move: float
    previous_explicit_count: int
    previous_assumption_count: int
    previous_word_count: int
    explicit_count: int
    assumption_count: int
    word_count: int
    previous_duration: float
    duration: float
    previous_log_duration: float
    log_duration: float
    previous_timestamp_source: str
    current_timestamp_source: str
    raw_gap: float
    pre_turn_gap: float
    log_gap: float
    overlap: int
    timeline_position: float
    previous_density_per_second: float
    density_per_second: float
    previous_log_density_per_second: float
    log_density_per_second: float
    delta_log_density_per_second: float
    previous_density_per_token: float
    density_per_token: float
    previous_log_density_per_token: float
    log_density_per_token: float
    delta_log_density_per_token: float
    words_per_second: float


class RegressionResult(Protocol):
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    rsquared: float
    rsquared_adj: float
    aic: float
    bic: float
    nobs: float

    def conf_int(self, alpha: float) -> pd.DataFrame:
        ...


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(cast(float | int | str, value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def first_finite(turn: dict[str, object], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in turn:
            parsed = finite_float(turn[name])
            if parsed is not None:
                return parsed
    return None


def parse_timing(turn: dict[str, object]) -> TimingRecord:
    start = first_finite(turn, ("start_time", "startTime"))
    end = first_finite(turn, ("end_time", "endTime"))
    duration = finite_float(turn.get("duration"))
    positive_duration = duration if duration is not None and duration > 0.0 else None

    if start is not None and end is not None:
        resolved_duration = end - start
        if resolved_duration <= 0.0:
            return {
                "valid": False,
                "source": None,
                "error": "nonpositive_endpoint_duration",
                "start_time": start,
                "end_time": end,
                "duration": None,
            }
        return {
            "valid": True,
            "source": "start_end",
            "error": None,
            "start_time": start,
            "end_time": end,
            "duration": resolved_duration,
        }
    if start is not None and positive_duration is not None:
        return {
            "valid": True,
            "source": "start_duration",
            "error": None,
            "start_time": start,
            "end_time": start + positive_duration,
            "duration": positive_duration,
        }
    if end is not None and positive_duration is not None:
        return {
            "valid": True,
            "source": "end_duration",
            "error": None,
            "start_time": end - positive_duration,
            "end_time": end,
            "duration": positive_duration,
        }
    error = "nonpositive_duration" if duration is not None and duration <= 0.0 else "missing_timing_endpoint"
    return {
        "valid": False,
        "source": None,
        "error": error,
        "start_time": start,
        "end_time": end,
        "duration": None,
    }


def list_count(value: object) -> int | None:
    if value is None:
        return 0
    if not isinstance(value, list):
        return None
    return len(value)


def parse_turn(turn: dict[str, object], raw_index: int) -> TurnRecord:
    raw_text = turn.get("turn_text")
    text = raw_text.strip() if isinstance(raw_text, str) and raw_text.strip() else None
    raw_speaker = turn.get("speaker_id", turn.get("speaker"))
    speaker = str(raw_speaker).strip() if raw_speaker is not None and str(raw_speaker).strip() else None
    raw_turn_type = turn.get("turn_type_label")
    turn_type = str(raw_turn_type).strip() if raw_turn_type is not None else None
    stance = finite_float(turn.get("stance_pt"))
    if stance is not None and not -5.0 <= stance <= 5.0:
        stance = None
    raw_turn_idx = finite_float(turn.get("turn_idx"))
    turn_idx = int(raw_turn_idx) if raw_turn_idx is not None and raw_turn_idx.is_integer() else raw_index
    return {
        "raw_index": raw_index,
        "turn_idx": turn_idx,
        "speaker_id": speaker,
        "turn_type_label": turn_type,
        "turn_text": text,
        "word_count": len(WORD_PATTERN.findall(text)) if text is not None else 0,
        "stance": stance,
        "explicit_count": list_count(turn.get("explicit_propositions")),
        "assumption_count": list_count(turn.get("assumptions")),
        "timing": parse_timing(turn),
    }


def load_episode(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_turns = payload
    elif isinstance(payload, dict) and isinstance(payload.get("turns"), list):
        raw_turns = payload["turns"]
    else:
        raise ValueError(f"{path}: expected a JSON list or an object with a 'turns' list")
    turns: list[dict[str, object]] = []
    for index, value in enumerate(raw_turns):
        if not isinstance(value, dict):
            raise TypeError(f"{path}: turn {index} is {type(value).__name__}, expected object")
        turns.append(cast(dict[str, object], value))
    return turns


def infer_episode_id(turns: list[dict[str, object]], path: Path) -> str:
    for turn in turns:
        value = turn.get("episode_id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return path.stem


def infer_category(turns: list[dict[str, object]], category_hint: str | None) -> str:
    if category_hint is not None and category_hint.strip():
        return category_hint.strip()
    for turn in turns:
        value = turn.get("category")
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def episode_timeline_bounds(turns: list[TurnRecord]) -> tuple[float, float] | None:
    timings = [turn["timing"] for turn in turns if turn["timing"]["valid"]]
    starts = [cast(float, timing["start_time"]) for timing in timings]
    ends = [cast(float, timing["end_time"]) for timing in timings]
    if not starts or not ends:
        return None
    episode_start = min(starts)
    episode_end = max(ends)
    if episode_end <= episode_start:
        return None
    return episode_start, episode_end


def window_exclusion_reason(window: tuple[TurnRecord, TurnRecord, TurnRecord]) -> str | None:
    lag_turn, previous, current = window
    if any(turn["turn_type_label"] != "Substantive" for turn in window):
        return "non_substantive_window"
    if any(turn["turn_text"] is None or turn["word_count"] <= 0 for turn in window):
        return "empty_or_zero_word_turn"
    if any(turn["stance"] is None for turn in window):
        return "missing_or_invalid_stance"
    if any(turn["speaker_id"] is None for turn in window):
        return "missing_speaker"
    if lag_turn["speaker_id"] == previous["speaker_id"] or previous["speaker_id"] == current["speaker_id"]:
        return "same_speaker_boundary"
    if previous["explicit_count"] is None or previous["assumption_count"] is None:
        return "invalid_previous_representation_fields"
    if current["explicit_count"] is None or current["assumption_count"] is None:
        return "invalid_current_representation_fields"
    if not previous["timing"]["valid"]:
        return f"invalid_previous_timing:{previous['timing']['error']}"
    if not current["timing"]["valid"]:
        return f"invalid_current_timing:{current['timing']['error']}"
    return None


def density(explicit_count: int, assumption_count: int, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError(f"Density denominator must be positive, received {denominator}")
    return (float(explicit_count) / float(assumption_count + 1)) / denominator


def build_observation(
    window: tuple[TurnRecord, TurnRecord, TurnRecord],
    category: str,
    episode: str,
    source_path: Path,
    timeline_bounds: tuple[float, float],
) -> Observation:
    lag_turn, previous, current = window
    previous_timing = previous["timing"]
    current_timing = current["timing"]
    previous_start = cast(float, previous_timing["start_time"])
    previous_end = cast(float, previous_timing["end_time"])
    previous_duration = cast(float, previous_timing["duration"])
    current_start = cast(float, current_timing["start_time"])
    current_end = cast(float, current_timing["end_time"])
    current_duration = cast(float, current_timing["duration"])
    previous_explicit = cast(int, previous["explicit_count"])
    previous_assumptions = cast(int, previous["assumption_count"])
    current_explicit = cast(int, current["explicit_count"])
    current_assumptions = cast(int, current["assumption_count"])
    previous_stance = cast(float, previous["stance"])
    current_stance = cast(float, current["stance"])
    lag_stance = cast(float, lag_turn["stance"])

    previous_per_second = density(previous_explicit, previous_assumptions, previous_duration)
    current_per_second = density(current_explicit, current_assumptions, current_duration)
    previous_per_token = density(previous_explicit, previous_assumptions, float(previous["word_count"]))
    current_per_token = density(current_explicit, current_assumptions, float(current["word_count"]))
    previous_log_per_second = math.log1p(previous_per_second)
    current_log_per_second = math.log1p(current_per_second)
    previous_log_per_token = math.log1p(previous_per_token)
    current_log_per_token = math.log1p(current_per_token)
    delta_stance = (current_stance - previous_stance) / 5.0
    lag_delta_stance = (previous_stance - lag_stance) / 5.0
    raw_gap = current_start - previous_end
    pre_turn_gap = max(raw_gap, 0.0)
    episode_start, episode_end = timeline_bounds
    midpoint = (current_start + current_end) / 2.0
    timeline_position = (midpoint - episode_start) / (episode_end - episode_start)

    return {
        "observation_id": (
            f"{category}:{episode}:{lag_turn['raw_index']}:"
            f"{previous['raw_index']}:{current['raw_index']}"
        ),
        "category": category,
        "episode": f"{category}/{episode}",
        "source_path": str(source_path),
        "lag_turn_raw_index": lag_turn["raw_index"],
        "previous_turn_raw_index": previous["raw_index"],
        "current_turn_raw_index": current["raw_index"],
        "previous_turn_idx": previous["turn_idx"],
        "current_turn_idx": current["turn_idx"],
        "previous_speaker_id": cast(str, previous["speaker_id"]),
        "current_speaker_id": cast(str, current["speaker_id"]),
        "previous_stance": previous_stance,
        "current_stance": current_stance,
        "delta_stance": delta_stance,
        "lag_delta_stance": lag_delta_stance,
        "agree_move": max(delta_stance, 0.0),
        "disagree_move": max(-delta_stance, 0.0),
        "previous_explicit_count": previous_explicit,
        "previous_assumption_count": previous_assumptions,
        "previous_word_count": previous["word_count"],
        "explicit_count": current_explicit,
        "assumption_count": current_assumptions,
        "word_count": current["word_count"],
        "previous_duration": previous_duration,
        "duration": current_duration,
        "previous_log_duration": math.log1p(previous_duration),
        "log_duration": math.log1p(current_duration),
        "previous_timestamp_source": cast(str, previous_timing["source"]),
        "current_timestamp_source": cast(str, current_timing["source"]),
        "raw_gap": raw_gap,
        "pre_turn_gap": pre_turn_gap,
        "log_gap": math.log1p(pre_turn_gap),
        "overlap": int(raw_gap < 0.0),
        "timeline_position": timeline_position,
        "previous_density_per_second": previous_per_second,
        "density_per_second": current_per_second,
        "previous_log_density_per_second": previous_log_per_second,
        "log_density_per_second": current_log_per_second,
        "delta_log_density_per_second": current_log_per_second - previous_log_per_second,
        "previous_density_per_token": previous_per_token,
        "density_per_token": current_per_token,
        "previous_log_density_per_token": previous_log_per_token,
        "log_density_per_token": current_log_per_token,
        "delta_log_density_per_token": current_log_per_token - previous_log_per_token,
        "words_per_second": float(current["word_count"]) / current_duration,
    }


def build_episode_observations(
    raw_turns: list[dict[str, object]],
    category: str,
    episode: str,
    source_path: Path,
) -> tuple[list[Observation], dict[str, int], dict[str, int]]:
    turns = [parse_turn(turn, index) for index, turn in enumerate(raw_turns)]
    timing_sources = Counter(
        cast(str, turn["timing"]["source"])
        for turn in turns
        if turn["timing"]["valid"]
    )
    bounds = episode_timeline_bounds(turns)
    exclusions: Counter[str] = Counter()
    observations: list[Observation] = []
    for current_index in range(2, len(turns)):
        window = (turns[current_index - 2], turns[current_index - 1], turns[current_index])
        reason = window_exclusion_reason(window)
        if reason is None and bounds is None:
            reason = "invalid_episode_timeline"
        if reason is not None:
            exclusions[reason] += 1
            continue
        observations.append(
            build_observation(window, category, episode, source_path, cast(tuple[float, float], bounds))
        )
    return observations, dict(sorted(exclusions.items())), dict(sorted(timing_sources.items()))


def discover_episode_files(
    data_dir: Path,
    category_data_subdir: str,
) -> list[tuple[str | None, Path]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")
    direct_files = sorted(data_dir.glob("*.json"))
    if direct_files:
        return [(None, path) for path in direct_files]
    single_category_dir = data_dir / category_data_subdir
    if single_category_dir.is_dir():
        return [(data_dir.name, path) for path in sorted(single_category_dir.glob("*.json"))]
    discovered: list[tuple[str | None, Path]] = []
    for category_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        parsed_dir = category_dir / category_data_subdir
        if parsed_dir.is_dir():
            discovered.extend((category_dir.name, path) for path in sorted(parsed_dir.glob("*.json")))
    if not discovered:
        raise FileNotFoundError(
            f"No JSON episodes found directly under {data_dir}, under {category_data_subdir}, "
            "or under category/parsed directories"
        )
    return discovered


def selected_category(category: str, categories: list[str]) -> bool:
    if not categories or any(value.casefold() == "all" for value in categories):
        return True
    requested = {value.casefold() for value in categories}
    return category.casefold() in requested


def observation_digest(rows: list[Observation]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def build_dataset(
    data_dir: Path,
    categories: list[str],
    category_data_subdir: str,
    max_episodes: int | None,
    show_progress: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    files = discover_episode_files(data_dir, category_data_subdir)
    iterator: object = files
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError as error:
            raise RuntimeError(
                "Progress display requires tqdm. Install the dependencies declared in pyproject.toml "
                "or pass --no_tqdm."
            ) from error
        iterator = tqdm(files, desc="Building strict RQ1 transitions")

    observations: list[Observation] = []
    exclusion_counts: Counter[str] = Counter()
    timing_source_counts: Counter[str] = Counter()
    category_episode_counts: Counter[str] = Counter()
    category_observation_counts: Counter[str] = Counter()
    selected_hashes: list[dict[str, str]] = []
    selected_episode_count = 0
    files_examined = 0
    for category_hint, path in cast(list[tuple[str | None, Path]], iterator):
        files_examined += 1
        raw_turns = load_episode(path)
        category = infer_category(raw_turns, category_hint)
        if not selected_category(category, categories):
            continue
        if max_episodes is not None and selected_episode_count >= max_episodes:
            break
        episode = infer_episode_id(raw_turns, path)
        episode_rows, episode_exclusions, episode_timing_sources = build_episode_observations(
            raw_turns,
            category,
            episode,
            path,
        )
        observations.extend(episode_rows)
        exclusion_counts.update(episode_exclusions)
        timing_source_counts.update(episode_timing_sources)
        category_episode_counts[category] += 1
        category_observation_counts[category] += len(episode_rows)
        selected_hashes.append({"path": str(path), "sha256": file_sha256(path)})
        selected_episode_count += 1

    if selected_episode_count == 0:
        raise RuntimeError(f"No episodes matched categories={categories or ['all']} under {data_dir}")
    if not observations:
        raise RuntimeError("No strict, timestamp-complete three-turn observations were retained")
    frame = pd.DataFrame(observations)
    frame = frame.sort_values(
        ["category", "episode", "current_turn_raw_index"],
        kind="stable",
    ).reset_index(drop=True)
    if frame["observation_id"].duplicated().any():
        duplicates = frame.loc[frame["observation_id"].duplicated(), "observation_id"].tolist()
        raise ValueError(f"Duplicate observation IDs: {duplicates[:5]}")

    gap_values = frame["raw_gap"].astype(float)
    duration_values = frame["duration"].astype(float)
    audit: dict[str, object] = {
        "data_dir": str(data_dir),
        "requested_categories": categories or ["all"],
        "category_data_subdir": category_data_subdir,
        "max_episodes": max_episodes,
        "files_discovered": len(files),
        "files_examined": files_examined,
        "selected_episode_count": selected_episode_count,
        "retained_observation_count": len(frame),
        "episode_count_with_observations": int(frame["episode"].nunique()),
        "candidate_window_count": len(frame) + sum(exclusion_counts.values()),
        "excluded_window_count": sum(exclusion_counts.values()),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "category_episode_counts": dict(sorted(category_episode_counts.items())),
        "category_observation_counts": dict(sorted(category_observation_counts.items())),
        "timestamp_source_counts": dict(sorted(timing_source_counts.items())),
        "overlap_count": int(frame["overlap"].sum()),
        "overlap_rate": float(frame["overlap"].mean()),
        "raw_gap_seconds": numeric_summary(gap_values),
        "current_duration_seconds": numeric_summary(duration_values),
        "current_words_per_second": numeric_summary(frame["words_per_second"].astype(float)),
        "selected_input_manifest_sha256": canonical_sha256(selected_hashes),
        "selected_input_manifest_hash_method": "sha256(canonical JSON of sorted path and file-sha256 records)",
        "observation_rows_sha256": observation_digest(observations),
        "observation_hash_method": "sha256(canonical JSON per row in construction order, newline-delimited)",
    }
    return frame, audit


def numeric_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Numeric audit summaries require a nonempty finite series")
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def require_analysis_dependencies() -> object:
    try:
        import statsmodels.formula.api as formula_api
    except ImportError as error:
        raise RuntimeError(
            "RQ1 regression requires statsmodels. Install the dependencies declared in pyproject.toml."
        ) from error
    return formula_api


def validate_model_frame(frame: pd.DataFrame) -> None:
    required = {
        "category",
        "episode",
        "delta_log_density_per_second",
        "delta_log_density_per_token",
        "agree_move",
        "disagree_move",
        "lag_delta_stance",
        "previous_log_density_per_second",
        "previous_log_density_per_token",
        "timeline_position",
        "log_duration",
        "log_gap",
        "overlap",
        "previous_log_duration",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Observation frame is missing model columns: {missing}")
    numeric_columns = sorted(required.difference({"category", "episode"}))
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Model frame contains missing numeric values")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Model frame contains nonfinite numeric values")
    if frame["episode"].nunique() < 2:
        raise ValueError("Episode-clustered regression requires at least two episodes")


def fit_models(frame: pd.DataFrame) -> dict[str, RegressionResult]:
    validate_model_frame(frame)
    formula_api = require_analysis_dependencies()
    model_frame = frame.copy()
    model_frame["category"] = model_frame["category"].astype(object)
    model_frame["episode"] = model_frame["episode"].astype(object)
    results: dict[str, RegressionResult] = {}
    for model_name in MODEL_ORDER:
        formula = FORMULAS[model_name]
        model = formula_api.ols(formula=formula, data=model_frame)
        result = cast(
            RegressionResult,
            model.fit(cov_type="cluster", cov_kwds={"groups": model_frame["episode"]}),
        )
        if int(result.nobs) != len(model_frame):
            raise RuntimeError(
                f"Model {model_name} used {int(result.nobs)} rows; expected the common sample of {len(model_frame)}"
            )
        results[model_name] = result
    return results


def coefficient_frame(results: dict[str, RegressionResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        result = results[model_name]
        intervals = result.conf_int(alpha=0.05)
        for term in result.params.index:
            rows.append(
                {
                    "model_name": model_name,
                    "term": str(term),
                    "coefficient": float(result.params[term]),
                    "clustered_se": float(result.bse[term]),
                    "z_or_t": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci95_low": float(intervals.loc[term, 0]),
                    "ci95_high": float(intervals.loc[term, 1]),
                }
            )
    return pd.DataFrame(rows)


def model_fit_frame(
    results: dict[str, RegressionResult],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        result = results[model_name]
        comparison_group = "per_second_outcome" if model_name == PER_SECOND_MODEL else "token_outcome"
        rows.append(
            {
                "model_name": model_name,
                "response_variable": (
                    "delta_log_density_per_second"
                    if model_name == PER_SECOND_MODEL
                    else "delta_log_density_per_token"
                ),
                "transition_count": len(frame),
                "episode_count": int(frame["episode"].nunique()),
                "r_squared": float(result.rsquared),
                "adjusted_r_squared": float(result.rsquared_adj),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "aic_bic_comparison_group": comparison_group,
                "aic_bic_comparable_to_other_models": comparison_group == "token_outcome",
            }
        )
    fit_frame = pd.DataFrame(rows)
    if fit_frame["transition_count"].nunique() != 1 or fit_frame["episode_count"].nunique() != 1:
        raise RuntimeError("Regression specifications did not use identical samples")
    return fit_frame


def extract_coefficient(
    coefficients: pd.DataFrame,
    model_name: str,
    term: str,
) -> dict[str, object]:
    selected = coefficients[
        (coefficients["model_name"] == model_name) & (coefficients["term"] == term)
    ]
    if len(selected) != 1:
        raise KeyError(f"Expected one coefficient for model={model_name}, term={term}; found {len(selected)}")
    return cast(dict[str, object], selected.iloc[0].to_dict())


def attenuation_percent(token_coefficient: float, timing_coefficient: float) -> float:
    if token_coefficient == 0.0:
        raise ZeroDivisionError("Cannot calculate attenuation from a zero token-normalized coefficient")
    return 100.0 * (1.0 - abs(timing_coefficient) / abs(token_coefficient))


def stance_comparison_frame(
    coefficients: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in STANCE_TERMS:
        token = extract_coefficient(coefficients, TOKEN_MODEL, term)
        timing = extract_coefficient(coefficients, TIMING_MODEL, term)
        attenuation = attenuation_percent(
            float(cast(float, token["coefficient"])),
            float(cast(float, timing["coefficient"])),
        )
        for model_name in HEADLINE_MODELS:
            coefficient = extract_coefficient(coefficients, model_name, term)
            rows.append(
                {
                    "stance_direction": "agreement" if term == "agree_move" else "disagreement",
                    "term": term,
                    "model_name": model_name,
                    "coefficient": coefficient["coefficient"],
                    "clustered_se": coefficient["clustered_se"],
                    "ci95_low": coefficient["ci95_low"],
                    "ci95_high": coefficient["ci95_high"],
                    "p_value": coefficient["p_value"],
                    "transition_count": len(frame),
                    "episode_count": int(frame["episode"].nunique()),
                    "timing_attenuation_percent": attenuation if model_name == TIMING_MODEL else None,
                    "attenuation_reference_model": TOKEN_MODEL if model_name == TIMING_MODEL else None,
                }
            )
    return pd.DataFrame(rows)


def coefficient_interpretation(comparison: pd.DataFrame, term: str) -> dict[str, object]:
    token = comparison[(comparison["term"] == term) & (comparison["model_name"] == TOKEN_MODEL)].iloc[0]
    timing = comparison[(comparison["term"] == term) & (comparison["model_name"] == TIMING_MODEL)].iloc[0]
    expected_negative = term == "agree_move"
    token_sign = int(np.sign(float(token["coefficient"])))
    timing_sign = int(np.sign(float(timing["coefficient"])))
    interval_expected = (
        float(timing["ci95_high"]) < 0.0
        if expected_negative
        else float(timing["ci95_low"]) > 0.0
    )
    if token_sign != timing_sign:
        status = "sign_reversed_after_timing_adjustment"
    elif interval_expected:
        status = "direction_preserved_and_interval_excludes_zero"
    else:
        status = "direction_preserved_but_interval_includes_zero_or_wrong_direction"
    return {
        "term": term,
        "direction": "agreement" if expected_negative else "disagreement",
        "token_normalized_coefficient": float(token["coefficient"]),
        "timing_adjusted_coefficient": float(timing["coefficient"]),
        "timing_adjusted_ci95_low": float(timing["ci95_low"]),
        "timing_adjusted_ci95_high": float(timing["ci95_high"]),
        "attenuation_percent": float(timing["timing_attenuation_percent"]),
        "status": status,
    }


def manuscript_reference_comparison(coefficients: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in STANCE_TERMS:
        strict = extract_coefficient(coefficients, TOKEN_MODEL, term)
        strict_value = float(cast(float, strict["coefficient"]))
        reference = MANUSCRIPT_REFERENCE_COEFFICIENTS[term]
        rows.append(
            {
                "term": term,
                "manuscript_approximate_coefficient": reference,
                "strict_adjacency_coefficient": strict_value,
                "strict_minus_manuscript": strict_value - reference,
                "sign_matches": int(np.sign(strict_value)) == int(np.sign(reference)),
                "comparison_note": (
                    "The manuscript value came from an earlier sample construction; differences must be "
                    "reported before changing the paper claim."
                ),
            }
        )
    return rows


def save_comparison_plot(comparison: pd.DataFrame, output_dir: Path, plot_dpi: int) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "RQ1 plotting requires matplotlib. Install the dependencies declared in pyproject.toml."
        ) from error

    model_labels = {
        PER_SECOND_MODEL: "Per-second",
        TOKEN_MODEL: "Token-normalized",
        TIMING_MODEL: "Token + timing",
    }
    colors = {
        PER_SECOND_MODEL: "#777777",
        TOKEN_MODEL: "#377eb8",
        TIMING_MODEL: "#e41a1c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=False, constrained_layout=True)
    for axis, term, title in zip(axes, STANCE_TERMS, ("Agreement movement", "Disagreement movement")):
        selected = comparison[comparison["term"] == term].set_index("model_name").loc[list(HEADLINE_MODELS)]
        y_positions = np.arange(len(HEADLINE_MODELS), dtype=float)
        for position, model_name in enumerate(HEADLINE_MODELS):
            row = selected.loc[model_name]
            coefficient = float(row["coefficient"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            axis.errorbar(
                coefficient,
                float(position),
                xerr=np.asarray([[coefficient - low], [high - coefficient]]),
                fmt="o",
                color=colors[model_name],
                ecolor=colors[model_name],
                capsize=4,
                markersize=6,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_yticks(y_positions, [model_labels[name] for name in HEADLINE_MODELS])
        axis.invert_yaxis()
        axis.set_xlabel("Coefficient with episode-clustered 95% CI")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("RQ1 stance-density estimates under explicit timing adjustment")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "rq1_timing_comparison.png"
    pdf_path = output_dir / "rq1_timing_comparison.pdf"
    png_temporary = png_path.with_suffix(".png.tmp")
    pdf_temporary = pdf_path.with_suffix(".pdf.tmp")
    fig.savefig(png_temporary, format="png", dpi=plot_dpi, bbox_inches="tight")
    fig.savefig(pdf_temporary, format="pdf", bbox_inches="tight")
    plt.close(fig)
    png_temporary.replace(png_path)
    pdf_temporary.replace(pdf_path)
    return [pdf_path, png_path]


def git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty_output)}


def package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "statsmodels", "matplotlib", "tqdm")
    }


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "coefficients": output_dir / "rq1_timing_coefficients.csv",
        "stance_comparison": output_dir / "rq1_timing_stance_comparison.csv",
        "model_fit": output_dir / "rq1_timing_model_fit.csv",
        "observations": output_dir / "rq1_timing_observations.csv",
        "data_audit": output_dir / "rq1_timing_data_audit.json",
        "summary": output_dir / "rq1_timing_summary.json",
        "plot_pdf": output_dir / "rq1_timing_comparison.pdf",
        "plot_png": output_dir / "rq1_timing_comparison.png",
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    logger.info(
        "Starting RQ1 timing analysis | version=%s | input=%s | output=%s",
        SCRIPT_VERSION,
        args.data_dir,
        args.output_dir,
    )
    frame, audit = build_dataset(
        Path(args.data_dir),
        list(args.categories or []),
        args.category_data_subdir,
        args.max_episodes,
        not args.no_tqdm,
    )
    results = fit_models(frame)
    coefficients = coefficient_frame(results)
    model_fit = model_fit_frame(results, frame)
    comparison = stance_comparison_frame(coefficients, frame)
    output_dir = Path(args.output_dir)
    paths = output_paths(output_dir)

    write_csv(paths["observations"], frame)
    write_csv(paths["coefficients"], coefficients)
    write_csv(paths["stance_comparison"], comparison)
    write_csv(paths["model_fit"], model_fit)
    save_comparison_plot(comparison, output_dir, args.plot_dpi)

    repository_state = git_state()
    versions = package_versions()
    audit.update(
        {
            "experiment": "ICLR RQ1 timing-aware explicit-implicit density analysis",
            "script_version": SCRIPT_VERSION,
            "script_sha256": file_sha256(Path(__file__)),
            "formulas": FORMULAS,
            "common_model_sample_verified": True,
            "word_count_definition": r"count of Unicode regex matches for \b\w+\b in turn_text",
            "density_per_second_definition": "(explicit_count / (assumption_count + 1)) / duration_seconds",
            "density_per_token_definition": "(explicit_count / (assumption_count + 1)) / word_count",
            "strict_window_definition": "three consecutive raw substantive turns with speaker changes at both boundaries",
            "inference": "OLS with episode-clustered covariance and two-sided 95% Wald intervals",
            "git": repository_state,
            "package_versions": versions,
            "observation_csv_sha256": file_sha256(paths["observations"]),
        }
    )
    write_json(paths["data_audit"], audit)

    interpretations = [coefficient_interpretation(comparison, term) for term in STANCE_TERMS]
    interval_survives = all(
        row["status"] == "direction_preserved_and_interval_excludes_zero"
        for row in interpretations
    )
    artifact_hashes = {
        path.name: file_sha256(path)
        for key, path in paths.items()
        if key != "summary" and path.exists()
    }
    summary: dict[str, object] = {
        "experiment": "ICLR RQ1 timing-aware explicit-implicit density analysis",
        "script_version": SCRIPT_VERSION,
        "configuration": {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "categories": list(args.categories or ["all"]),
            "category_data_subdir": args.category_data_subdir,
            "max_episodes": args.max_episodes,
            "plot_dpi": args.plot_dpi,
        },
        "git": repository_state,
        "package_versions": versions,
        "formulas": FORMULAS,
        "transition_count": len(frame),
        "episode_count": int(frame["episode"].nunique()),
        "headline_coefficients": {
            f"{model_name}:{term}": extract_coefficient(coefficients, model_name, term)
            for model_name in HEADLINE_MODELS
            for term in STANCE_TERMS
        },
        "timing_adjustment_interpretation": interpretations,
        "manuscript_reference_comparison": manuscript_reference_comparison(coefficients),
        "both_directional_intervals_survive_timing_adjustment": interval_survives,
        "estimands": {
            PER_SECOND_MODEL: "combined stance-density-pacing association",
            TOKEN_MODEL: "association after removing duration from the outcome",
            TIMING_MODEL: "conditional association holding observed duration and response timing constant",
        },
        "causal_claim": False,
        "duration_note": "Duration may be a mediator rather than a conventional confound.",
        "artifact_sha256": artifact_hashes,
        "summary_self_hash_excluded": True,
    }
    write_json(paths["summary"], summary)
    logger.info(
        "RQ1 timing analysis complete | transitions=%d | episodes=%d | output=%s",
        len(frame),
        int(frame["episode"].nunique()),
        output_dir,
    )
    return summary


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-contained ICLR RQ1 timing-aware density analysis."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--category_data_subdir", default=DEFAULT_CATEGORY_DATA_SUBDIR)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--plot_dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--no_tqdm", action="store_true")
    args = parser.parse_args(argv)
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max_episodes must be positive")
    if args.plot_dpi < 72:
        parser.error("--plot_dpi must be at least 72")
    return args


def main() -> None:
    summary = run_analysis(parse_args(None))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
