from __future__ import annotations

"""Step-wise timing sensitivity analysis with a duration-free iceberg-ratio outcome."""

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


SCRIPT_VERSION = "5.1.0"
DEFAULT_DATA_DIR = Path("data/stance_labeled/1024")
DEFAULT_OUTPUT_DIR = Path("iclr/rq1_timing_analysis/results")
DEFAULT_CATEGORY_DATA_SUBDIR = "parsed"
DEFAULT_PLOT_DPI = 300
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

RATIO_MODEL = "iceberg_ratio"
DURATION_MODEL = "iceberg_ratio_duration_adjusted"
TIMING_MODEL = "iceberg_ratio_timing_adjusted"
PREVIOUS_DURATION_MODEL = "iceberg_ratio_timing_previous_duration"
STANCE_ONLY_MODEL = "step_1_stance_only"
LAGGED_STANCE_MODEL = "step_2_lagged_stance"
PREVIOUS_OUTCOME_MODEL = "step_3_previous_outcome"
TIMELINE_MODEL = "step_4_timeline"
MODEL_ORDER = (
    RATIO_MODEL,
    DURATION_MODEL,
    TIMING_MODEL,
    PREVIOUS_DURATION_MODEL,
)
HEADLINE_MODELS = (RATIO_MODEL, DURATION_MODEL, TIMING_MODEL)
STEPWISE_MODEL_ORDER = (
    STANCE_ONLY_MODEL,
    LAGGED_STANCE_MODEL,
    PREVIOUS_OUTCOME_MODEL,
    TIMELINE_MODEL,
    RATIO_MODEL,
    DURATION_MODEL,
    TIMING_MODEL,
    PREVIOUS_DURATION_MODEL,
)
STEPWISE_ADDED_GROUPS = {
    STANCE_ONLY_MODEL: "stance movement",
    LAGGED_STANCE_MODEL: "lagged stance movement",
    PREVIOUS_OUTCOME_MODEL: "previous iceberg ratio",
    TIMELINE_MODEL: "timeline position",
    RATIO_MODEL: "category fixed effects",
    DURATION_MODEL: "current-turn duration",
    TIMING_MODEL: "pre-turn gap and overlap",
    PREVIOUS_DURATION_MODEL: "previous-turn duration",
}
STANCE_TERMS = ("agree_move", "disagree_move")
ORIGINAL_EXP2_REFERENCE_COEFFICIENTS = {
    "agree_move": -0.024848939290319824,
    "disagree_move": 0.015577205239494248,
}

FORMULAS = {
    RATIO_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move + previous_log_iceberg_ratio + "
        "timeline_position + I(timeline_position ** 2) + C(category)"
    ),
    DURATION_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move + previous_log_iceberg_ratio + "
        "timeline_position + I(timeline_position ** 2) + C(category) + log_duration"
    ),
    TIMING_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move + previous_log_iceberg_ratio + "
        "timeline_position + I(timeline_position ** 2) + C(category) + log_duration + "
        "log_gap + overlap"
    ),
    PREVIOUS_DURATION_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move + previous_log_iceberg_ratio + "
        "timeline_position + I(timeline_position ** 2) + C(category) + log_duration + "
        "log_gap + overlap + previous_log_duration"
    ),
}

STEPWISE_FORMULAS = {
    STANCE_ONLY_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move"
    ),
    LAGGED_STANCE_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move"
    ),
    PREVIOUS_OUTCOME_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move + previous_log_iceberg_ratio"
    ),
    TIMELINE_MODEL: (
        "delta_log_iceberg_ratio ~ agree_move + disagree_move + "
        "lag_agree_move + lag_disagree_move + previous_log_iceberg_ratio + "
        "timeline_position + I(timeline_position ** 2)"
    ),
    RATIO_MODEL: FORMULAS[RATIO_MODEL],
    DURATION_MODEL: FORMULAS[DURATION_MODEL],
    TIMING_MODEL: FORMULAS[TIMING_MODEL],
    PREVIOUS_DURATION_MODEL: FORMULAS[PREVIOUS_DURATION_MODEL],
}

CURRENT_STANCE_GROUP = "current_stance"
LAGGED_STANCE_GROUP = "lagged_stance"
PREVIOUS_OUTCOME_GROUP = "previous_outcome"
TIMELINE_GROUP = "timeline"
CATEGORY_GROUP = "category_fixed_effects"
CURRENT_DURATION_GROUP = "current_duration"
RESPONSE_TIMING_GROUP = "response_timing"
PREVIOUS_DURATION_GROUP = "previous_duration"
SPECIFICATION_GROUP_ORDER = (
    CURRENT_STANCE_GROUP,
    LAGGED_STANCE_GROUP,
    PREVIOUS_OUTCOME_GROUP,
    TIMELINE_GROUP,
    CATEGORY_GROUP,
    CURRENT_DURATION_GROUP,
    RESPONSE_TIMING_GROUP,
    PREVIOUS_DURATION_GROUP,
)
CONTROL_GROUP_ORDER = SPECIFICATION_GROUP_ORDER[1:]
VARIABLE_GROUP_LABELS = {
    CURRENT_STANCE_GROUP: "Current stance (agreement + disagreement)",
    LAGGED_STANCE_GROUP: "Lagged stance (agreement + disagreement)",
    PREVIOUS_OUTCOME_GROUP: "Previous outcome",
    TIMELINE_GROUP: "Timeline (linear + quadratic)",
    CATEGORY_GROUP: "Category fixed effects",
    CURRENT_DURATION_GROUP: "Current-turn duration",
    RESPONSE_TIMING_GROUP: "Response timing (gap + overlap)",
    PREVIOUS_DURATION_GROUP: "Previous-turn duration",
}
VARIABLE_GROUP_TERMS = {
    CURRENT_STANCE_GROUP: ("agree_move", "disagree_move"),
    LAGGED_STANCE_GROUP: ("lag_agree_move", "lag_disagree_move"),
    PREVIOUS_OUTCOME_GROUP: ("previous_log_iceberg_ratio",),
    TIMELINE_GROUP: ("timeline_position", "I(timeline_position ** 2)"),
    CATEGORY_GROUP: ("C(category)",),
    CURRENT_DURATION_GROUP: ("log_duration",),
    RESPONSE_TIMING_GROUP: ("log_gap", "overlap"),
    PREVIOUS_DURATION_GROUP: ("previous_log_duration",),
}
EXP2_VARIABLE_GROUPS = SPECIFICATION_GROUP_ORDER[:5]
FULL_VARIABLE_GROUPS = SPECIFICATION_GROUP_ORDER


def formula_for_variable_groups(
    included_groups: tuple[str, ...],
    response_variable: str,
    previous_outcome_term: str,
) -> str:
    if not included_groups or included_groups[0] != CURRENT_STANCE_GROUP:
        raise ValueError("Every specification must begin with the current-stance group")
    unknown_groups = sorted(set(included_groups).difference(VARIABLE_GROUP_TERMS))
    if unknown_groups:
        raise KeyError(f"Unknown specification variable groups: {unknown_groups}")
    if len(set(included_groups)) != len(included_groups):
        raise ValueError(f"Specification repeats a variable group: {included_groups}")
    terms = [
        term
        for group_name in included_groups
        for term in (
            (previous_outcome_term,)
            if group_name == PREVIOUS_OUTCOME_GROUP
            else VARIABLE_GROUP_TERMS[group_name]
        )
    ]
    return response_variable + " ~ " + " + ".join(terms)


SPECIFICATION_MODEL_ORDER = (
    "spec_01_stance_core",
    "spec_02_add_lagged_stance",
    "spec_03_add_previous_outcome",
    "spec_04_add_timeline",
    "spec_05_add_category",
    "spec_06_add_current_duration",
    "spec_07_add_response_timing",
    "spec_08_add_previous_duration",
    "spec_09_exp2_baseline",
    "spec_10_full_timing",
    "spec_11_drop_lagged_stance",
    "spec_12_drop_previous_outcome",
    "spec_13_drop_timeline",
    "spec_14_drop_category",
    "spec_15_drop_current_duration",
    "spec_16_drop_response_timing",
    "spec_17_drop_previous_duration",
)
SPECIFICATION_INCLUDED_GROUPS = {
    SPECIFICATION_MODEL_ORDER[0]: (CURRENT_STANCE_GROUP,),
    SPECIFICATION_MODEL_ORDER[1]: (CURRENT_STANCE_GROUP, LAGGED_STANCE_GROUP),
    SPECIFICATION_MODEL_ORDER[2]: (CURRENT_STANCE_GROUP, PREVIOUS_OUTCOME_GROUP),
    SPECIFICATION_MODEL_ORDER[3]: (CURRENT_STANCE_GROUP, TIMELINE_GROUP),
    SPECIFICATION_MODEL_ORDER[4]: (CURRENT_STANCE_GROUP, CATEGORY_GROUP),
    SPECIFICATION_MODEL_ORDER[5]: (CURRENT_STANCE_GROUP, CURRENT_DURATION_GROUP),
    SPECIFICATION_MODEL_ORDER[6]: (CURRENT_STANCE_GROUP, RESPONSE_TIMING_GROUP),
    SPECIFICATION_MODEL_ORDER[7]: (CURRENT_STANCE_GROUP, PREVIOUS_DURATION_GROUP),
    SPECIFICATION_MODEL_ORDER[8]: EXP2_VARIABLE_GROUPS,
    SPECIFICATION_MODEL_ORDER[9]: FULL_VARIABLE_GROUPS,
    SPECIFICATION_MODEL_ORDER[10]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != LAGGED_STANCE_GROUP
    ),
    SPECIFICATION_MODEL_ORDER[11]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != PREVIOUS_OUTCOME_GROUP
    ),
    SPECIFICATION_MODEL_ORDER[12]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != TIMELINE_GROUP
    ),
    SPECIFICATION_MODEL_ORDER[13]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != CATEGORY_GROUP
    ),
    SPECIFICATION_MODEL_ORDER[14]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != CURRENT_DURATION_GROUP
    ),
    SPECIFICATION_MODEL_ORDER[15]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != RESPONSE_TIMING_GROUP
    ),
    SPECIFICATION_MODEL_ORDER[16]: tuple(
        group_name for group_name in FULL_VARIABLE_GROUPS if group_name != PREVIOUS_DURATION_GROUP
    ),
}
SPECIFICATION_SHORT_LABELS = {
    SPECIFICATION_MODEL_ORDER[0]: "Core",
    SPECIFICATION_MODEL_ORDER[1]: "+ Lagged stance",
    SPECIFICATION_MODEL_ORDER[2]: "+ Previous outcome",
    SPECIFICATION_MODEL_ORDER[3]: "+ Timeline",
    SPECIFICATION_MODEL_ORDER[4]: "+ Category FE",
    SPECIFICATION_MODEL_ORDER[5]: "+ Current duration",
    SPECIFICATION_MODEL_ORDER[6]: "+ Gap/overlap",
    SPECIFICATION_MODEL_ORDER[7]: "+ Previous duration",
    SPECIFICATION_MODEL_ORDER[8]: "Exp2 controls",
    SPECIFICATION_MODEL_ORDER[9]: "Full timing",
    SPECIFICATION_MODEL_ORDER[10]: "− Lagged stance",
    SPECIFICATION_MODEL_ORDER[11]: "− Previous outcome",
    SPECIFICATION_MODEL_ORDER[12]: "− Timeline",
    SPECIFICATION_MODEL_ORDER[13]: "− Category FE",
    SPECIFICATION_MODEL_ORDER[14]: "− Current duration",
    SPECIFICATION_MODEL_ORDER[15]: "− Gap/overlap",
    SPECIFICATION_MODEL_ORDER[16]: "− Previous duration",
}
SPECIFICATION_FAMILIES = {
    SPECIFICATION_MODEL_ORDER[0]: "reference",
    **{model_name: "add_one_to_core" for model_name in SPECIFICATION_MODEL_ORDER[1:8]},
    SPECIFICATION_MODEL_ORDER[8]: "reference",
    SPECIFICATION_MODEL_ORDER[9]: "reference",
    **{model_name: "remove_one_from_full" for model_name in SPECIFICATION_MODEL_ORDER[10:]},
}
SPECIFICATION_REFERENCE_MODELS: dict[str, str | None] = {
    SPECIFICATION_MODEL_ORDER[0]: None,
    **{
        model_name: SPECIFICATION_MODEL_ORDER[0]
        for model_name in SPECIFICATION_MODEL_ORDER[1:9]
    },
    SPECIFICATION_MODEL_ORDER[9]: SPECIFICATION_MODEL_ORDER[8],
    **{
        model_name: SPECIFICATION_MODEL_ORDER[9]
        for model_name in SPECIFICATION_MODEL_ORDER[10:]
    },
}
SPECIFICATION_CHANGED_GROUPS: dict[str, str] = {
    SPECIFICATION_MODEL_ORDER[0]: CURRENT_STANCE_GROUP,
    **{
        model_name: group_name
        for model_name, group_name in zip(
            SPECIFICATION_MODEL_ORDER[1:8],
            CONTROL_GROUP_ORDER,
            strict=True,
        )
    },
    SPECIFICATION_MODEL_ORDER[8]: "exp2_control_block",
    SPECIFICATION_MODEL_ORDER[9]: "timing_control_block",
    **{
        model_name: group_name
        for model_name, group_name in zip(
            SPECIFICATION_MODEL_ORDER[10:],
            CONTROL_GROUP_ORDER,
            strict=True,
        )
    },
}
SPECIFICATION_FORMULAS = {
    model_name: formula_for_variable_groups(
        SPECIFICATION_INCLUDED_GROUPS[model_name],
        "delta_log_iceberg_ratio",
        "previous_log_iceberg_ratio",
    )
    for model_name in SPECIFICATION_MODEL_ORDER
}
IMPLICIT_ASSUMPTION_SPECIFICATION_FORMULAS = {
    model_name: formula_for_variable_groups(
        SPECIFICATION_INCLUDED_GROUPS[model_name],
        "delta_log_assumption_count",
        "previous_log_assumption_count",
    )
    for model_name in SPECIFICATION_MODEL_ORDER
}
EXPLICIT_CLAIM_SPECIFICATION_FORMULAS = {
    model_name: formula_for_variable_groups(
        SPECIFICATION_INCLUDED_GROUPS[model_name],
        "delta_log_explicit_count",
        "previous_log_explicit_count",
    )
    for model_name in SPECIFICATION_MODEL_ORDER
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
    lag_agree_move: float
    lag_disagree_move: float
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
    previous_iceberg_ratio: float
    iceberg_ratio: float
    previous_log_iceberg_ratio: float
    log_iceberg_ratio: float
    delta_log_iceberg_ratio: float
    previous_log_assumption_count: float
    log_assumption_count: float
    delta_log_assumption_count: float
    previous_log_explicit_count: float
    log_explicit_count: float
    delta_log_explicit_count: float
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


class SpecificationPlotConfig(TypedDict):
    previous_outcome_term: str
    previous_outcome_label: str
    figure_title: str
    response_label: str
    estimated_change_label: str
    output_stem: str


RATIO_SPECIFICATION_PLOT: SpecificationPlotConfig = {
    "previous_outcome_term": "previous_log_iceberg_ratio",
    "previous_outcome_label": "Previous iceberg ratio",
    "figure_title": "Iceberg ratio: grouped specifications",
    "response_label": "Δ log(1 + iceberg ratio)",
    "estimated_change_label": "Estimated change in 1 + iceberg ratio (%)",
    "output_stem": "rq1_specification_panel",
}
IMPLICIT_ASSUMPTION_SPECIFICATION_PLOT: SpecificationPlotConfig = {
    "previous_outcome_term": "previous_log_assumption_count",
    "previous_outcome_label": "Previous implicit assumptions",
    "figure_title": "Implicit assumptions: grouped specifications",
    "response_label": "Δ log(1 + implicit assumptions)",
    "estimated_change_label": "Estimated change in 1 + implicit assumptions (%)",
    "output_stem": "rq1_specification_panel_implicit_assumptions",
}
EXPLICIT_CLAIM_SPECIFICATION_PLOT: SpecificationPlotConfig = {
    "previous_outcome_term": "previous_log_explicit_count",
    "previous_outcome_label": "Previous explicit claims",
    "figure_title": "Explicit claims: grouped specifications",
    "response_label": "Δ log(1 + explicit claims)",
    "estimated_change_label": "Estimated change in 1 + explicit claims (%)",
    "output_stem": "rq1_specification_panel_explicit_claims",
}


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


def iceberg_ratio(explicit_count: int, assumption_count: int) -> float:
    return float(explicit_count) / float(assumption_count + 1)


def normalized_iceberg_ratio(
    explicit_count: int,
    assumption_count: int,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        raise ValueError(f"Normalization denominator must be positive, received {denominator}")
    return iceberg_ratio(explicit_count, assumption_count) / denominator


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

    previous_ratio = iceberg_ratio(previous_explicit, previous_assumptions)
    current_ratio = iceberg_ratio(current_explicit, current_assumptions)
    previous_per_second = normalized_iceberg_ratio(
        previous_explicit,
        previous_assumptions,
        previous_duration,
    )
    current_per_second = normalized_iceberg_ratio(
        current_explicit,
        current_assumptions,
        current_duration,
    )
    previous_per_token = normalized_iceberg_ratio(
        previous_explicit,
        previous_assumptions,
        float(previous["word_count"]),
    )
    current_per_token = normalized_iceberg_ratio(
        current_explicit,
        current_assumptions,
        float(current["word_count"]),
    )
    previous_log_ratio = math.log1p(previous_ratio)
    current_log_ratio = math.log1p(current_ratio)
    previous_log_assumptions = math.log1p(previous_assumptions)
    current_log_assumptions = math.log1p(current_assumptions)
    previous_log_explicit = math.log1p(previous_explicit)
    current_log_explicit = math.log1p(current_explicit)
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
        "lag_agree_move": max(lag_delta_stance, 0.0),
        "lag_disagree_move": max(-lag_delta_stance, 0.0),
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
        "previous_iceberg_ratio": previous_ratio,
        "iceberg_ratio": current_ratio,
        "previous_log_iceberg_ratio": previous_log_ratio,
        "log_iceberg_ratio": current_log_ratio,
        "delta_log_iceberg_ratio": current_log_ratio - previous_log_ratio,
        "previous_log_assumption_count": previous_log_assumptions,
        "log_assumption_count": current_log_assumptions,
        "delta_log_assumption_count": (
            current_log_assumptions - previous_log_assumptions
        ),
        "previous_log_explicit_count": previous_log_explicit,
        "log_explicit_count": current_log_explicit,
        "delta_log_explicit_count": current_log_explicit - previous_log_explicit,
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
        "delta_log_iceberg_ratio",
        "delta_log_assumption_count",
        "delta_log_explicit_count",
        "agree_move",
        "disagree_move",
        "lag_agree_move",
        "lag_disagree_move",
        "previous_log_iceberg_ratio",
        "previous_log_assumption_count",
        "previous_log_explicit_count",
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


def fit_formula_models(
    frame: pd.DataFrame,
    formulas: dict[str, str],
    model_order: tuple[str, ...],
) -> dict[str, RegressionResult]:
    validate_model_frame(frame)
    formula_api = require_analysis_dependencies()
    model_frame = frame.copy()
    model_frame["category"] = model_frame["category"].astype(object)
    model_frame["episode"] = model_frame["episode"].astype(object)
    results: dict[str, RegressionResult] = {}
    for model_name in model_order:
        if model_name not in formulas:
            raise KeyError(f"Missing regression formula for model: {model_name}")
        formula = formulas[model_name]
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


def fit_models(frame: pd.DataFrame) -> dict[str, RegressionResult]:
    return fit_formula_models(frame, FORMULAS, MODEL_ORDER)


def fit_stepwise_models(frame: pd.DataFrame) -> dict[str, RegressionResult]:
    return fit_formula_models(frame, STEPWISE_FORMULAS, STEPWISE_MODEL_ORDER)


def fit_specification_models(
    frame: pd.DataFrame,
    reusable_results: dict[str, RegressionResult],
) -> dict[str, RegressionResult]:
    unknown_reusable_models = sorted(set(reusable_results).difference(STEPWISE_FORMULAS))
    if unknown_reusable_models:
        raise KeyError(f"Cannot reuse unknown model results: {unknown_reusable_models}")
    reusable_by_formula = {
        STEPWISE_FORMULAS[model_name]: result
        for model_name, result in reusable_results.items()
    }
    missing_model_order = tuple(
        model_name
        for model_name in SPECIFICATION_MODEL_ORDER
        if SPECIFICATION_FORMULAS[model_name] not in reusable_by_formula
    )
    missing_formulas = {
        model_name: SPECIFICATION_FORMULAS[model_name]
        for model_name in missing_model_order
    }
    fitted_results = (
        fit_formula_models(frame, missing_formulas, missing_model_order)
        if missing_model_order
        else {}
    )
    results: dict[str, RegressionResult] = {}
    for model_name in SPECIFICATION_MODEL_ORDER:
        formula = SPECIFICATION_FORMULAS[model_name]
        if formula in reusable_by_formula:
            results[model_name] = reusable_by_formula[formula]
        else:
            results[model_name] = fitted_results[model_name]
    return results


def coefficient_frame_for_order(
    results: dict[str, RegressionResult],
    model_order: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in model_order:
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


def coefficient_frame(results: dict[str, RegressionResult]) -> pd.DataFrame:
    return coefficient_frame_for_order(results, MODEL_ORDER)


def stepwise_coefficient_frame(results: dict[str, RegressionResult]) -> pd.DataFrame:
    return coefficient_frame_for_order(results, STEPWISE_MODEL_ORDER)


def specification_coefficient_frame(
    results: dict[str, RegressionResult],
) -> pd.DataFrame:
    coefficients = coefficient_frame_for_order(results, SPECIFICATION_MODEL_ORDER)
    model_numbers = {
        model_name: model_number
        for model_number, model_name in enumerate(SPECIFICATION_MODEL_ORDER, start=1)
    }
    coefficients.insert(
        0,
        "model_number",
        coefficients["model_name"].map(model_numbers).astype(int),
    )
    coefficients.insert(
        2,
        "short_label",
        coefficients["model_name"].map(SPECIFICATION_SHORT_LABELS),
    )
    return coefficients


def model_fit_frame_for_order(
    results: dict[str, RegressionResult],
    frame: pd.DataFrame,
    model_order: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in model_order:
        result = results[model_name]
        rows.append(
            {
                "model_name": model_name,
                "response_variable": "delta_log_iceberg_ratio",
                "transition_count": len(frame),
                "episode_count": int(frame["episode"].nunique()),
                "r_squared": float(result.rsquared),
                "adjusted_r_squared": float(result.rsquared_adj),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "aic_bic_comparison_group": "duration_free_iceberg_ratio_outcome",
                "aic_bic_comparable_to_other_models": True,
            }
        )
    fit_frame = pd.DataFrame(rows)
    if fit_frame["transition_count"].nunique() != 1 or fit_frame["episode_count"].nunique() != 1:
        raise RuntimeError("Regression specifications did not use identical samples")
    return fit_frame


def model_fit_frame(
    results: dict[str, RegressionResult],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return model_fit_frame_for_order(results, frame, MODEL_ORDER)


def stepwise_model_fit_frame(
    results: dict[str, RegressionResult],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return model_fit_frame_for_order(results, frame, STEPWISE_MODEL_ORDER)


def specification_panel_frame(
    results: dict[str, RegressionResult],
    frame: pd.DataFrame,
    formulas: dict[str, str],
) -> pd.DataFrame:
    if set(results) != set(SPECIFICATION_MODEL_ORDER):
        missing = sorted(set(SPECIFICATION_MODEL_ORDER).difference(results))
        extra = sorted(set(results).difference(SPECIFICATION_MODEL_ORDER))
        raise KeyError(f"Specification results do not match the model order; missing={missing}, extra={extra}")
    if set(formulas) != set(SPECIFICATION_MODEL_ORDER):
        missing = sorted(set(SPECIFICATION_MODEL_ORDER).difference(formulas))
        extra = sorted(set(formulas).difference(SPECIFICATION_MODEL_ORDER))
        raise KeyError(
            "Specification formulas do not match the model order; "
            f"missing={missing}, extra={extra}"
        )
    rows: list[dict[str, object]] = []
    for model_number, model_name in enumerate(SPECIFICATION_MODEL_ORDER, start=1):
        result = results[model_name]
        if int(result.nobs) != len(frame):
            raise RuntimeError(
                f"Specification {model_name} used {int(result.nobs)} rows; expected {len(frame)}"
            )
        intervals = result.conf_int(alpha=0.05)
        reference_model = SPECIFICATION_REFERENCE_MODELS[model_name]
        reference_adjusted_r_squared = (
            None
            if reference_model is None
            else float(results[reference_model].rsquared_adj)
        )
        adjusted_r_squared = float(result.rsquared_adj)
        included_groups = SPECIFICATION_INCLUDED_GROUPS[model_name]
        row: dict[str, object] = {
            "model_number": model_number,
            "model_name": model_name,
            "short_label": SPECIFICATION_SHORT_LABELS[model_name],
            "comparison_family": SPECIFICATION_FAMILIES[model_name],
            "reference_model": reference_model,
            "changed_group": SPECIFICATION_CHANGED_GROUPS[model_name],
            "included_variable_groups": "|".join(included_groups),
            "formula": formulas[model_name],
            "agree_move_coefficient": float(result.params["agree_move"]),
            "agree_move_estimated_change_percent": 100.0
            * math.expm1(float(result.params["agree_move"])),
            "agree_move_clustered_se": float(result.bse["agree_move"]),
            "agree_move_ci95_low": float(intervals.loc["agree_move", 0]),
            "agree_move_ci95_high": float(intervals.loc["agree_move", 1]),
            "disagree_move_coefficient": float(result.params["disagree_move"]),
            "disagree_move_estimated_change_percent": 100.0
            * math.expm1(float(result.params["disagree_move"])),
            "disagree_move_clustered_se": float(result.bse["disagree_move"]),
            "disagree_move_ci95_low": float(intervals.loc["disagree_move", 0]),
            "disagree_move_ci95_high": float(intervals.loc["disagree_move", 1]),
            "r_squared": float(result.rsquared),
            "adjusted_r_squared": adjusted_r_squared,
            "reference_adjusted_r_squared": reference_adjusted_r_squared,
            "delta_adjusted_r_squared": (
                None
                if reference_adjusted_r_squared is None
                else adjusted_r_squared - reference_adjusted_r_squared
            ),
            "transition_count": len(frame),
            "episode_count": int(frame["episode"].nunique()),
        }
        row.update(
            {
                f"includes_{group_name}": group_name in included_groups
                for group_name in SPECIFICATION_GROUP_ORDER
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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


def attenuation_percent(baseline_coefficient: float, adjusted_coefficient: float) -> float:
    if baseline_coefficient == 0.0:
        raise ZeroDivisionError("Cannot calculate attenuation from a zero baseline coefficient")
    return 100.0 * (1.0 - abs(adjusted_coefficient) / abs(baseline_coefficient))


def stance_comparison_frame(
    coefficients: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in STANCE_TERMS:
        baseline = extract_coefficient(coefficients, RATIO_MODEL, term)
        duration = extract_coefficient(coefficients, DURATION_MODEL, term)
        timing = extract_coefficient(coefficients, TIMING_MODEL, term)
        baseline_coefficient = float(cast(float, baseline["coefficient"]))
        duration_coefficient = float(cast(float, duration["coefficient"]))
        timing_coefficient = float(cast(float, timing["coefficient"]))
        attenuation_from_baseline: dict[str, float] = {
            DURATION_MODEL: attenuation_percent(baseline_coefficient, duration_coefficient),
            TIMING_MODEL: attenuation_percent(baseline_coefficient, timing_coefficient),
        }
        incremental_attenuation: dict[str, float] = {
            DURATION_MODEL: attenuation_percent(baseline_coefficient, duration_coefficient),
            TIMING_MODEL: attenuation_percent(duration_coefficient, timing_coefficient),
        }
        incremental_reference: dict[str, str] = {
            DURATION_MODEL: RATIO_MODEL,
            TIMING_MODEL: DURATION_MODEL,
        }
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
                    "attenuation_from_baseline_percent": attenuation_from_baseline.get(
                        model_name
                    ),
                    "attenuation_reference_model": (
                        RATIO_MODEL if model_name in attenuation_from_baseline else None
                    ),
                    "incremental_attenuation_percent": incremental_attenuation.get(model_name),
                    "incremental_reference_model": incremental_reference.get(model_name),
                    "timing_attenuation_percent": (
                        attenuation_from_baseline.get(model_name)
                        if model_name == TIMING_MODEL
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def stepwise_stance_comparison_frame(
    coefficients: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in STANCE_TERMS:
        term_rows: list[dict[str, object]] = []
        previous_coefficient: float | None = None
        previous_model: str | None = None
        for stage_number, model_name in enumerate(STEPWISE_MODEL_ORDER, start=1):
            coefficient = extract_coefficient(coefficients, model_name, term)
            coefficient_value = float(cast(float, coefficient["coefficient"]))
            incremental_change = (
                None
                if previous_coefficient is None
                else coefficient_value - previous_coefficient
            )
            term_rows.append(
                {
                    "stage_number": stage_number,
                    "stance_direction": (
                        "agreement" if term == "agree_move" else "disagreement"
                    ),
                    "term": term,
                    "model_name": model_name,
                    "added_variable_group": STEPWISE_ADDED_GROUPS[model_name],
                    "reference_model": previous_model,
                    "coefficient": coefficient_value,
                    "clustered_se": coefficient["clustered_se"],
                    "ci95_low": coefficient["ci95_low"],
                    "ci95_high": coefficient["ci95_high"],
                    "p_value": coefficient["p_value"],
                    "coefficient_change_from_previous": incremental_change,
                    "absolute_change_from_previous": (
                        None if incremental_change is None else abs(incremental_change)
                    ),
                    "sign_changed_from_previous": (
                        None
                        if previous_coefficient is None
                        else int(np.sign(coefficient_value))
                        != int(np.sign(previous_coefficient))
                    ),
                    "ci_excludes_zero": (
                        float(cast(float, coefficient["ci95_low"])) > 0.0
                        or float(cast(float, coefficient["ci95_high"])) < 0.0
                    ),
                    "transition_count": len(frame),
                    "episode_count": int(frame["episode"].nunique()),
                }
            )
            previous_coefficient = coefficient_value
            previous_model = model_name
        incremental_rows = [
            row
            for row in term_rows
            if row["absolute_change_from_previous"] is not None
        ]
        largest_change = max(
            incremental_rows,
            key=lambda row: float(cast(float, row["absolute_change_from_previous"])),
        )
        largest_stage = int(cast(int, largest_change["stage_number"]))
        rows.extend(
            {
                **row,
                "largest_incremental_change_for_term": (
                    int(cast(int, row["stage_number"])) == largest_stage
                ),
            }
            for row in term_rows
        )
    return pd.DataFrame(rows)


def stepwise_largest_changes(comparison: pd.DataFrame) -> list[dict[str, object]]:
    selected = comparison[comparison["largest_incremental_change_for_term"]].copy()
    columns = [
        "term",
        "stance_direction",
        "stage_number",
        "model_name",
        "added_variable_group",
        "reference_model",
        "coefficient",
        "coefficient_change_from_previous",
        "sign_changed_from_previous",
    ]
    return [cast(dict[str, object], row) for row in selected[columns].to_dict("records")]


def direction_status(
    term: str,
    baseline_coefficient: float,
    adjusted_coefficient: float,
    adjusted_ci95_low: float,
    adjusted_ci95_high: float,
) -> str:
    if term not in STANCE_TERMS:
        raise ValueError(f"Unsupported stance term: {term}")
    if int(np.sign(baseline_coefficient)) != int(np.sign(adjusted_coefficient)):
        return "sign_reversed_after_adjustment"
    if adjusted_ci95_low > 0.0 or adjusted_ci95_high < 0.0:
        return "direction_preserved_and_interval_excludes_zero"
    return "direction_preserved_but_interval_includes_zero"


def coefficient_interpretation(comparison: pd.DataFrame, term: str) -> dict[str, object]:
    baseline = comparison[
        (comparison["term"] == term) & (comparison["model_name"] == RATIO_MODEL)
    ].iloc[0]
    duration = comparison[
        (comparison["term"] == term) & (comparison["model_name"] == DURATION_MODEL)
    ].iloc[0]
    timing = comparison[(comparison["term"] == term) & (comparison["model_name"] == TIMING_MODEL)].iloc[0]
    expected_negative = term == "agree_move"
    baseline_coefficient = float(baseline["coefficient"])
    duration_coefficient = float(duration["coefficient"])
    timing_coefficient = float(timing["coefficient"])
    duration_status = direction_status(
        term,
        baseline_coefficient,
        duration_coefficient,
        float(duration["ci95_low"]),
        float(duration["ci95_high"]),
    )
    timing_status = direction_status(
        term,
        baseline_coefficient,
        timing_coefficient,
        float(timing["ci95_low"]),
        float(timing["ci95_high"]),
    )
    if duration_status == "sign_reversed_after_adjustment":
        duration_status = "sign_reversed_after_duration_adjustment"
    if timing_status == "sign_reversed_after_adjustment":
        timing_status = "sign_reversed_after_timing_adjustment"
    return {
        "term": term,
        "direction": "agreement" if expected_negative else "disagreement",
        "ratio_baseline_coefficient": baseline_coefficient,
        "duration_adjusted_coefficient": duration_coefficient,
        "duration_adjusted_ci95_low": float(duration["ci95_low"]),
        "duration_adjusted_ci95_high": float(duration["ci95_high"]),
        "duration_attenuation_percent": float(
            duration["attenuation_from_baseline_percent"]
        ),
        "duration_status": duration_status,
        "timing_adjusted_coefficient": timing_coefficient,
        "timing_adjusted_ci95_low": float(timing["ci95_low"]),
        "timing_adjusted_ci95_high": float(timing["ci95_high"]),
        "attenuation_percent": float(timing["attenuation_from_baseline_percent"]),
        "gap_overlap_incremental_attenuation_percent": float(
            timing["incremental_attenuation_percent"]
        ),
        "status": timing_status,
    }


def exp2_reference_comparison(coefficients: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in STANCE_TERMS:
        ratio_sample = extract_coefficient(coefficients, RATIO_MODEL, term)
        ratio_sample_value = float(cast(float, ratio_sample["coefficient"]))
        reference = ORIGINAL_EXP2_REFERENCE_COEFFICIENTS[term]
        rows.append(
            {
                "term": term,
                "original_exp2_full_sample_coefficient": reference,
                "duration_free_ratio_coefficient": ratio_sample_value,
                "outcomes_comparable": False,
                "comparison_note": (
                    "The original Exp2 coefficient used per-second iceberg density; this "
                    "analysis uses the duration-free iceberg ratio, so magnitudes and signs "
                    "must not be treated as a direct replication comparison."
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

    model_labels: tuple[str, str, str] = (
        "Ratio baseline\n(no duration denominator)",
        "+ duration",
        "+ gap & overlap",
    )
    term_labels: dict[str, str] = {
        "agree_move": "Agreement movement",
        "disagree_move": "Disagreement movement",
    }
    colors: dict[str, str] = {
        "agree_move": "#0072B2",
        "disagree_move": "#D55E00",
    }
    markers: dict[str, str] = {
        "agree_move": "o",
        "disagree_move": "D",
    }
    selected_by_term: dict[str, pd.DataFrame] = {
        term: comparison[comparison["term"] == term]
        .set_index("model_name")
        .loc[list(HEADLINE_MODELS)]
        for term in STANCE_TERMS
    }
    x_positions: np.ndarray = np.arange(len(HEADLINE_MODELS), dtype=float)

    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    ):
        fig, path_axis = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)

        for term in STANCE_TERMS:
            selected = selected_by_term[term]
            coefficients = selected["coefficient"].to_numpy(dtype=float)
            lows = selected["ci95_low"].to_numpy(dtype=float)
            highs = selected["ci95_high"].to_numpy(dtype=float)
            path_axis.plot(
                x_positions,
                coefficients,
                color=colors[term],
                linewidth=2.0,
                zorder=2,
            )
            path_axis.errorbar(
                x_positions,
                coefficients,
                yerr=np.vstack((coefficients - lows, highs - coefficients)),
                fmt=markers[term],
                color=colors[term],
                ecolor=colors[term],
                capsize=3,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=1.5,
                zorder=3,
                label=term_labels[term],
            )

        path_axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
        path_axis.set_xticks(x_positions, model_labels)
        path_axis.set_ylabel("Coefficient (episode-clustered 95% CI)")
        path_axis.set_title("Estimates across specifications", loc="left", fontweight="bold")
        path_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        path_axis.legend(frameon=False, loc="lower right")
        fig.suptitle(
            "Duration-free iceberg-ratio model with timing sensitivity checks",
            fontsize=15,
            fontweight="bold",
        )

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


def save_stepwise_plot(
    comparison: pd.DataFrame,
    output_dir: Path,
    plot_dpi: int,
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "RQ1 plotting requires matplotlib. Install the dependencies declared in pyproject.toml."
        ) from error

    term_labels: dict[str, str] = {
        "agree_move": "Agreement movement",
        "disagree_move": "Disagreement movement",
    }
    colors: dict[str, str] = {
        "agree_move": "#0072B2",
        "disagree_move": "#D55E00",
    }
    markers: dict[str, str] = {
        "agree_move": "o",
        "disagree_move": "D",
    }
    stage_labels = tuple(
        (
            f"{stage_number}. Stance movement only"
            if stage_number == 1
            else f"{stage_number}. + {STEPWISE_ADDED_GROUPS[model_name]} (Exp2 controls)"
            if model_name == RATIO_MODEL
            else f"{stage_number}. + {STEPWISE_ADDED_GROUPS[model_name]}"
        )
        for stage_number, model_name in enumerate(STEPWISE_MODEL_ORDER, start=1)
    )
    y_positions = np.arange(len(STEPWISE_MODEL_ORDER), dtype=float)

    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.2, 6.6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for axis, term in zip(axes, STANCE_TERMS, strict=True):
            selected = (
                comparison[comparison["term"] == term]
                .set_index("model_name")
                .loc[list(STEPWISE_MODEL_ORDER)]
            )
            coefficients = selected["coefficient"].to_numpy(dtype=float)
            lows = selected["ci95_low"].to_numpy(dtype=float)
            highs = selected["ci95_high"].to_numpy(dtype=float)
            largest = selected[selected["largest_incremental_change_for_term"].astype(bool)]
            if len(largest) != 1:
                raise ValueError(
                    f"Expected one largest stepwise coefficient change for {term}; found {len(largest)}"
                )
            largest_stage = int(largest.iloc[0]["stage_number"])
            largest_delta = float(largest.iloc[0]["coefficient_change_from_previous"])

            axis.axhspan(
                largest_stage - 1.45,
                largest_stage - 0.55,
                color=colors[term],
                alpha=0.08,
                zorder=0,
            )
            axis.plot(
                coefficients,
                y_positions,
                color=colors[term],
                linewidth=1.8,
                zorder=2,
            )
            axis.errorbar(
                coefficients,
                y_positions,
                xerr=np.vstack((coefficients - lows, highs - coefficients)),
                fmt=markers[term],
                color=colors[term],
                ecolor=colors[term],
                capsize=3,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=1.4,
                zorder=3,
            )
            axis.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
            axis.set_title(
                f"{term_labels[term]}\nLargest shift: stage {largest_stage} "
                f"(Δβ={largest_delta:+.3f})",
                loc="left",
                fontweight="bold",
            )
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)

        axes[0].set_yticks(y_positions, stage_labels)
        axes[0].invert_yaxis()
        fig.supxlabel("Stance coefficient (episode-clustered 95% CI)")
        fig.suptitle(
            "Experiment 2: step-wise stance coefficient stability",
            fontsize=15,
            fontweight="bold",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "rq1_stepwise_comparison.png"
        pdf_path = output_dir / "rq1_stepwise_comparison.pdf"
        png_temporary = png_path.with_suffix(".png.tmp")
        pdf_temporary = pdf_path.with_suffix(".pdf.tmp")
        fig.savefig(png_temporary, format="png", dpi=plot_dpi, bbox_inches="tight")
        fig.savefig(pdf_temporary, format="pdf", bbox_inches="tight")
        plt.close(fig)
    png_temporary.replace(png_path)
    pdf_temporary.replace(pdf_path)
    return [pdf_path, png_path]


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def coefficient_cell_text(coefficient: float, clustered_se: float, p_value: float) -> str:
    displayed_coefficient = 0.0 if abs(coefficient) < 0.00005 else coefficient
    return (
        f"{displayed_coefficient:.4f}{significance_stars(p_value)}\n"
        f"({clustered_se:.4f})"
    )


def save_specification_panel(
    coefficients: pd.DataFrame,
    panel: pd.DataFrame,
    output_dir: Path,
    plot_dpi: int,
    plot_config: SpecificationPlotConfig,
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "RQ1 plotting requires matplotlib. Install the dependencies declared in pyproject.toml."
        ) from error

    ordered_panel = (
        panel.set_index("model_name")
        .loc[list(SPECIFICATION_MODEL_ORDER)]
        .reset_index()
    )
    if ordered_panel["model_number"].tolist() != list(
        range(1, len(SPECIFICATION_MODEL_ORDER) + 1)
    ):
        raise ValueError("Specification panel model numbers do not match the configured order")
    if coefficients.duplicated(["model_name", "term"]).any():
        raise ValueError("Specification coefficient table contains duplicate model-term rows")

    category_terms = tuple(
        sorted(
            term
            for term in coefficients["term"].astype(str).unique()
            if term.startswith("C(category)[T.")
        )
    )
    shared_terms = (
        plot_config["previous_outcome_term"],
        "timeline_position",
        "I(timeline_position ** 2)",
        *category_terms,
        "log_duration",
        "log_gap",
        "overlap",
        "previous_log_duration",
        "Intercept",
    )
    agreement_terms = ("agree_move", "lag_agree_move", *shared_terms)
    disagreement_terms = ("disagree_move", "lag_disagree_move", *shared_terms)
    term_labels: dict[str, str] = {
        "agree_move": "Current agreement movement",
        "lag_agree_move": "Lagged agreement movement",
        "disagree_move": "Current disagreement movement",
        "lag_disagree_move": "Lagged disagreement movement",
        plot_config["previous_outcome_term"]: plot_config["previous_outcome_label"],
        "timeline_position": "Timeline position",
        "I(timeline_position ** 2)": "Timeline position²",
        "log_duration": "Current-turn duration (log)",
        "log_gap": "Pre-turn gap (log)",
        "overlap": "Overlap",
        "previous_log_duration": "Previous-turn duration (log)",
        "Intercept": "Intercept",
    }
    term_labels.update(
        {
            term: f"Category: {term.removeprefix('C(category)[T.').removesuffix(']')}"
            for term in category_terms
        }
    )
    statistic_labels = (
        "N",
        "Episode clusters",
        "Adjusted R²",
        plot_config["estimated_change_label"],
    )
    x_positions = np.arange(len(SPECIFICATION_MODEL_ORDER), dtype=float)
    coefficient_lookup = coefficients.set_index(["model_name", "term"])
    compact_labels = (
        "Core",
        "+ Lag",
        "+ Prev. outcome",
        "+ Timeline",
        "+ Category FE",
        "+ Current dur.",
        "+ Gap/overlap",
        "+ Previous dur.",
        "Exp2 controls",
        "Full timing",
        "− Lag",
        "− Prev. outcome",
        "− Timeline",
        "− Category FE",
        "− Current dur.",
        "− Gap/overlap",
        "− Previous dur.",
    )

    with plt.rc_context(
        {
            "axes.spines.top": True,
            "axes.spines.right": False,
            "font.size": 9,
        }
    ):
        row_count = len(agreement_terms) + len(statistic_labels)
        figure_width = max(19.0, len(SPECIFICATION_MODEL_ORDER) * 0.95 + 4.0)
        figure_height = max(20.0, row_count * 1.15)
        fig, axes = plt.subplots(2, 1, figsize=(figure_width, figure_height))
        fig.subplots_adjust(
            left=0.19,
            right=0.995,
            top=0.91,
            bottom=0.055,
            hspace=0.42,
        )

        panel_definitions = (
            ("Agreement movement", "agree_move", agreement_terms),
            ("Disagreement movement", "disagree_move", disagreement_terms),
        )
        for axis, (direction_label, current_stance_term, terms) in zip(
            axes,
            panel_definitions,
            strict=True,
        ):
            all_row_labels = [term_labels[term] for term in terms] + list(statistic_labels)
            statistic_start = len(terms)
            for row_number in range(row_count):
                if row_number % 2 == 1:
                    axis.axhspan(
                        row_number - 0.5,
                        row_number + 0.5,
                        color="#777777",
                        alpha=0.035,
                        zorder=0,
                    )
                axis.axhline(
                    row_number + 0.5,
                    color="#E1E1E1",
                    linewidth=0.55,
                    zorder=1,
                )
            axis.axhline(
                statistic_start - 0.5,
                color="#333333",
                linewidth=1.1,
                zorder=2,
            )
            axis.axhline(
                row_count - 1.5,
                color="#333333",
                linewidth=1.1,
                zorder=2,
            )
            for divider in (0.5, 7.5, 9.5):
                axis.axvline(divider, color="#A0A0A0", linewidth=0.8, zorder=1)

            for model_position, model_name in enumerate(SPECIFICATION_MODEL_ORDER):
                for row_number, term in enumerate(terms):
                    key = (model_name, term)
                    if key not in coefficient_lookup.index:
                        continue
                    selected = coefficient_lookup.loc[key]
                    axis.text(
                        model_position,
                        row_number,
                        coefficient_cell_text(
                            float(selected["coefficient"]),
                            float(selected["clustered_se"]),
                            float(selected["p_value"]),
                        ),
                        ha="center",
                        va="center",
                        fontsize=6.7,
                        linespacing=1.2,
                        zorder=3,
                    )
                model_row = ordered_panel.iloc[model_position]
                statistic_values = (
                    f"{int(model_row['transition_count']):,}",
                    f"{int(model_row['episode_count']):,}",
                    f"{float(model_row['adjusted_r_squared']):.3f}",
                    f"{float(model_row[f'{current_stance_term}_estimated_change_percent']):+.2f}%",
                )
                for statistic_offset, statistic_value in enumerate(statistic_values):
                    axis.text(
                        model_position,
                        statistic_start + statistic_offset,
                        statistic_value,
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        zorder=3,
                    )

            axis.set_xlim(-0.5, len(SPECIFICATION_MODEL_ORDER) - 0.5)
            axis.set_ylim(row_count - 0.5, -0.5)
            axis.set_yticks(np.arange(row_count, dtype=float), all_row_labels)
            axis.set_xticks(
                x_positions,
                [
                    f"({model_number})\n{compact_label}"
                    for model_number, compact_label in enumerate(compact_labels, start=1)
                ],
                rotation=38,
                ha="left",
            )
            axis.xaxis.tick_top()
            axis.tick_params(axis="x", length=0, pad=5, labelsize=7.5)
            axis.tick_params(axis="y", length=0, labelsize=8.2)
            axis.set_title(direction_label, loc="left", pad=84, fontweight="bold")
            axis.spines["bottom"].set_visible(False)

        fig.suptitle(
            plot_config["figure_title"],
            fontsize=16,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.018,
            (
                f"Cells report β for {plot_config['response_label']}, with "
                "episode-clustered SE in parentheses. Both stance directions are jointly "
                "estimated. Bottom-row change = 100 × [exp(β stance) − 1]. "
                "* p<.05, ** p<.01, *** p<.001."
            ),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / f"{plot_config['output_stem']}.png"
        pdf_path = output_dir / f"{plot_config['output_stem']}.pdf"
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
        "stepwise_coefficients": output_dir / "rq1_stepwise_coefficients.csv",
        "stepwise_stance_comparison": output_dir / "rq1_stepwise_stance_comparison.csv",
        "stepwise_model_fit": output_dir / "rq1_stepwise_model_fit.csv",
        "specification_panel": output_dir / "rq1_specification_panel.csv",
        "specification_coefficients": output_dir / "rq1_specification_coefficients.csv",
        "implicit_assumption_specification_panel": (
            output_dir / "rq1_specification_panel_implicit_assumptions.csv"
        ),
        "implicit_assumption_specification_coefficients": (
            output_dir / "rq1_specification_coefficients_implicit_assumptions.csv"
        ),
        "explicit_claim_specification_panel": (
            output_dir / "rq1_specification_panel_explicit_claims.csv"
        ),
        "explicit_claim_specification_coefficients": (
            output_dir / "rq1_specification_coefficients_explicit_claims.csv"
        ),
        "observations": output_dir / "rq1_timing_observations.csv",
        "data_audit": output_dir / "rq1_timing_data_audit.json",
        "summary": output_dir / "rq1_timing_summary.json",
        "specification_plot_pdf": output_dir / "rq1_specification_panel.pdf",
        "specification_plot_png": output_dir / "rq1_specification_panel.png",
        "implicit_assumption_specification_plot_pdf": (
            output_dir / "rq1_specification_panel_implicit_assumptions.pdf"
        ),
        "implicit_assumption_specification_plot_png": (
            output_dir / "rq1_specification_panel_implicit_assumptions.png"
        ),
        "explicit_claim_specification_plot_pdf": (
            output_dir / "rq1_specification_panel_explicit_claims.pdf"
        ),
        "explicit_claim_specification_plot_png": (
            output_dir / "rq1_specification_panel_explicit_claims.png"
        ),
    }


def remove_obsolete_comparison_plots(output_dir: Path) -> None:
    obsolete_paths = (
        output_dir / "rq1_timing_comparison.pdf",
        output_dir / "rq1_timing_comparison.png",
        output_dir / "rq1_stepwise_comparison.pdf",
        output_dir / "rq1_stepwise_comparison.png",
    )
    for obsolete_path in obsolete_paths:
        if obsolete_path.exists():
            obsolete_path.unlink()


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
    stepwise_results = fit_stepwise_models(frame)
    results = {model_name: stepwise_results[model_name] for model_name in MODEL_ORDER}
    coefficients = coefficient_frame(results)
    model_fit = model_fit_frame(results, frame)
    comparison = stance_comparison_frame(coefficients, frame)
    stepwise_coefficients = stepwise_coefficient_frame(stepwise_results)
    stepwise_model_fit = stepwise_model_fit_frame(stepwise_results, frame)
    stepwise_comparison = stepwise_stance_comparison_frame(
        stepwise_coefficients,
        frame,
    )
    specification_results = fit_specification_models(frame, stepwise_results)
    specification_coefficients = specification_coefficient_frame(specification_results)
    specification_panel = specification_panel_frame(
        specification_results,
        frame,
        SPECIFICATION_FORMULAS,
    )
    implicit_assumption_specification_results = fit_formula_models(
        frame,
        IMPLICIT_ASSUMPTION_SPECIFICATION_FORMULAS,
        SPECIFICATION_MODEL_ORDER,
    )
    implicit_assumption_specification_coefficients = specification_coefficient_frame(
        implicit_assumption_specification_results
    )
    implicit_assumption_specification_panel = specification_panel_frame(
        implicit_assumption_specification_results,
        frame,
        IMPLICIT_ASSUMPTION_SPECIFICATION_FORMULAS,
    )
    explicit_claim_specification_results = fit_formula_models(
        frame,
        EXPLICIT_CLAIM_SPECIFICATION_FORMULAS,
        SPECIFICATION_MODEL_ORDER,
    )
    explicit_claim_specification_coefficients = specification_coefficient_frame(
        explicit_claim_specification_results
    )
    explicit_claim_specification_panel = specification_panel_frame(
        explicit_claim_specification_results,
        frame,
        EXPLICIT_CLAIM_SPECIFICATION_FORMULAS,
    )
    output_dir = Path(args.output_dir)
    remove_obsolete_comparison_plots(output_dir)
    paths = output_paths(output_dir)

    write_csv(paths["observations"], frame)
    write_csv(paths["coefficients"], coefficients)
    write_csv(paths["stance_comparison"], comparison)
    write_csv(paths["model_fit"], model_fit)
    write_csv(paths["stepwise_coefficients"], stepwise_coefficients)
    write_csv(paths["stepwise_stance_comparison"], stepwise_comparison)
    write_csv(paths["stepwise_model_fit"], stepwise_model_fit)
    write_csv(paths["specification_panel"], specification_panel)
    write_csv(paths["specification_coefficients"], specification_coefficients)
    write_csv(
        paths["implicit_assumption_specification_panel"],
        implicit_assumption_specification_panel,
    )
    write_csv(
        paths["implicit_assumption_specification_coefficients"],
        implicit_assumption_specification_coefficients,
    )
    write_csv(
        paths["explicit_claim_specification_panel"],
        explicit_claim_specification_panel,
    )
    write_csv(
        paths["explicit_claim_specification_coefficients"],
        explicit_claim_specification_coefficients,
    )
    save_specification_panel(
        specification_coefficients,
        specification_panel,
        output_dir,
        args.plot_dpi,
        RATIO_SPECIFICATION_PLOT,
    )
    save_specification_panel(
        implicit_assumption_specification_coefficients,
        implicit_assumption_specification_panel,
        output_dir,
        args.plot_dpi,
        IMPLICIT_ASSUMPTION_SPECIFICATION_PLOT,
    )
    save_specification_panel(
        explicit_claim_specification_coefficients,
        explicit_claim_specification_panel,
        output_dir,
        args.plot_dpi,
        EXPLICIT_CLAIM_SPECIFICATION_PLOT,
    )

    repository_state = git_state()
    versions = package_versions()
    audit.update(
        {
            "experiment": "Duration-free iceberg-ratio timing sensitivity analysis",
            "script_version": SCRIPT_VERSION,
            "script_sha256": file_sha256(Path(__file__)),
            "formulas": FORMULAS,
            "stepwise_formulas": STEPWISE_FORMULAS,
            "stepwise_added_variable_groups": STEPWISE_ADDED_GROUPS,
            "specification_formulas": SPECIFICATION_FORMULAS,
            "implicit_assumption_specification_formulas": (
                IMPLICIT_ASSUMPTION_SPECIFICATION_FORMULAS
            ),
            "explicit_claim_specification_formulas": (
                EXPLICIT_CLAIM_SPECIFICATION_FORMULAS
            ),
            "specification_variable_groups": VARIABLE_GROUP_TERMS,
            "specification_previous_outcome_terms": {
                "iceberg_ratio": "previous_log_iceberg_ratio",
                "implicit_assumptions": "previous_log_assumption_count",
                "explicit_claims": "previous_log_explicit_count",
            },
            "specification_included_groups": SPECIFICATION_INCLUDED_GROUPS,
            "specification_reference_models": SPECIFICATION_REFERENCE_MODELS,
            "common_model_sample_verified": True,
            "stepwise_common_model_sample_verified": True,
            "specification_common_model_sample_verified": True,
            "implicit_assumption_specification_common_model_sample_verified": True,
            "explicit_claim_specification_common_model_sample_verified": True,
            "word_count_definition": r"count of Unicode regex matches for \b\w+\b in turn_text",
            "iceberg_ratio_definition": "explicit_count / (assumption_count + 1)",
            "response_variable_definition": (
                "delta_log_iceberg_ratio = log1p(current_iceberg_ratio) - "
                "log1p(previous_iceberg_ratio); duration is not in the outcome"
            ),
            "alternative_response_variable_definitions": {
                "delta_log_assumption_count": (
                    "log1p(current_assumption_count) - "
                    "log1p(previous_assumption_count)"
                ),
                "delta_log_explicit_count": (
                    "log1p(current_explicit_count) - log1p(previous_explicit_count)"
                ),
            },
            "iceberg_density_per_second_definition": (
                "(explicit_count / (assumption_count + 1)) / duration_seconds"
            ),
            "specification_estimated_change_percent_definition": (
                "100 * (exp(current_stance_coefficient) - 1)"
            ),
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
        "experiment": "Duration-free iceberg-ratio timing sensitivity analysis",
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
        "stepwise_formulas": STEPWISE_FORMULAS,
        "stepwise_added_variable_groups": STEPWISE_ADDED_GROUPS,
        "specification_formulas": SPECIFICATION_FORMULAS,
        "implicit_assumption_specification_formulas": (
            IMPLICIT_ASSUMPTION_SPECIFICATION_FORMULAS
        ),
        "explicit_claim_specification_formulas": EXPLICIT_CLAIM_SPECIFICATION_FORMULAS,
        "specification_variable_groups": VARIABLE_GROUP_TERMS,
        "specification_previous_outcome_terms": {
            "iceberg_ratio": "previous_log_iceberg_ratio",
            "implicit_assumptions": "previous_log_assumption_count",
            "explicit_claims": "previous_log_explicit_count",
        },
        "specification_included_groups": SPECIFICATION_INCLUDED_GROUPS,
        "specification_reference_models": SPECIFICATION_REFERENCE_MODELS,
        "specification_model_count": len(SPECIFICATION_MODEL_ORDER),
        "specification_estimated_change_percent_definition": (
            "100 * (exp(current_stance_coefficient) - 1)"
        ),
        "specification_fit_metric": (
            "adjusted R-squared; preferred over raw R-squared because model sizes differ"
        ),
        "best_adjusted_r_squared_specification": {
            "model_name": str(
                specification_panel.loc[
                    specification_panel["adjusted_r_squared"].idxmax(),
                    "model_name",
                ]
            ),
            "adjusted_r_squared": float(specification_panel["adjusted_r_squared"].max()),
        },
        "alternative_specification_outcomes": {
            "implicit_assumptions": {
                "response_variable": "delta_log_assumption_count",
                "best_model_name": str(
                    implicit_assumption_specification_panel.loc[
                        implicit_assumption_specification_panel[
                            "adjusted_r_squared"
                        ].idxmax(),
                        "model_name",
                    ]
                ),
                "best_adjusted_r_squared": float(
                    implicit_assumption_specification_panel["adjusted_r_squared"].max()
                ),
            },
            "explicit_claims": {
                "response_variable": "delta_log_explicit_count",
                "best_model_name": str(
                    explicit_claim_specification_panel.loc[
                        explicit_claim_specification_panel[
                            "adjusted_r_squared"
                        ].idxmax(),
                        "model_name",
                    ]
                ),
                "best_adjusted_r_squared": float(
                    explicit_claim_specification_panel["adjusted_r_squared"].max()
                ),
            },
        },
        "transition_count": len(frame),
        "episode_count": int(frame["episode"].nunique()),
        "headline_coefficients": {
            f"{model_name}:{term}": extract_coefficient(coefficients, model_name, term)
            for model_name in HEADLINE_MODELS
            for term in STANCE_TERMS
        },
        "stepwise_largest_coefficient_changes": stepwise_largest_changes(
            stepwise_comparison
        ),
        "timing_adjustment_interpretation": interpretations,
        "original_exp2_reference_comparison": exp2_reference_comparison(coefficients),
        "both_directional_intervals_survive_timing_adjustment": interval_survives,
        "estimands": {
            RATIO_MODEL: (
                "directional association with the duration-free iceberg ratio using the "
                "original Exp2 control structure on the strict timing sample"
            ),
            DURATION_MODEL: (
                "duration-free iceberg-ratio association conditional on current-turn duration"
            ),
            TIMING_MODEL: (
                "duration-free iceberg-ratio association conditional on current-turn duration "
                "and response timing"
            ),
            PREVIOUS_DURATION_MODEL: (
                "duration-free iceberg-ratio association conditional on current- and "
                "previous-turn duration and response timing"
            ),
        },
        "causal_claim": False,
        "duration_note": (
            "Duration is excluded from the outcome denominator and enters only as an explicit "
            "predictor in duration-adjusted specifications."
        ),
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
        description="Run the duration-free iceberg-ratio timing sensitivity analysis."
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
