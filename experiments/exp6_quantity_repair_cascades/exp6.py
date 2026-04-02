import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = "data/conversation_moves_labeled"
DEFAULT_OUTPUT_DIR = "experiments/exp6_quantity_repair_cascades/results"
DEFAULT_EVENT_WINDOW_SEC = 180.0
DEFAULT_MAX_FOLLOWUP_TURNS = 8
DEFAULT_STABILITY_GAP_SEC = 45.0
DEFAULT_UNDER_ASSUMPTION_QUANTILE = 0.75
DEFAULT_UNDER_DURATION_QUANTILE = 0.25
DEFAULT_OVER_ASSUMPTION_QUANTILE = 0.25
DEFAULT_OVER_DURATION_QUANTILE = 0.75
DEFAULT_SAMPLE_CASCADE_COUNT = 18
DEFAULT_TEXT_PREVIEW_CHARS = 180
MIN_FOLLOWUP_OFFSET_SEC = 1e-3
RNG_SEED = 42

TRIGGER_ORDER = ("under_info", "over_info")
TRIGGER_LABELS = {
    "under_info": "Under-info",
    "over_info": "Over-info",
}
TRIGGER_COLORS = {
    "under_info": "#0f4c5c",
    "over_info": "#bc6c25",
}
ANSWER_RELEVANT_CURRENT_MOVES = {
    "Answer",
    "Agree / Align",
    "Stonewalling / Non-Response",
    "Correction / Challenge",
}
CLARIFICATION_MOVES = {
    "Clarification Request (Generic)",
    "Clarification Request (Specific)",
}
CHALLENGE_MOVES = {
    "Correction / Challenge",
    "Self-Correction",
}
CONTINUATION_MOVES = {
    "Assert / Elaborate",
    "Answer",
}
EXIT_MOVES = {
    "Topic Shift",
    "Stonewalling / Non-Response",
}
EVENT_COLORS = {
    "repair_event": "#1f77b4",
    "challenge_event": "#d62728",
    "continuation_event": "#6a994e",
    "support_marker": "#7f7f7f",
    "exit_marker": "#2a9d8f",
}
EVENT_MARKERS = {
    "repair_event": "o",
    "challenge_event": "s",
    "continuation_event": "^",
    "support_marker": "x",
    "exit_marker": "D",
}
INTRO_PREFIXES = (
    "welcome to",
    "you are listening to",
    "you're listening to",
    "i'm your host",
    "im your host",
    "hello and welcome",
    "today on",
    "thanks for joining",
    "in this episode",
    "this episode of",
)


def safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_int(value: object, fallback: int) -> int:
    try:
        if value is None:
            return fallback
        return int(float(value))
    except Exception:
        return fallback


def finite_or_none(value: object) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def clamp_followup_offset(offset_sec: float) -> float:
    if not math.isfinite(offset_sec):
        raise ValueError(f"Non-finite follow-up offset: {offset_sec}")
    return max(offset_sec, MIN_FOLLOWUP_OFFSET_SEC)


def normalize_text_preview(text: object, max_chars: int) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def sanitize_json_payload(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize_json_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_payload(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def sort_turns(turns: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed_turns: List[Tuple[int, int, Dict[str, Any]]] = []
    for position, turn in enumerate(turns):
        indexed_turns.append((safe_int(turn.get("turn_idx"), position), position, turn))
    indexed_turns.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed_turns]


def load_episode(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    return sort_turns(payload)


def turn_text(turn: Dict[str, Any]) -> str:
    return str(turn.get("turn_text") or turn.get("transcript") or "").strip()


def turn_start_time(turn: Dict[str, Any]) -> float:
    return safe_float(turn.get("startTime", turn.get("start_time")))


def turn_end_time(turn: Dict[str, Any]) -> float:
    return safe_float(turn.get("endTime", turn.get("end_time")))


def turn_duration(turn: Dict[str, Any]) -> float:
    explicit_duration = safe_float(turn.get("duration"))
    if math.isfinite(explicit_duration) and explicit_duration >= 0.0:
        return explicit_duration
    start_time = turn_start_time(turn)
    end_time = turn_end_time(turn)
    if math.isfinite(start_time) and math.isfinite(end_time) and end_time >= start_time:
        return end_time - start_time
    return float("nan")


def turn_type(turn: Dict[str, Any]) -> str:
    return str(turn.get("turn_type_label") or "").strip()


def move_label(turn: Dict[str, Any]) -> str:
    return str(turn.get("conversation_move_label") or "").strip()


def assumption_count(turn: Dict[str, Any]) -> int:
    assumptions = turn.get("assumptions")
    return len(assumptions) if isinstance(assumptions, list) else 0


def turn_word_count(turn: Dict[str, Any]) -> int:
    explicit_word_count = safe_float(turn.get("wordCount", turn.get("word_count")))
    if math.isfinite(explicit_word_count) and explicit_word_count >= 0.0:
        return int(round(explicit_word_count))
    text = turn_text(turn)
    return len(text.split()) if text else 0


def previous_turn_invites_response(previous_turn: Optional[Dict[str, Any]]) -> bool:
    if previous_turn is None:
        return False
    previous_move = move_label(previous_turn)
    previous_text = turn_text(previous_turn)
    if "?" in previous_text:
        return True
    return previous_move in CLARIFICATION_MOVES or previous_move in CHALLENGE_MOVES


def looks_like_intro_monologue(turn: Dict[str, Any], trigger_position: int) -> bool:
    if trigger_position <= 0:
        return True
    current_text = turn_text(turn).lower()
    return any(current_text.startswith(prefix) for prefix in INTRO_PREFIXES)


def passes_trigger_context_gate(
    trigger_type: str,
    current_turn: Dict[str, Any],
    previous_turn: Optional[Dict[str, Any]],
    trigger_position: int,
) -> bool:
    if trigger_position <= 0:
        return False

    previous_invites_response = previous_turn_invites_response(previous_turn)
    current_move = move_label(current_turn)
    current_word_count = turn_word_count(current_turn)

    if trigger_type == "under_info":
        if not previous_invites_response:
            return False
        if current_move in ANSWER_RELEVANT_CURRENT_MOVES:
            return True
        return current_move == "Assert / Elaborate" and current_word_count <= 80

    if trigger_type == "over_info":
        if not previous_invites_response:
            return False
        if looks_like_intro_monologue(current_turn, trigger_position):
            return False
        return current_move in {"Answer", "Assert / Elaborate", "Correction / Challenge", "Agree / Align"}

    return False


def collect_category_paths(
    input_dir: Path,
    categories: Sequence[str],
    max_episodes_per_category: int,
) -> List[Tuple[str, Path]]:
    category_paths: List[Tuple[str, Path]] = []
    available_category_dirs = sorted([path for path in input_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    if available_category_dirs:
        selected_categories = list(categories) if categories else [path.name for path in available_category_dirs]
        for category in selected_categories:
            category_dir = input_dir / category
            if not category_dir.exists():
                raise FileNotFoundError(f"Category directory not found: {category_dir}")
            episode_paths = sorted(category_dir.glob("*.json"))
            if max_episodes_per_category > 0:
                episode_paths = episode_paths[:max_episodes_per_category]
            category_paths.extend((category, episode_path) for episode_path in episode_paths)
        return category_paths

    if categories and len(categories) > 1:
        raise ValueError("Flat input directories support at most one explicit category.")

    inferred_category = categories[0] if categories else input_dir.name
    episode_paths = sorted(input_dir.glob("*.json"))
    if max_episodes_per_category > 0:
        episode_paths = episode_paths[:max_episodes_per_category]
    category_paths.extend((inferred_category, episode_path) for episode_path in episode_paths)
    return category_paths


def should_keep_trigger_candidate(turn: Dict[str, Any]) -> bool:
    if turn_type(turn) != "Substantive":
        return False
    duration_sec = turn_duration(turn)
    trigger_end = turn_end_time(turn)
    return bool(math.isfinite(duration_sec) and duration_sec >= 0.0 and math.isfinite(trigger_end))


def collect_trigger_metric_rows(
    episode_paths: Sequence[Tuple[str, Path]],
    show_progress: bool,
) -> pd.DataFrame:
    metric_rows: List[Dict[str, Any]] = []
    for category, episode_path in tqdm(episode_paths, desc="Collecting trigger metrics", unit="file", disable=not show_progress):
        turns = load_episode(episode_path)
        if not turns:
            continue
        episode_id = str(turns[0].get("episode_id") or episode_path.stem)
        for position, turn in enumerate(turns):
            if not should_keep_trigger_candidate(turn):
                continue
            metric_rows.append(
                {
                    "category": category,
                    "episode_id": episode_id,
                    "turn_idx": safe_int(turn.get("turn_idx"), position),
                    "assumption_count": int(assumption_count(turn)),
                    "duration_sec": float(turn_duration(turn)),
                }
            )
    return pd.DataFrame(metric_rows)


def build_threshold_table(
    metric_df: pd.DataFrame,
    under_assumption_quantile: float,
    under_duration_quantile: float,
    over_assumption_quantile: float,
    over_duration_quantile: float,
) -> pd.DataFrame:
    if metric_df.empty:
        raise RuntimeError("No substantive trigger candidates were found in the input data.")

    rows: List[Dict[str, Any]] = []
    for category, category_df in metric_df.groupby("category", observed=False):
        assumption_values = category_df["assumption_count"].astype(float).to_numpy()
        duration_values = category_df["duration_sec"].astype(float).to_numpy()
        rows.append(
            {
                "category": str(category),
                "n_candidate_turns": int(len(category_df)),
                "under_assumption_threshold": float(np.quantile(assumption_values, under_assumption_quantile)),
                "under_duration_threshold": float(np.quantile(duration_values, under_duration_quantile)),
                "over_assumption_threshold": float(np.quantile(assumption_values, over_assumption_quantile)),
                "over_duration_threshold": float(np.quantile(duration_values, over_duration_quantile)),
            }
        )
    return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)


def build_threshold_lookup(threshold_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in threshold_df.iterrows():
        lookup[str(row["category"])] = {
            "under_assumption_threshold": float(row["under_assumption_threshold"]),
            "under_duration_threshold": float(row["under_duration_threshold"]),
            "over_assumption_threshold": float(row["over_assumption_threshold"]),
            "over_duration_threshold": float(row["over_duration_threshold"]),
        }
    return lookup


def classify_trigger_type(
    assumption_value: int,
    duration_sec: float,
    category_thresholds: Dict[str, float],
    current_turn: Dict[str, Any],
    previous_turn: Optional[Dict[str, Any]],
    trigger_position: int,
) -> Optional[str]:
    under_trigger = (
        assumption_value >= category_thresholds["under_assumption_threshold"]
        and duration_sec <= category_thresholds["under_duration_threshold"]
    )
    over_trigger = (
        assumption_value <= category_thresholds["over_assumption_threshold"]
        and duration_sec >= category_thresholds["over_duration_threshold"]
    )
    if under_trigger and over_trigger:
        return None
    if under_trigger and passes_trigger_context_gate("under_info", current_turn, previous_turn, trigger_position):
        return "under_info"
    if over_trigger and passes_trigger_context_gate("over_info", current_turn, previous_turn, trigger_position):
        return "over_info"
    return None


def immediate_next_turn_latency(turns: Sequence[Dict[str, Any]], trigger_position: int, trigger_end: float) -> float:
    if trigger_position + 1 >= len(turns):
        return float("nan")
    next_start = turn_start_time(turns[trigger_position + 1])
    if not math.isfinite(next_start):
        return float("nan")
    effective_next_start = max(next_start, trigger_end)
    latency = effective_next_start - trigger_end
    return latency if latency >= 0.0 else float("nan")


def classify_followup_event(turn: Dict[str, Any]) -> Optional[str]:
    current_move = move_label(turn)
    current_type = turn_type(turn)
    if current_move in EXIT_MOVES:
        return "exit_marker"
    if current_move in CHALLENGE_MOVES:
        return "challenge_event"
    if current_move in CLARIFICATION_MOVES:
        return "repair_event"
    if current_move in CONTINUATION_MOVES:
        return "continuation_event"
    if current_type == "Backchannel":
        return "support_marker"
    return None


def extract_trigger_and_cascade_rows(
    episode_paths: Sequence[Tuple[str, Path]],
    threshold_lookup: Dict[str, Dict[str, float]],
    event_window_sec: float,
    max_followup_turns: int,
    stability_gap_sec: float,
    text_preview_chars: int,
    show_progress: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trigger_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    trigger_columns = [
        "category",
        "episode_id",
        "trigger_turn_idx",
        "trigger_type",
        "assumption_count",
        "duration_sec",
        "next_turn_latency_sec",
        "repair_event_count",
        "challenge_event_count",
        "continuation_event_count",
        "support_marker_count",
        "topic_shift_indicator",
        "silence_void_outcome",
        "cascade_duration_sec",
        "observation_horizon_sec",
        "followup_turns_observed",
        "exit_move",
        "trigger_text",
    ]
    event_columns = [
        "category",
        "episode_id",
        "trigger_turn_idx",
        "trigger_type",
        "followup_turn_idx",
        "event_type",
        "move_label",
        "turn_type_label",
        "relative_start_sec",
        "relative_end_sec",
        "gap_from_previous_sec",
        "turn_text",
    ]

    for category, episode_path in tqdm(episode_paths, desc="Extracting cascades", unit="file", disable=not show_progress):
        turns = load_episode(episode_path)
        if not turns:
            continue
        if category not in threshold_lookup:
            raise KeyError(f"Missing thresholds for category: {category}")

        episode_id = str(turns[0].get("episode_id") or episode_path.stem)
        category_thresholds = threshold_lookup[category]

        for trigger_position, trigger_turn in enumerate(turns):
            if not should_keep_trigger_candidate(trigger_turn):
                continue
            trigger_duration_sec = turn_duration(trigger_turn)
            trigger_assumptions = assumption_count(trigger_turn)
            previous_turn = turns[trigger_position - 1] if trigger_position > 0 else None
            trigger_type = classify_trigger_type(
                trigger_assumptions,
                trigger_duration_sec,
                category_thresholds,
                trigger_turn,
                previous_turn,
                trigger_position,
            )
            if trigger_type is None:
                continue

            trigger_turn_idx = safe_int(trigger_turn.get("turn_idx"), trigger_position)
            trigger_end = turn_end_time(trigger_turn)
            next_turn_latency_sec = immediate_next_turn_latency(turns, trigger_position, trigger_end)

            repair_event_count = 0
            challenge_event_count = 0
            continuation_event_count = 0
            support_marker_count = 0
            topic_shift_indicator = 0
            silence_void_outcome = 1
            exit_move = ""
            last_excited_event_time_sec = float("nan")
            observation_horizon_sec = 0.0
            followup_turns_observed = 0
            previous_end = trigger_end

            for offset in range(1, max_followup_turns + 1):
                followup_position = trigger_position + offset
                if followup_position >= len(turns):
                    break

                followup_turn = turns[followup_position]
                raw_followup_start = turn_start_time(followup_turn)
                if not math.isfinite(raw_followup_start):
                    break

                effective_followup_start = max(raw_followup_start, previous_end) if math.isfinite(previous_end) else raw_followup_start
                relative_start_sec_raw = effective_followup_start - trigger_end
                relative_start_sec = clamp_followup_offset(relative_start_sec_raw)
                if relative_start_sec_raw > event_window_sec:
                    observation_horizon_sec = max(observation_horizon_sec, event_window_sec)
                    break

                gap_from_previous_sec = (
                    effective_followup_start - previous_end if math.isfinite(previous_end) else float("nan")
                )
                if math.isfinite(gap_from_previous_sec) and gap_from_previous_sec > stability_gap_sec:
                    observation_horizon_sec = max(observation_horizon_sec, min(event_window_sec, max(relative_start_sec, 0.0)))
                    break

                followup_turns_observed += 1
                event_type = classify_followup_event(followup_turn)
                raw_followup_end = turn_end_time(followup_turn)
                followup_end = (
                    max(raw_followup_end, effective_followup_start) if math.isfinite(raw_followup_end) else effective_followup_start
                )
                relative_end_raw = followup_end - trigger_end if math.isfinite(followup_end) else relative_start_sec
                relative_end_sec = max(relative_end_raw, relative_start_sec, MIN_FOLLOWUP_OFFSET_SEC)
                observation_horizon_sec = max(
                    observation_horizon_sec,
                    min(event_window_sec, max(relative_end_sec, relative_start_sec, MIN_FOLLOWUP_OFFSET_SEC)),
                )

                if event_type == "repair_event":
                    repair_event_count += 1
                    silence_void_outcome = 0
                    last_excited_event_time_sec = relative_start_sec
                elif event_type == "challenge_event":
                    challenge_event_count += 1
                    silence_void_outcome = 0
                    last_excited_event_time_sec = relative_start_sec
                elif event_type == "continuation_event":
                    continuation_event_count += 1
                elif event_type == "support_marker":
                    support_marker_count += 1
                elif event_type == "exit_marker":
                    exit_move = move_label(followup_turn)
                    if exit_move == "Topic Shift":
                        topic_shift_indicator = 1

                if event_type is not None:
                    event_rows.append(
                        {
                            "category": category,
                            "episode_id": episode_id,
                            "trigger_turn_idx": trigger_turn_idx,
                            "trigger_type": trigger_type,
                            "followup_turn_idx": safe_int(followup_turn.get("turn_idx"), followup_position),
                            "event_type": event_type,
                            "move_label": move_label(followup_turn),
                            "turn_type_label": turn_type(followup_turn),
                            "relative_start_sec": float(relative_start_sec),
                            "relative_end_sec": float(relative_end_sec),
                            "gap_from_previous_sec": float(gap_from_previous_sec) if math.isfinite(gap_from_previous_sec) else math.nan,
                            "turn_text": normalize_text_preview(turn_text(followup_turn), text_preview_chars),
                        }
                    )

                previous_end = followup_end if math.isfinite(followup_end) else effective_followup_start
                if event_type == "exit_marker":
                    break

            if followup_turns_observed == 0 and math.isfinite(next_turn_latency_sec):
                observation_horizon_sec = min(event_window_sec, max(next_turn_latency_sec, 0.0))

            trigger_rows.append(
                {
                    "category": category,
                    "episode_id": episode_id,
                    "trigger_turn_idx": trigger_turn_idx,
                    "trigger_type": trigger_type,
                    "assumption_count": trigger_assumptions,
                    "duration_sec": float(trigger_duration_sec),
                    "next_turn_latency_sec": float(next_turn_latency_sec) if math.isfinite(next_turn_latency_sec) else math.nan,
                    "repair_event_count": int(repair_event_count),
                    "challenge_event_count": int(challenge_event_count),
                    "continuation_event_count": int(continuation_event_count),
                    "support_marker_count": int(support_marker_count),
                    "topic_shift_indicator": int(topic_shift_indicator),
                    "silence_void_outcome": int(silence_void_outcome),
                    "cascade_duration_sec": float(last_excited_event_time_sec) if math.isfinite(last_excited_event_time_sec) else 0.0,
                    "observation_horizon_sec": float(min(observation_horizon_sec, event_window_sec)),
                    "followup_turns_observed": int(followup_turns_observed),
                    "exit_move": exit_move,
                    "trigger_text": normalize_text_preview(turn_text(trigger_turn), text_preview_chars),
                }
            )

    trigger_df = pd.DataFrame(trigger_rows, columns=trigger_columns)
    event_df = pd.DataFrame(event_rows, columns=event_columns)
    return trigger_df, event_df


def compute_mean_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "mean": None,
            "std": None,
            "se": None,
            "ci_low": None,
            "ci_high": None,
        }
    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
    se_value = float(std_value / math.sqrt(finite_values.size)) if finite_values.size > 1 else 0.0
    margin = 1.96 * se_value
    return {
        "mean": mean_value,
        "std": std_value,
        "se": se_value,
        "ci_low": mean_value - margin,
        "ci_high": mean_value + margin,
    }


def prepare_hawkes_inputs(trigger_df: pd.DataFrame, event_df: pd.DataFrame, trigger_type: str) -> Tuple[List[np.ndarray], List[float]]:
    group_triggers = trigger_df[trigger_df["trigger_type"] == trigger_type].copy()
    group_events = event_df[
        (event_df["trigger_type"] == trigger_type)
        & (event_df["event_type"].isin(["repair_event", "challenge_event"]))
    ].copy()

    if group_triggers.empty:
        return [], []

    cascades: List[np.ndarray] = []
    horizons: List[float] = []
    grouped_events = group_events.groupby(["episode_id", "trigger_turn_idx"], observed=False)

    for _, trigger_row in group_triggers.iterrows():
        key = (str(trigger_row["episode_id"]), int(trigger_row["trigger_turn_idx"]))
        horizon = finite_or_none(trigger_row["observation_horizon_sec"])
        if horizon is None or horizon <= 0.0:
            horizon = finite_or_none(trigger_row["next_turn_latency_sec"]) or 0.0
        horizon = float(max(horizon, 0.0))

        if key in grouped_events.groups:
            repair_times = grouped_events.get_group(key)["relative_start_sec"].astype(float).to_numpy()
            repair_times = repair_times[np.isfinite(repair_times)]
            repair_times = repair_times[repair_times >= 0.0]
            if horizon > 0.0:
                repair_times = repair_times[repair_times <= horizon]
            repair_times = np.sort(repair_times)
        else:
            repair_times = np.asarray([], dtype=float)

        cascades.append(repair_times)
        horizons.append(horizon)

    return cascades, horizons


def hawkes_negative_log_likelihood(raw_params: np.ndarray, cascades: Sequence[np.ndarray], horizons: Sequence[float]) -> float:
    mu = float(np.exp(raw_params[0]))
    alpha = float(np.exp(raw_params[1]))
    beta = float(np.exp(raw_params[2]))

    total = 0.0
    for repair_times, horizon in zip(cascades, horizons):
        intensity_memory = 0.0
        previous_time = 0.0

        for event_time in repair_times:
            if event_time > horizon:
                continue
            elapsed = max(float(event_time) - previous_time, 0.0)
            decay_exponent = min(beta * elapsed, 700.0)
            intensity_memory *= math.exp(-decay_exponent)
            intensity = mu + alpha * intensity_memory
            total -= math.log(max(intensity, 1e-12))
            intensity_memory += 1.0
            previous_time = float(event_time)

        integral = mu * horizon
        if repair_times.size:
            integral += (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (horizon - repair_times[repair_times <= horizon]))))
        total += integral

    return float(total)


def fit_exponential_hawkes(cascades: Sequence[np.ndarray], horizons: Sequence[float], trigger_type: str) -> Dict[str, Any]:
    cascade_count = len(cascades)
    total_events = int(sum(len(repair_times) for repair_times in cascades))
    total_horizon = float(sum(horizons))

    if cascade_count == 0:
        return {
            "trigger_type": trigger_type,
            "n_cascades": 0,
            "n_events": 0,
            "mu": None,
            "alpha": None,
            "beta": None,
            "branching_ratio": None,
            "log_likelihood": None,
            "fit_status": "no_cascades",
        }

    if total_events < 5 or total_horizon <= 0.0:
        return {
            "trigger_type": trigger_type,
            "n_cascades": cascade_count,
            "n_events": total_events,
            "mu": None,
            "alpha": None,
            "beta": None,
            "branching_ratio": None,
            "log_likelihood": None,
            "fit_status": "too_sparse",
        }

    inter_event_gaps: List[float] = []
    for repair_times in cascades:
        if repair_times.size > 1:
            inter_event_gaps.extend(np.diff(repair_times).tolist())
    median_gap = float(np.median(inter_event_gaps)) if inter_event_gaps else 5.0
    event_rate = max(total_events / total_horizon, 1e-4)

    initial_params = np.log(
        np.asarray(
            [
                max(event_rate * 0.65, 1e-4),
                max(event_rate * 0.35, 1e-4),
                max(1.0 / max(median_gap, 1.0), 1e-4),
            ],
            dtype=float,
        )
    )

    optimization = minimize(
        fun=hawkes_negative_log_likelihood,
        x0=initial_params,
        args=(cascades, horizons),
        method="L-BFGS-B",
        bounds=[(-12.0, 6.0), (-12.0, 6.0), (-10.0, 6.0)],
    )

    if not optimization.success:
        return {
            "trigger_type": trigger_type,
            "n_cascades": cascade_count,
            "n_events": total_events,
            "mu": None,
            "alpha": None,
            "beta": None,
            "branching_ratio": None,
            "log_likelihood": None,
            "fit_status": "optimization_failed",
        }

    fitted_params = np.exp(np.asarray(optimization.x, dtype=float))
    mu = float(fitted_params[0])
    alpha = float(fitted_params[1])
    beta = float(fitted_params[2])
    return {
        "trigger_type": trigger_type,
        "n_cascades": cascade_count,
        "n_events": total_events,
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": float(alpha / beta) if beta > 0.0 else None,
        "log_likelihood": float(-optimization.fun),
        "fit_status": "ok",
    }


def build_hawkes_summary(trigger_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for trigger_type in TRIGGER_ORDER:
        cascades, horizons = prepare_hawkes_inputs(trigger_df, event_df, trigger_type)
        rows.append(fit_exponential_hawkes(cascades, horizons, trigger_type))
    return pd.DataFrame(rows)


def build_outcome_summary(trigger_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for trigger_type in TRIGGER_ORDER:
        group_df = trigger_df[trigger_df["trigger_type"] == trigger_type].copy()
        latency_summary = compute_mean_summary(group_df["next_turn_latency_sec"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        repair_summary = compute_mean_summary(group_df["repair_event_count"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        duration_summary = compute_mean_summary(group_df["cascade_duration_sec"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        topic_shift_summary = compute_mean_summary(group_df["topic_shift_indicator"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        void_summary = compute_mean_summary(group_df["silence_void_outcome"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        challenge_summary = compute_mean_summary(group_df["challenge_event_count"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        continuation_summary = compute_mean_summary(group_df["continuation_event_count"].astype(float).to_numpy()) if not group_df.empty else compute_mean_summary(np.asarray([], dtype=float))
        hawkes_event_values = (
            group_df["repair_event_count"].astype(float).to_numpy() + group_df["challenge_event_count"].astype(float).to_numpy()
            if not group_df.empty
            else np.asarray([], dtype=float)
        )
        hawkes_event_summary = compute_mean_summary(hawkes_event_values)

        summary[trigger_type] = {
            "n_triggers": int(len(group_df)),
            "mean_next_turn_latency_sec": latency_summary["mean"],
            "mean_repair_event_count": repair_summary["mean"],
            "mean_challenge_event_count": challenge_summary["mean"],
            "mean_hawkes_event_count": hawkes_event_summary["mean"],
            "mean_continuation_event_count": continuation_summary["mean"],
            "mean_cascade_duration_sec": duration_summary["mean"],
            "topic_shift_rate": topic_shift_summary["mean"],
            "void_rate": void_summary["mean"],
        }
    return summary


def build_verdict(trigger_df: pd.DataFrame, hawkes_df: pd.DataFrame) -> Dict[str, Any]:
    def value_for(trigger_type: str, column: str) -> Optional[float]:
        group_df = trigger_df[trigger_df["trigger_type"] == trigger_type]
        if group_df.empty:
            return None
        values = group_df[column].astype(float).to_numpy()
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None
        return float(np.mean(finite_values))

    under_alpha = finite_or_none(hawkes_df.loc[hawkes_df["trigger_type"] == "under_info", "alpha"].iloc[0]) if not hawkes_df.empty and "under_info" in set(hawkes_df["trigger_type"]) else None
    over_alpha = finite_or_none(hawkes_df.loc[hawkes_df["trigger_type"] == "over_info", "alpha"].iloc[0]) if not hawkes_df.empty and "over_info" in set(hawkes_df["trigger_type"]) else None
    under_hawkes = value_for("under_info", "repair_event_count")
    over_hawkes = value_for("over_info", "repair_event_count")
    under_challenge = value_for("under_info", "challenge_event_count")
    over_challenge = value_for("over_info", "challenge_event_count")
    under_latency = value_for("under_info", "next_turn_latency_sec")
    over_latency = value_for("over_info", "next_turn_latency_sec")
    under_topic_shift = value_for("under_info", "topic_shift_indicator")
    over_topic_shift = value_for("over_info", "topic_shift_indicator")
    under_void = value_for("under_info", "silence_void_outcome")
    over_void = value_for("over_info", "silence_void_outcome")

    if under_hawkes is not None and under_challenge is not None:
        under_hawkes += under_challenge
    else:
        under_hawkes = None

    if over_hawkes is not None and over_challenge is not None:
        over_hawkes += over_challenge
    else:
        over_hawkes = None

    under_info_burst_supported = bool(
        under_alpha is not None
        and over_alpha is not None
        and under_hawkes is not None
        and over_hawkes is not None
        and under_alpha > over_alpha
        and under_hawkes > over_hawkes
    )
    over_info_void_supported = bool(
        over_latency is not None
        and under_latency is not None
        and over_topic_shift is not None
        and under_topic_shift is not None
        and over_void is not None
        and under_void is not None
        and over_latency > under_latency
        and over_topic_shift >= under_topic_shift
        and over_void >= under_void
    )

    return {
        "under_info_burst_supported": under_info_burst_supported,
        "over_info_void_supported": over_info_void_supported,
    }


def augment_threshold_counts(threshold_df: pd.DataFrame, trigger_df: pd.DataFrame) -> pd.DataFrame:
    if trigger_df.empty:
        threshold_df["under_trigger_count"] = 0
        threshold_df["over_trigger_count"] = 0
        return threshold_df

    counts_df = (
        trigger_df.groupby(["category", "trigger_type"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    merged_df = threshold_df.merge(counts_df, on="category", how="left")
    if "under_info" not in merged_df.columns:
        merged_df["under_info"] = 0
    if "over_info" not in merged_df.columns:
        merged_df["over_info"] = 0
    merged_df["under_trigger_count"] = merged_df["under_info"].fillna(0).astype(int)
    merged_df["over_trigger_count"] = merged_df["over_info"].fillna(0).astype(int)
    return merged_df.drop(columns=[column for column in ["under_info", "over_info"] if column in merged_df.columns])


def save_overview_plot(trigger_df: pd.DataFrame, hawkes_df: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 6, figsize=(21.0, 4.6))

    x_positions = np.arange(len(TRIGGER_ORDER), dtype=float)
    labels = [TRIGGER_LABELS[trigger_type] for trigger_type in TRIGGER_ORDER]
    colors = [TRIGGER_COLORS[trigger_type] for trigger_type in TRIGGER_ORDER]

    branching_ratio_values: List[float] = []
    alpha_values: List[float] = []
    for trigger_type in TRIGGER_ORDER:
        row_df = hawkes_df[hawkes_df["trigger_type"] == trigger_type]
        branching_ratio_values.append(
            float(row_df["branching_ratio"].iloc[0])
            if not row_df.empty and finite_or_none(row_df["branching_ratio"].iloc[0]) is not None
            else 0.0
        )
        alpha_values.append(
            float(row_df["alpha"].iloc[0])
            if not row_df.empty and finite_or_none(row_df["alpha"].iloc[0]) is not None
            else 0.0
        )

    axes[0].bar(x_positions, branching_ratio_values, color=colors, width=0.62, edgecolor="#1f2933", linewidth=0.6)
    axes[0].set_title("Excitation alpha / beta", fontsize=11)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels)

    axes[1].bar(x_positions, alpha_values, color=colors, width=0.62, edgecolor="#1f2933", linewidth=0.6)
    axes[1].set_title("Excitation alpha", fontsize=11)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels)

    metric_specs = [
        ("cascade_duration_sec", "Cascade duration (s)"),
        ("hawkes_event_count", "Repair/challenge events"),
        ("topic_shift_indicator", "Topic-shift rate"),
        ("next_turn_latency_sec", "Next-turn latency (s)"),
    ]

    for axis, (column_name, axis_title) in zip(axes[2:], metric_specs):
        means: List[float] = []
        lower_errors: List[float] = []
        upper_errors: List[float] = []
        for trigger_type in TRIGGER_ORDER:
            group_df = trigger_df[trigger_df["trigger_type"] == trigger_type]
            if group_df.empty:
                summary = compute_mean_summary(np.asarray([], dtype=float))
            elif column_name == "hawkes_event_count":
                values = group_df["repair_event_count"].astype(float).to_numpy() + group_df["challenge_event_count"].astype(float).to_numpy()
                summary = compute_mean_summary(values)
            else:
                summary = compute_mean_summary(group_df[column_name].astype(float).to_numpy())
            mean_value = summary["mean"] if summary["mean"] is not None else 0.0
            ci_low = summary["ci_low"] if summary["ci_low"] is not None else mean_value
            ci_high = summary["ci_high"] if summary["ci_high"] is not None else mean_value
            means.append(float(mean_value))
            lower_errors.append(float(mean_value - ci_low))
            upper_errors.append(float(ci_high - mean_value))

        axis.bar(
            x_positions,
            means,
            color=colors,
            width=0.62,
            edgecolor="#1f2933",
            linewidth=0.6,
            yerr=np.vstack([lower_errors, upper_errors]),
            capsize=4,
        )
        axis.set_title(axis_title, fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels)

    fig.suptitle("Experiment 6: Quantity Violations and Repair Cascades", fontsize=14, y=0.99)
    fig.subplots_adjust(top=0.80, left=0.04, right=0.995, bottom=0.16, wspace=0.34)
    fig.savefig(output_dir / "exp6_cascade_overview.png", dpi=200)
    plt.close(fig)


def sample_trigger_rows(trigger_df: pd.DataFrame, trigger_type: str, sample_count: int) -> pd.DataFrame:
    group_df = trigger_df[trigger_df["trigger_type"] == trigger_type].copy()
    if group_df.empty:
        return group_df

    group_df["hawkes_event_count"] = group_df["repair_event_count"].astype(int) + group_df["challenge_event_count"].astype(int)

    ordered_df = group_df.sort_values(
        ["hawkes_event_count", "cascade_duration_sec", "continuation_event_count", "episode_id", "trigger_turn_idx"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    if len(ordered_df) <= sample_count:
        return ordered_df

    top_count = max(1, sample_count // 2)
    top_df = ordered_df.head(top_count)
    remaining_df = ordered_df.iloc[top_count:].copy()
    remaining_needed = sample_count - len(top_df)
    rng = np.random.default_rng(RNG_SEED)
    sampled_positions = rng.choice(len(remaining_df), size=remaining_needed, replace=False)
    sampled_df = remaining_df.iloc[np.sort(sampled_positions)]
    return pd.concat([top_df, sampled_df], ignore_index=True)


def save_event_cascade_plot(trigger_df: pd.DataFrame, event_df: pd.DataFrame, output_dir: Path, sample_count: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), sharex=True)

    for axis, trigger_type in zip(axes, TRIGGER_ORDER):
        sampled_triggers = sample_trigger_rows(trigger_df, trigger_type, sample_count)
        axis.axvline(0.0, color="#495057", linewidth=1.0, linestyle="--", alpha=0.8)
        axis.set_title(TRIGGER_LABELS[trigger_type], fontsize=12, pad=10)
        axis.set_xlabel("Seconds from trigger turn")
        axis.set_ylabel("Cascade sample")
        axis.grid(alpha=0.12, linewidth=0.6)

        if sampled_triggers.empty:
            axis.text(0.5, 0.5, "No sampled triggers", ha="center", va="center", transform=axis.transAxes, fontsize=11)
            continue

        for row_position, (_, trigger_row) in enumerate(sampled_triggers.iterrows()):
            axis.hlines(
                y=row_position,
                xmin=0.0,
                xmax=float(trigger_row["observation_horizon_sec"]),
                color="#d9e2ec",
                linewidth=1.2,
                alpha=0.95,
            )

            trigger_events = event_df[
                (event_df["trigger_type"] == trigger_type)
                & (event_df["episode_id"] == trigger_row["episode_id"])
                & (event_df["trigger_turn_idx"] == trigger_row["trigger_turn_idx"])
            ].copy()
            for event_type, event_group in trigger_events.groupby("event_type", observed=False):
                axis.scatter(
                    event_group["relative_start_sec"].astype(float).to_numpy(),
                    np.full(len(event_group), row_position, dtype=float),
                    color=EVENT_COLORS[event_type],
                    marker=EVENT_MARKERS[event_type],
                    s=38,
                    alpha=0.92,
                    zorder=4,
                )

        axis.set_ylim(-0.75, len(sampled_triggers) - 0.25)
        axis.set_yticks(np.arange(len(sampled_triggers), dtype=float))
        axis.set_yticklabels([str(index + 1) for index in range(len(sampled_triggers))])

    legend_handles = []
    legend_labels = []
    for event_type in ["repair_event", "challenge_event", "continuation_event", "support_marker", "exit_marker"]:
        handle = plt.Line2D(
            [0],
            [0],
            marker=EVENT_MARKERS[event_type],
            color="none",
            markerfacecolor=EVENT_COLORS[event_type],
            markeredgecolor=EVENT_COLORS[event_type],
            markersize=7,
        )
        legend_handles.append(handle)
        legend_labels.append(event_type.replace("_", " ").title())

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Experiment 6: Event Cascades After Quantity-Like Triggers", fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.88, left=0.07, right=0.99, bottom=0.10, wspace=0.16)
    fig.savefig(output_dir / "exp6_event_cascades.png", dpi=200)
    plt.close(fig)


def run_experiment(
    input_dir: Path,
    output_dir: Path,
    categories: Sequence[str],
    max_episodes_per_category: int,
    event_window_sec: float,
    max_followup_turns: int,
    stability_gap_sec: float,
    under_assumption_quantile: float,
    under_duration_quantile: float,
    over_assumption_quantile: float,
    over_duration_quantile: float,
    show_progress: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = collect_category_paths(input_dir, categories, max_episodes_per_category)
    if not episode_paths:
        raise RuntimeError(f"No episode files were found under {input_dir}.")

    metric_df = collect_trigger_metric_rows(episode_paths, show_progress)
    threshold_df = build_threshold_table(
        metric_df,
        under_assumption_quantile,
        under_duration_quantile,
        over_assumption_quantile,
        over_duration_quantile,
    )
    threshold_lookup = build_threshold_lookup(threshold_df)

    trigger_df, event_df = extract_trigger_and_cascade_rows(
        episode_paths,
        threshold_lookup,
        event_window_sec,
        max_followup_turns,
        stability_gap_sec,
        DEFAULT_TEXT_PREVIEW_CHARS,
        show_progress,
    )
    if trigger_df.empty:
        raise RuntimeError("No under-info or over-info triggers were extracted from the input data.")

    threshold_df = augment_threshold_counts(threshold_df, trigger_df)
    hawkes_df = build_hawkes_summary(trigger_df, event_df)
    outcome_summary = build_outcome_summary(trigger_df)
    verdict = build_verdict(trigger_df, hawkes_df)

    threshold_df.to_csv(output_dir / "exp6_trigger_thresholds.csv", index=False)
    trigger_df.to_csv(output_dir / "exp6_trigger_outcomes.csv", index=False)
    event_df.to_csv(output_dir / "exp6_cascade_events.csv", index=False)
    hawkes_df.to_csv(output_dir / "exp6_hawkes_summary.csv", index=False)

    save_overview_plot(trigger_df, hawkes_df, output_dir)
    save_event_cascade_plot(trigger_df, event_df, output_dir, DEFAULT_SAMPLE_CASCADE_COUNT)

    summary = {
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "categories": list(categories) if categories else sorted(trigger_df["category"].astype(str).unique().tolist()),
            "max_episodes_per_category": int(max_episodes_per_category),
            "event_window_sec": float(event_window_sec),
            "max_followup_turns": int(max_followup_turns),
            "stability_gap_sec": float(stability_gap_sec),
            "under_assumption_quantile": float(under_assumption_quantile),
            "under_duration_quantile": float(under_duration_quantile),
            "over_assumption_quantile": float(over_assumption_quantile),
            "over_duration_quantile": float(over_duration_quantile),
        },
        "dataset": {
            "n_episode_files": int(len(episode_paths)),
            "n_categories": int(trigger_df["category"].nunique()),
            "n_candidate_turns": int(len(metric_df)),
            "n_triggers": int(len(trigger_df)),
            "n_cascade_events": int(len(event_df)),
        },
        "trigger_counts_by_type": {
            trigger_type: int((trigger_df["trigger_type"] == trigger_type).sum()) for trigger_type in TRIGGER_ORDER
        },
        "trigger_counts_by_category": (
            trigger_df.groupby(["category", "trigger_type"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .to_dict(orient="records")
        ),
        "outcome_summary": outcome_summary,
        "hawkes_summary": hawkes_df.to_dict(orient="records"),
        "verdict": verdict,
    }

    summary = sanitize_json_payload(summary)
    with (output_dir / "exp6_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_default)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 6: Quantity Violations and Repair Cascades")
    parser.add_argument("--input_dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--max_episodes_per_category", type=int, default=0)
    parser.add_argument("--event_window_sec", type=float, default=DEFAULT_EVENT_WINDOW_SEC)
    parser.add_argument("--max_followup_turns", type=int, default=DEFAULT_MAX_FOLLOWUP_TURNS)
    parser.add_argument("--stability_gap_sec", type=float, default=DEFAULT_STABILITY_GAP_SEC)
    parser.add_argument("--under_assumption_quantile", type=float, default=DEFAULT_UNDER_ASSUMPTION_QUANTILE)
    parser.add_argument("--under_duration_quantile", type=float, default=DEFAULT_UNDER_DURATION_QUANTILE)
    parser.add_argument("--over_assumption_quantile", type=float, default=DEFAULT_OVER_ASSUMPTION_QUANTILE)
    parser.add_argument("--over_duration_quantile", type=float, default=DEFAULT_OVER_DURATION_QUANTILE)
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categories=args.categories,
        max_episodes_per_category=int(args.max_episodes_per_category),
        event_window_sec=float(args.event_window_sec),
        max_followup_turns=int(args.max_followup_turns),
        stability_gap_sec=float(args.stability_gap_sec),
        under_assumption_quantile=float(args.under_assumption_quantile),
        under_duration_quantile=float(args.under_duration_quantile),
        over_assumption_quantile=float(args.over_assumption_quantile),
        over_duration_quantile=float(args.over_duration_quantile),
        show_progress=not bool(args.no_tqdm),
    )
    print(json.dumps(summary["verdict"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
