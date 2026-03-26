import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from tqdm.auto import tqdm


DEFAULT_INPUT_DIR = "data/conversation_moves_labeled"
DEFAULT_MAXIM_DIR = "data/maxim_violations_labeled"
DEFAULT_OUTPUT_DIR = "experiments/exp7_social_power/results"
ANNOTATED_VIOLATION_TRIGGER = "annotated_local_context"
ANNOTATED_VIOLATION_SOURCE = "maxim_violation_label"
HEURISTIC_VIOLATION_SOURCE = "heuristic_fallback"

ROLE_ORDER = ("guest", "host")
VIOLATION_ORDER = ("Manner", "Relation", "Quantity")
VIOLATION_SEVERITY = {
    "Manner": 1,
    "Relation": 2,
    "Quantity": 3,
}

REPAIR_MOVES = {
    "Clarification Request (Generic)",
    "Clarification Request (Specific)",
}
CHALLENGE_MOVES = {
    "Correction / Challenge",
}
SELF_REPAIR_MOVES = {
    "Self-Correction",
}
POLICING_MOVES = REPAIR_MOVES | CHALLENGE_MOVES
FIXED_EFFECT_COLUMNS = [
    "Intercept",
    "Role[host]",
    "Violation[Relation]",
    "Violation[Quantity]",
    "Role[host]:Violation[Relation]",
    "Role[host]:Violation[Quantity]",
]
NON_VIOLATION_LABELS = {
    "no violation",
    "none",
    "no maxim violation",
}
POLAR_QUESTION_RE = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|can|could|will|would|should|have|has|had|"
    r"am|didn't|doesn't|isn't|aren't|wasn't|weren't|won't|can't|couldn't|shouldn't)\b",
    re.IGNORECASE,
)
GREETING_CHECK_RE = re.compile(
    r"\b(?:how are you|how're you|how you doing|how's it going|how have you been)\b",
    re.IGNORECASE,
)
AUDIO_CHECK_RE = re.compile(
    r"\b(?:can you hear me|can you hear us|can you see me|can you see us|am i audible|"
    r"can you hear that|audio check|mic check)\b",
    re.IGNORECASE,
)
BRIEF_EXPECTED_RE = re.compile(
    r"\b(?:rapid fire|favorite|one or two words|one word|in two words|finish this sentence|"
    r"tell us (?:his|her|their) name again|what(?:'s| is) your favorite|which\b|who\b|where\b|"
    r"when\b|how many\b|how much\b|how old\b|what year\b|what city\b|what state\b|what food\b|"
    r"what movie\b|what book\b|what band\b|what musician\b|what team\b)\b",
    re.IGNORECASE,
)
OPEN_ENDED_RE = re.compile(
    r"\b(?:why\b|how do you\b|how would you\b|what do you think\b|what are your thoughts\b|"
    r"tell me about\b|tell us about\b|walk me through\b|walk us through\b|can you explain\b|"
    r"help me understand\b|what happened\b|what should\b|how should\b|"
    r"what does .* mean(?: to you)?\b|what does .* look like\b)\b",
    re.IGNORECASE,
)
STONEWALL_RE = re.compile(
    r"\b(?:maybe|i guess|guess so|not really|hard to say|depends|not sure|i don't know|"
    r"don't know|no comment|we'll see|who knows|whatever|perhaps|probably)\b",
    re.IGNORECASE,
)
POLAR_ANSWER_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|no|nope|nah|absolutely|definitely|certainly|correct|right|"
    r"exactly|of course|not really)\b",
    re.IGNORECASE,
)
ACK_CLOSING_RE = re.compile(
    r"^\s*(?:thanks?|thank you|appreciate it|take care|bye(?:-bye)?|goodbye|sounds good|"
    r"all right|alright|okay|ok)\b[\s.!?,;:'\"-]*$",
    re.IGNORECASE,
)
CONTENT_WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "for",
    "i",
    "i'm",
    "im",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "we",
    "you",
    "your",
}


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        return int(float(value))
    except Exception:
        return default


def safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def normal_survival(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def finite_or_none(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def sort_turns(turns: Iterable[Dict]) -> List[Dict]:
    indexed: List[Tuple[int, int, Dict]] = []
    for pos, turn in enumerate(turns):
        indexed.append((safe_int(turn.get("turn_idx"), pos), pos, turn))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def load_turns(path: Path) -> Optional[List[Dict]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, list):
        return sort_turns(payload)
    if isinstance(payload, dict) and isinstance(payload.get("turns"), list):
        return sort_turns(payload["turns"])
    return None


def turn_text(turn: Optional[Dict]) -> str:
    if not turn:
        return ""
    return str(turn.get("turn_text") or turn.get("transcript") or "").strip()


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def speaker_id(turn: Optional[Dict]) -> str:
    if not turn:
        return ""
    return str(turn.get("speaker_id") or turn.get("speaker") or "").strip()


def same_speaker(turn_a: Optional[Dict], turn_b: Optional[Dict]) -> bool:
    speaker_a = speaker_id(turn_a)
    speaker_b = speaker_id(turn_b)
    return bool(speaker_a and speaker_b and speaker_a == speaker_b)


def normalize_role(turn: Dict) -> str:
    role = str(turn.get("speaker_role") or "").strip().lower()
    return role if role in ROLE_ORDER else "unknown"


def word_count(turn: Dict) -> int:
    explicit = safe_float(turn.get("wordCount", turn.get("word_count")))
    if math.isfinite(explicit):
        return max(0, int(round(explicit)))
    text = turn_text(turn)
    if not text:
        return 0
    return len(text.split())


def previous_turn_projects_response(prev_turn: Optional[Dict], turn: Dict) -> bool:
    if not prev_turn or same_speaker(prev_turn, turn):
        return False
    move = str(prev_turn.get("conversation_move_label") or "").strip()
    if move in REPAIR_MOVES or move in CHALLENGE_MOVES:
        return True
    return "?" in turn_text(prev_turn)


def is_substantive_turn(turn: Dict) -> bool:
    turn_type = str(turn.get("turn_type_label") or "").strip()
    if turn_type in {"Backchannel", "Procedural"}:
        return False
    return bool(turn_text(turn))


def content_word_count(text: str) -> int:
    tokens = [tok.lower() for tok in CONTENT_WORD_RE.findall(text)]
    return sum(1 for tok in tokens if tok not in STOPWORDS)


def response_features(text: str) -> Dict[str, object]:
    lowered = normalize_space(text).lower()
    words = len(lowered.split()) if lowered else 0
    features = {
        "word_count": words,
        "has_stonewall_marker": bool(STONEWALL_RE.search(lowered)),
        "is_polar_answer": bool(POLAR_ANSWER_RE.match(lowered)),
        "is_ack_or_closing": bool(ACK_CLOSING_RE.match(lowered)),
        "content_word_count": content_word_count(lowered),
    }
    features["looks_contentful"] = bool(
        words > 0
        and not features["has_stonewall_marker"]
        and not features["is_ack_or_closing"]
        and features["content_word_count"] >= 1
    )
    return features


def question_features(previous_turns: Sequence[Dict]) -> Dict[str, object]:
    prev_turn = previous_turns[-1] if previous_turns else None
    prev_text = turn_text(prev_turn)
    prev_lower = prev_text.lower()
    context_text = " ".join(turn_text(turn) for turn in previous_turns[-3:])
    context_lower = context_text.lower()

    is_greeting_check = bool(GREETING_CHECK_RE.search(prev_lower))
    is_audio_check = bool(AUDIO_CHECK_RE.search(prev_lower))
    is_brief_expected = bool(BRIEF_EXPECTED_RE.search(prev_lower) or BRIEF_EXPECTED_RE.search(context_lower))
    is_forced_choice = (" or " in f" {prev_lower} " and "?" in prev_text)
    is_polar = bool(POLAR_QUESTION_RE.match(prev_lower) and "?" in prev_text)
    is_open_ended = bool(
        OPEN_ENDED_RE.search(prev_lower)
        and not is_brief_expected
        and not is_greeting_check
        and not is_audio_check
        and not is_polar
    )
    return {
        "is_polar_question": is_polar,
        "is_brief_expected_prompt": bool(is_brief_expected or is_forced_choice),
        "is_open_ended_prompt": is_open_ended,
        "is_greeting_check": is_greeting_check,
        "is_audio_check": is_audio_check,
    }


def normalize_maxim_violation_label(value: object) -> Optional[str]:
    text = normalize_space(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in NON_VIOLATION_LABELS:
        return None
    for label in VIOLATION_ORDER:
        if lowered == label.lower():
            return label
    return None


def explicit_no_violation_label(value: object) -> bool:
    text = normalize_space(value)
    return bool(text) and text.lower() in NON_VIOLATION_LABELS


def annotation_violation_label(turn: Dict) -> Tuple[Optional[str], bool]:
    raw = turn.get("maxim_violation_label")
    if raw is None:
        return None, False
    label = normalize_maxim_violation_label(raw)
    if label is not None:
        return label, True
    if explicit_no_violation_label(raw):
        return None, True
    return None, False


def heuristic_violation_label(
    turn: Dict,
    prev_turn: Optional[Dict],
    previous_turns: Sequence[Dict],
    short_answer_max_words: int,
) -> Tuple[Optional[str], Optional[str]]:
    move = str(turn.get("conversation_move_label") or "").strip()
    turn_type = str(turn.get("turn_type_label") or "").strip()
    words = word_count(turn)
    text = turn_text(turn)

    if not text:
        return None, None

    if move == "Self-Correction":
        return None, None

    if turn_type == "Disrupted" and is_substantive_turn(turn):
        return "Manner", "incomplete_or_unclear_turn"

    if not is_substantive_turn(turn) or not previous_turn_projects_response(prev_turn, turn):
        return None, None

    q_meta = question_features(previous_turns)
    r_meta = response_features(text)

    if move == "Topic Shift":
        return "Relation", "nonresponsive_shift_after_projected_response"

    if move == "Stonewalling / Non-Response":
        if q_meta["is_greeting_check"] or q_meta["is_audio_check"] or q_meta["is_brief_expected_prompt"]:
            return None, None
        if q_meta["is_polar_question"] and r_meta["is_polar_answer"] and not r_meta["has_stonewall_marker"]:
            return None, None
        if r_meta["is_ack_or_closing"] and not q_meta["is_open_ended_prompt"]:
            return None, None
        return "Quantity", "underinformative_nonresponse"

    if move == "Answer" and prev_turn and "?" in turn_text(prev_turn):
        if (
            q_meta["is_open_ended_prompt"]
            and words <= short_answer_max_words
            and (
                r_meta["has_stonewall_marker"]
                or not r_meta["looks_contentful"]
                or int(r_meta["content_word_count"]) <= 1
            )
        ):
            return "Quantity", "underinformative_short_answer"

    return None, None


def classify_violation(
    turn: Dict,
    prev_turn: Optional[Dict],
    previous_turns: Sequence[Dict],
    short_answer_max_words: int,
) -> Tuple[Optional[str], Optional[str], str]:
    annotated_label, annotation_available = annotation_violation_label(turn)
    if annotation_available:
        if annotated_label is None:
            return None, None, ANNOTATED_VIOLATION_SOURCE
        return annotated_label, ANNOTATED_VIOLATION_TRIGGER, ANNOTATED_VIOLATION_SOURCE

    violation, trigger = heuristic_violation_label(
        turn=turn,
        prev_turn=prev_turn,
        previous_turns=previous_turns,
        short_answer_max_words=short_answer_max_words,
    )
    return violation, trigger, HEURISTIC_VIOLATION_SOURCE


def classify_next_turn(
    source_turn: Dict,
    next_turn: Optional[Dict],
) -> Dict[str, object]:
    source_speaker = speaker_id(source_turn)
    next_speaker = ""
    next_role = ""
    next_move = ""
    same_speaker = False
    repair = 0
    challenge = 0
    external_policing = 0
    self_repair = 0

    if next_turn:
        next_speaker = speaker_id(next_turn)
        next_role = normalize_role(next_turn)
        next_move = str(next_turn.get("conversation_move_label") or "").strip()
        same_speaker = bool(source_speaker and next_speaker and source_speaker == next_speaker)
        repair = int(next_move in REPAIR_MOVES)
        challenge = int(next_move in CHALLENGE_MOVES)
        if same_speaker and next_move in SELF_REPAIR_MOVES:
            self_repair = 1
        if (not same_speaker) and next_move in POLICING_MOVES:
            external_policing = 1

    return {
        "next_move": next_move,
        "next_role": next_role,
        "same_speaker": int(same_speaker),
        "repair": repair,
        "challenge": challenge,
        "self_repair": self_repair,
        "policed": external_policing,
    }


def merge_maxim_annotations(turns: List[Dict], annotation_turns: Sequence[Dict]) -> None:
    by_idx: Dict[int, Dict] = {}
    for pos, ann_turn in enumerate(annotation_turns):
        turn_idx = safe_int(ann_turn.get("turn_idx"), pos)
        by_idx.setdefault(turn_idx, ann_turn)

    for pos, turn in enumerate(turns):
        turn_idx = safe_int(turn.get("turn_idx"), pos)
        ann_turn = by_idx.get(turn_idx)
        if not ann_turn:
            continue
        for field in ("maxim_violation_label", "maxim_violation_scheme", "maxim_violation_label_error"):
            if field in ann_turn:
                turn[field] = ann_turn[field]


def build_rows(
    input_dir: Path,
    maxim_dir: Optional[Path],
    short_answer_max_words: int,
    max_files: int = 0,
) -> Tuple[List[Dict], Counter, Counter, Counter]:
    direct_files = sorted(input_dir.glob("*.json"))
    files = direct_files if direct_files else sorted(input_dir.glob("*/*.json"))
    if max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
    role_turn_counts: Counter = Counter()
    violation_counts: Counter = Counter()
    violation_source_counts: Counter = Counter()

    for path in tqdm(files, desc="Scanning episodes", unit="file"):
        turns = load_turns(path)
        if not turns:
            continue

        if maxim_dir is not None:
            try:
                relative_path = path.relative_to(input_dir)
            except ValueError:
                relative_path = path.name
            maxim_path = maxim_dir / relative_path
            if maxim_path.exists() and maxim_path.resolve() != path.resolve():
                maxim_turns = load_turns(maxim_path)
                if maxim_turns:
                    merge_maxim_annotations(turns, maxim_turns)

        episode_id = str(turns[0].get("episode_id") or path.stem)
        for idx, turn in enumerate(turns):
            role = normalize_role(turn)
            if role in ROLE_ORDER:
                role_turn_counts[role] += 1

            prev_turn = turns[idx - 1] if idx > 0 else None
            previous_turns = turns[max(0, idx - 3):idx]
            next_turn = turns[idx + 1] if idx + 1 < len(turns) else None
            violation, trigger, violation_source = classify_violation(
                turn=turn,
                prev_turn=prev_turn,
                previous_turns=previous_turns,
                short_answer_max_words=short_answer_max_words,
            )
            if violation is None or role not in ROLE_ORDER:
                continue

            next_meta = classify_next_turn(turn, next_turn)
            turn_idx = safe_int(turn.get("turn_idx"), idx)
            wc = word_count(turn)
            violation_counts[(role, violation)] += 1
            violation_source_counts[violation_source] += 1

            rows.append(
                {
                    "episode_id": episode_id,
                    "turn_idx": turn_idx,
                    "speaker_role": role,
                    "speaker_id": str(turn.get("speaker_id") or turn.get("speaker") or ""),
                    "violation_type": violation,
                    "violation_trigger": trigger,
                    "violation_source": violation_source,
                    "severity": VIOLATION_SEVERITY[violation],
                    "conversation_move_label": str(turn.get("conversation_move_label") or ""),
                    "turn_type_label": str(turn.get("turn_type_label") or ""),
                    "word_count": wc,
                    "turn_text": turn_text(turn),
                    "next_move": next_meta["next_move"],
                    "next_role": next_meta["next_role"],
                    "same_speaker_next": next_meta["same_speaker"],
                    "repair_next": next_meta["repair"],
                    "challenge_next": next_meta["challenge"],
                    "self_repair_next": next_meta["self_repair"],
                    "policed_next": next_meta["policed"],
                }
            )

    rows.sort(key=lambda row: (str(row["episode_id"]), int(row["turn_idx"])))
    return rows, role_turn_counts, violation_counts, violation_source_counts


def canonical_term_name(name: str) -> str:
    raw = str(name or "").strip()
    if raw in FIXED_EFFECT_COLUMNS:
        return raw
    if raw == "Intercept":
        return raw

    has_role = "speaker_role" in raw and "[T.host]" in raw
    has_relation = "violation_type" in raw and "[T.Relation]" in raw
    has_quantity = "violation_type" in raw and "[T.Quantity]" in raw

    if has_role and has_relation:
        return "Role[host]:Violation[Relation]"
    if has_role and has_quantity:
        return "Role[host]:Violation[Quantity]"
    if has_role:
        return "Role[host]"
    if has_relation:
        return "Violation[Relation]"
    if has_quantity:
        return "Violation[Quantity]"
    return raw


def fit_statsmodels_mixed_logit(
    rows: Sequence[Dict],
    max_iter: int,
) -> Dict[str, object]:
    tqdm.write("Fitting mixed-effects logistic model with statsmodels...")
    frame = pd.DataFrame(
        {
            "episode_id": [str(row["episode_id"]) for row in rows],
            "speaker_role": [str(row["speaker_role"]) for row in rows],
            "violation_type": [str(row["violation_type"]) for row in rows],
            "policed_next": [int(row["policed_next"]) for row in rows],
        }
    )
    frame["speaker_role"] = pd.Categorical(frame["speaker_role"], categories=list(ROLE_ORDER), ordered=True)
    frame["violation_type"] = pd.Categorical(frame["violation_type"], categories=list(VIOLATION_ORDER), ordered=True)

    formula = (
        "policed_next ~ "
        "C(speaker_role, Treatment(reference='guest')) * "
        "C(violation_type, Treatment(reference='Manner'))"
    )
    vc_formulas = {
        "episode_intercept": "0 + C(episode_id)",
    }

    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, frame)
    try:
        result = model.fit_vb(
            fit_method="BFGS",
            minim_opts={"maxiter": max_iter},
        )
        fit_mode = "statsmodels_vb"
    except Exception:
        result = model.fit_map(
            method="BFGS",
            minim_opts={"maxiter": max_iter},
        )
        fit_mode = "statsmodels_map"

    term_names = [canonical_term_name(name) for name in getattr(model, "exog_names", [])]
    fe_mean = np.asarray(getattr(result, "fe_mean"), dtype=float)
    fe_sd = np.asarray(getattr(result, "fe_sd"), dtype=float)
    term_to_idx = {name: idx for idx, name in enumerate(term_names)}

    beta = np.zeros(len(FIXED_EFFECT_COLUMNS), dtype=float)
    se = np.full(len(FIXED_EFFECT_COLUMNS), np.nan, dtype=float)
    for idx, term in enumerate(FIXED_EFFECT_COLUMNS):
        source_idx = term_to_idx.get(term)
        if source_idx is None:
            continue
        beta[idx] = float(fe_mean[source_idx])
        se[idx] = float(fe_sd[source_idx])

    predicted = result.predict()
    p_hat = np.asarray(predicted, dtype=float)

    log_lik = float(
        np.sum(
            frame["policed_next"].to_numpy(dtype=float) * np.log(np.clip(p_hat, 1e-12, 1.0))
            + (1.0 - frame["policed_next"].to_numpy(dtype=float)) * np.log(np.clip(1.0 - p_hat, 1e-12, 1.0))
        )
    )

    optim = getattr(result, "optim_retvals", None)
    converged = True
    iterations = 0
    objective = None
    if isinstance(optim, dict):
        converged = bool(optim.get("success", True))
        iterations = safe_int(optim.get("nit"), 0)
        objective = finite_or_none(optim.get("fun"))

    return {
        "beta": beta,
        "u": np.array([], dtype=float),
        "cov": np.diag(np.where(np.isfinite(se), se ** 2, np.nan)),
        "se": se,
        "p_hat": p_hat,
        "log_lik": log_lik,
        "objective": objective,
        "converged": converged,
        "iterations": iterations,
        "method": fit_mode,
    }


def coefficient_rows(
    columns: Sequence[str],
    beta: np.ndarray,
    se: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(columns):
        estimate = float(beta[idx])
        std_err = float(se[idx])
        z_score = estimate / std_err if std_err > 0 else float("nan")
        p_value = 2.0 * normal_survival(abs(z_score)) if math.isfinite(z_score) else float("nan")
        ci_low = estimate - (1.96 * std_err)
        ci_high = estimate + (1.96 * std_err)
        rows.append(
            {
                "term": name,
                "estimate": estimate,
                "std_error": std_err,
                "z_score": z_score,
                "p_value_approx": p_value,
                "odds_ratio": math.exp(estimate),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return rows


def violation_matrix_rows(
    role_turn_counts: Counter,
    violation_counts: Counter,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role in ROLE_ORDER:
        total_turns = int(role_turn_counts.get(role, 0))
        role_violation_total = sum(int(violation_counts.get((role, v), 0)) for v in VIOLATION_ORDER)
        for violation in VIOLATION_ORDER:
            count = int(violation_counts.get((role, violation), 0))
            rows.append(
                {
                    "speaker_role": role,
                    "violation_type": violation,
                    "total_turns_for_role": total_turns,
                    "violation_count": count,
                    "violation_rate_per_turn": (count / total_turns) if total_turns else 0.0,
                    "share_of_role_violations": (count / role_violation_total) if role_violation_total else 0.0,
                }
            )
    return rows


def enforcement_summary_rows(rows: Sequence[Dict]) -> List[Dict[str, object]]:
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        key = (str(row["speaker_role"]), str(row["violation_type"]))
        bucket = summary.setdefault(
            key,
            {
                "n_violations": 0.0,
                "policed_next": 0.0,
                "repair_next": 0.0,
                "challenge_next": 0.0,
                "self_repair_next": 0.0,
            },
        )
        bucket["n_violations"] += 1.0
        bucket["policed_next"] += float(row["policed_next"])
        bucket["repair_next"] += float(row["repair_next"])
        bucket["challenge_next"] += float(row["challenge_next"])
        bucket["self_repair_next"] += float(row["self_repair_next"])

    out: List[Dict[str, object]] = []
    for role in ROLE_ORDER:
        for violation in VIOLATION_ORDER:
            bucket = summary.get((role, violation), None)
            n = float(bucket["n_violations"]) if bucket else 0.0
            policed = float(bucket["policed_next"]) if bucket else 0.0
            repairs = float(bucket["repair_next"]) if bucket else 0.0
            challenges = float(bucket["challenge_next"]) if bucket else 0.0
            self_repairs = float(bucket["self_repair_next"]) if bucket else 0.0
            out.append(
                {
                    "speaker_role": role,
                    "violation_type": violation,
                    "n_violations": int(n),
                    "policed_next_count": int(policed),
                    "repair_next_count": int(repairs),
                    "challenge_next_count": int(challenges),
                    "self_repair_next_count": int(self_repairs),
                    "policed_next_rate": (policed / n) if n else 0.0,
                    "repair_next_rate": (repairs / n) if n else 0.0,
                    "challenge_next_rate": (challenges / n) if n else 0.0,
                    "self_repair_next_rate": (self_repairs / n) if n else 0.0,
                }
            )
    return out


def predicted_curve(beta: np.ndarray) -> List[Dict[str, object]]:
    curve: List[Dict[str, object]] = []
    for role in ROLE_ORDER:
        role_host = 1.0 if role == "host" else 0.0
        for violation in VIOLATION_ORDER:
            is_relation = 1.0 if violation == "Relation" else 0.0
            is_quantity = 1.0 if violation == "Quantity" else 0.0
            features = np.array(
                [
                    1.0,
                    role_host,
                    is_relation,
                    is_quantity,
                    role_host * is_relation,
                    role_host * is_quantity,
                ],
                dtype=float,
            )
            eta = float(np.dot(features, beta))
            probability = float(sigmoid(np.array([eta], dtype=float))[0])
            curve.append(
                {
                    "speaker_role": role,
                    "violation_type": violation,
                    "severity": VIOLATION_SEVERITY[violation],
                    "predicted_policed_probability": probability,
                }
            )
    curve.sort(key=lambda row: (str(row["speaker_role"]), int(row["severity"])))
    return curve


def overall_rate(rows: Sequence[Dict], role: str, field: str) -> float:
    filtered = [row for row in rows if row["speaker_role"] == role]
    if not filtered:
        return 0.0
    total = sum(float(row[field]) for row in filtered)
    return total / float(len(filtered))


def model_weighted_gap(curve_rows: Sequence[Dict], rows: Sequence[Dict]) -> float:
    violation_counts = Counter(str(row["violation_type"]) for row in rows)
    total = float(sum(violation_counts.values()))
    if total <= 0:
        return 0.0

    by_key = {
        (str(row["speaker_role"]), str(row["violation_type"])): float(row["predicted_policed_probability"])
        for row in curve_rows
    }
    guest = 0.0
    host = 0.0
    for violation in VIOLATION_ORDER:
        weight = float(violation_counts.get(violation, 0)) / total
        guest += weight * by_key.get(("guest", violation), 0.0)
        host += weight * by_key.get(("host", violation), 0.0)
    return guest - host


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg_plot(path: Path, curve_rows: Sequence[Dict[str, object]]) -> None:
    width = 920
    height = 560
    left = 90
    right = 40
    top = 50
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(severity: int) -> float:
        if len(VIOLATION_ORDER) == 1:
            return left + (plot_w / 2.0)
        return left + ((severity - 1) / float(len(VIOLATION_ORDER) - 1)) * plot_w

    def y_pos(probability: float) -> float:
        return top + ((1.0 - probability) * plot_h)

    colors = {
        "guest": "#b42318",
        "host": "#175cd3",
    }

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for tick in range(0, 6):
        probability = tick / 5.0
        y = y_pos(probability)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        label = f"{probability:.1f}"
        lines.append(f'<text x="{left - 14}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#344054">{label}</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#101828" stroke-width="2"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#101828" stroke-width="2"/>')

    for violation in VIOLATION_ORDER:
        severity = VIOLATION_SEVERITY[violation]
        x = x_pos(severity)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 7}" stroke="#101828" stroke-width="2"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" font-size="12" text-anchor="middle" fill="#344054">'
            f'{escape_xml(f"{severity}. {violation}")}</text>'
        )

    for role in ROLE_ORDER:
        role_rows = [row for row in curve_rows if row["speaker_role"] == role]
        role_rows.sort(key=lambda row: int(row["severity"]))
        points: List[Tuple[float, float]] = []
        for row in role_rows:
            x = x_pos(int(row["severity"]))
            y = y_pos(float(row["predicted_policed_probability"]))
            points.append((x, y))
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{colors[role]}" stroke-width="3"/>')
        for (x, y), row in zip(points, role_rows):
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[role]}"/>')
            label = f'{float(row["predicted_policed_probability"]):.2f}'
            lines.append(f'<text x="{x:.2f}" y="{y - 10:.2f}" font-size="11" text-anchor="middle" fill="{colors[role]}">{label}</text>')

    lines.append('<text x="460" y="26" font-size="20" text-anchor="middle" fill="#101828">Exp 7: Status Shield Interaction Effect</text>')
    lines.append('<text x="460" y="44" font-size="12" text-anchor="middle" fill="#475467">Predicted probability that the next turn polices the violation</text>')
    lines.append(
        f'<text x="{left + (plot_w / 2.0):.2f}" y="{height - 22}" font-size="13" text-anchor="middle" fill="#101828">'
        'Violation ordering used in analysis (Manner -&gt; Relation -&gt; Quantity)</text>'
    )
    lines.append(
        f'<text transform="translate(22 {top + (plot_h / 2.0):.2f}) rotate(-90)" font-size="13" text-anchor="middle" fill="#101828">'
        'Probability of policing</text>'
    )

    legend_y = top + 12
    legend_x = width - 200
    for idx, role in enumerate(ROLE_ORDER):
        y = legend_y + (idx * 24)
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{colors[role]}" stroke-width="3"/>')
        lines.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="4" fill="{colors[role]}"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-size="12" fill="#344054">{role.title()}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    input_dir: Path,
    maxim_dir: Optional[Path],
    output_dir: Path,
    short_answer_max_words: int,
    max_files: int,
    max_iter: int,
) -> Dict[str, object]:
    rows, role_turn_counts, violation_counts, violation_source_counts = build_rows(
        input_dir=input_dir,
        maxim_dir=maxim_dir,
        short_answer_max_words=short_answer_max_words,
        max_files=max_files,
    )
    if not rows:
        raise RuntimeError("No violation rows were constructed from the input directory.")

    groups = sorted({str(row["episode_id"]) for row in rows})
    model = fit_statsmodels_mixed_logit(
        rows=rows,
        max_iter=max_iter,
    )

    coeffs = coefficient_rows(FIXED_EFFECT_COLUMNS, model["beta"], model["se"])
    matrix_rows = violation_matrix_rows(role_turn_counts, violation_counts)
    enforcement_rows = enforcement_summary_rows(rows)
    curve_rows = predicted_curve(model["beta"])

    observed_guest = overall_rate(rows, "guest", "policed_next")
    observed_host = overall_rate(rows, "host", "policed_next")
    observed_gap = observed_guest - observed_host
    model_gap = model_weighted_gap(curve_rows, rows)

    interaction_map = {row["term"]: row["estimate"] for row in coeffs}
    interaction_effects = {
        "role_host_x_relation": float(interaction_map.get("Role[host]:Violation[Relation]", 0.0)),
        "role_host_x_quantity": float(interaction_map.get("Role[host]:Violation[Quantity]", 0.0)),
    }
    interaction_magnitude = max(abs(val) for val in interaction_effects.values()) if interaction_effects else 0.0

    summary = {
        "experiment": "exp7_social_power",
        "input_dir": str(input_dir),
        "maxim_dir": str(maxim_dir) if maxim_dir is not None else None,
        "output_dir": str(output_dir),
        "n_violation_rows": len(rows),
        "n_conversations": len(groups),
        "short_answer_max_words": short_answer_max_words,
        "model_formula_proxy": "P(policed_next) ~ violation_type * speaker_role + (1|episode_id)",
        "operationalization": {
            "definition": (
                "A maxim violation is a substantive source turn that, relative to the immediately "
                "preceding turn and the action it projects, creates a local problem of sufficiency, "
                "relevance, or interpretability strong enough that a cooperative recipient would be "
                "warranted in withholding straightforward uptake."
            ),
            "label_source_precedence": [
                "Use maxim_violation_label when present on the turn or in a matching sidecar annotation file.",
                "Otherwise use conservative prev/current-turn heuristics only.",
            ],
            "Quantity": [
                "Underinformative or overinformative relative to the immediate discourse demand.",
                "Fallback heuristic: stonewalling/non-response only when the previous turn projects a response and the reply is not an adequate brief or polar answer.",
                f"Fallback heuristic: very short answer (<= {short_answer_max_words} words) only after an open-ended question when it remains underinformative.",
            ],
            "Relation": [
                "Insufficiently responsive to the question, challenge, or repair request currently on the table.",
                "Fallback heuristic: Topic Shift only when the previous turn projects a response from the current speaker.",
            ],
            "Manner": [
                "Too unclear, incomplete, disordered, or under-specified for current purposes.",
                "Fallback heuristic: disrupted turn type when the turn is locally incomplete or unclear.",
            ],
            "ExcludedFromViolation": [
                "Backchannel and Procedural turns.",
                "Ordinary topic management when no response is projected from the current speaker.",
                "Self-Correction as such; self-repair within the same turn is not automatically a violation.",
                "Rudeness, disagreement, or socially dispreferred but usable turns.",
            ],
            "PolicedNext": [
                "N+1 is a different speaker",
                "N+1 move is a clarification request or correction / challenge",
            ],
        },
        "violation_source_counts": {str(key): int(value) for key, value in violation_source_counts.items()},
        "role_turn_counts": {role: int(role_turn_counts.get(role, 0)) for role in ROLE_ORDER},
        "observed_policing_rate": {
            "guest": observed_guest,
            "host": observed_host,
            "guest_minus_host": observed_gap,
        },
        "model_weighted_guest_minus_host_gap": model_gap,
        "status_shield_supported": bool(model_gap > 0.0),
        "interaction_effects": interaction_effects,
        "interaction_effect_magnitude": interaction_magnitude,
        "model_fit": {
            "method": str(model.get("method") or "unknown"),
            "converged": bool(model["converged"]),
            "iterations": int(model["iterations"]),
            "log_likelihood": finite_or_none(model.get("log_lik")),
            "objective": finite_or_none(model.get("objective")),
        },
    }

    output_jobs = [
        (
            "exp7_violation_turns.csv",
            lambda: write_csv(
                output_dir / "exp7_violation_turns.csv",
                rows,
                [
                    "episode_id",
                    "turn_idx",
                    "speaker_role",
                    "speaker_id",
                    "violation_type",
                    "violation_trigger",
                    "violation_source",
                    "severity",
                    "conversation_move_label",
                    "turn_type_label",
                    "word_count",
                    "turn_text",
                    "next_move",
                    "next_role",
                    "same_speaker_next",
                    "repair_next",
                    "challenge_next",
                    "self_repair_next",
                    "policed_next",
                ],
            ),
        ),
        (
            "exp7_violation_matrix.csv",
            lambda: write_csv(
                output_dir / "exp7_violation_matrix.csv",
                matrix_rows,
                [
                    "speaker_role",
                    "violation_type",
                    "total_turns_for_role",
                    "violation_count",
                    "violation_rate_per_turn",
                    "share_of_role_violations",
                ],
            ),
        ),
        (
            "exp7_enforcement_summary.csv",
            lambda: write_csv(
                output_dir / "exp7_enforcement_summary.csv",
                enforcement_rows,
                [
                    "speaker_role",
                    "violation_type",
                    "n_violations",
                    "policed_next_count",
                    "repair_next_count",
                    "challenge_next_count",
                    "self_repair_next_count",
                    "policed_next_rate",
                    "repair_next_rate",
                    "challenge_next_rate",
                    "self_repair_next_rate",
                ],
            ),
        ),
        (
            "exp7_model_coefficients.csv",
            lambda: write_csv(
                output_dir / "exp7_model_coefficients.csv",
                coeffs,
                [
                    "term",
                    "estimate",
                    "std_error",
                    "z_score",
                    "p_value_approx",
                    "odds_ratio",
                    "ci95_low",
                    "ci95_high",
                ],
            ),
        ),
        (
            "exp7_interaction_curve.csv",
            lambda: write_csv(
                output_dir / "exp7_interaction_curve.csv",
                curve_rows,
                [
                    "speaker_role",
                    "violation_type",
                    "severity",
                    "predicted_policed_probability",
                ],
            ),
        ),
        (
            "exp7_status_shield_plot.svg",
            lambda: render_svg_plot(output_dir / "exp7_status_shield_plot.svg", curve_rows),
        ),
        (
            "exp7_summary.json",
            lambda: (output_dir / "exp7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"),
        ),
    ]

    for label, job in tqdm(output_jobs, desc="Writing outputs", unit="file"):
        job()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 7: test whether host turns are policed less often than guest turns "
            "for comparable local maxim violations, preferring grounding-based maxim_violation_label "
            "annotations when available."
        )
    )
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--maxim_dir",
        type=str,
        default=DEFAULT_MAXIM_DIR,
        help="Optional sidecar directory with per-turn maxim_violation_label annotations.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--short_answer_max_words", type=int, default=6)
    parser.add_argument("--max_files", type=int, default=0, help="Optional debug cap on the number of episode files.")
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    maxim_dir = Path(args.maxim_dir) if str(args.maxim_dir).strip() else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if maxim_dir is not None and not maxim_dir.exists():
        maxim_dir = None

    summary = run_analysis(
        input_dir=input_dir,
        maxim_dir=maxim_dir,
        output_dir=output_dir,
        short_answer_max_words=max(1, int(args.short_answer_max_words)),
        max_files=max(0, int(args.max_files)),
        max_iter=max(20, int(args.max_iter)),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
