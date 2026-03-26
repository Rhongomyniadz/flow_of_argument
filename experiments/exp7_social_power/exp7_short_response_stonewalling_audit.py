import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_VIOLATION_CSV = "experiments/exp7_social_power/results/exp7_violation_turns.csv"
DEFAULT_INPUT_DIR = "data/conversation_moves_labeled"
DEFAULT_OUTPUT_DIR = "experiments/exp7_social_power/results"

TARGET_TRIGGERS = {
    "underinformative_short_answer",
    "short_answer_after_question",
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
    r"^\s*(?:thanks?|thank you|appreciate it|take care|bye(?:-bye)?|goodbye|do it|sounds good|"
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
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        return int(float(value))
    except Exception:
        return default


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sort_turns(turns: Sequence[Dict]) -> List[Dict]:
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


def episode_id_from_turns(turns: Sequence[Dict], path: Path) -> str:
    if turns:
        first = turns[0]
        raw = first.get("episode_id")
        if raw not in (None, ""):
            return normalize_space(raw)
    match = re.search(r"(\d+)\.json$", path.name)
    if match:
        return match.group(1)
    return path.stem


def turn_text(turn: Optional[Dict]) -> str:
    if not turn:
        return ""
    return normalize_space(turn.get("turn_text") or turn.get("transcript") or "")


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def short_response_rows(violation_csv: Path, max_rows: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with violation_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("violation_trigger") or "").strip() not in TARGET_TRIGGERS:
                continue
            rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def discover_episode_files(input_dir: Path) -> List[Path]:
    direct_files = sorted(input_dir.glob("*.json"))
    return direct_files if direct_files else sorted(input_dir.glob("*/*.json"))


def build_episode_index(input_dir: Path, target_episode_ids: Sequence[str]) -> Dict[str, Tuple[Path, List[Dict]]]:
    remaining = set(str(ep) for ep in target_episode_ids)
    index: Dict[str, Tuple[Path, List[Dict]]] = {}

    for path in discover_episode_files(input_dir):
        if not remaining:
            break
        turns = load_turns(path)
        if not turns:
            continue
        episode_id = episode_id_from_turns(turns, path)
        if episode_id not in remaining:
            continue
        index[episode_id] = (path, turns)
        remaining.remove(episode_id)

    return index


def find_turn_position(turns: Sequence[Dict], target_turn_idx: int) -> Optional[int]:
    for pos, turn in enumerate(turns):
        if safe_int(turn.get("turn_idx"), pos) == target_turn_idx:
            return pos
    return None


def content_word_count(text: str) -> int:
    tokens = [tok.lower() for tok in CONTENT_WORD_RE.findall(text)]
    return sum(1 for tok in tokens if tok not in STOPWORDS)


def response_features(text: str) -> Dict[str, object]:
    lowered = text.lower()
    words = word_count(text)
    features = {
        "word_count": words,
        "has_stonewall_marker": bool(STONEWALL_RE.search(lowered)),
        "is_polar_answer": bool(POLAR_ANSWER_RE.match(lowered)),
        "is_ack_or_closing": bool(ACK_CLOSING_RE.match(lowered)),
        "content_word_count": content_word_count(text),
    }
    features["looks_contentful"] = bool(
        words > 0
        and not features["has_stonewall_marker"]
        and not features["is_ack_or_closing"]
        and features["content_word_count"] >= 1
    )
    return features


def question_features(previous_turns: Sequence[Dict], next_turn: Optional[Dict]) -> Dict[str, object]:
    prev_turn = previous_turns[-1] if previous_turns else None
    prev_text = turn_text(prev_turn)
    prev_lower = prev_text.lower()
    context_text = " ".join(turn_text(turn) for turn in previous_turns[-3:])
    context_lower = context_text.lower()
    next_text = turn_text(next_turn)
    next_lower = next_text.lower()

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
    rapid_fire_context = bool(
        "rapid fire" in context_lower
        or "favorite" in prev_lower
        or "finish this sentence" in context_lower
        or "one or two words" in context_lower
        or "one word" in context_lower
    )
    next_turn_continues_question_sequence = bool(
        "?" in next_text
        or BRIEF_EXPECTED_RE.search(next_lower)
        or re.search(r"^\s*(?:second|third|final question)\b", next_lower)
    )

    question_type = "other"
    if is_greeting_check:
        question_type = "greeting_check"
    elif is_audio_check:
        question_type = "audio_check"
    elif is_brief_expected or is_forced_choice:
        question_type = "brief_expected_prompt"
    elif is_polar:
        question_type = "polar_question"
    elif is_open_ended:
        question_type = "open_ended_prompt"

    return {
        "previous_text": prev_text,
        "question_type": question_type,
        "is_polar_question": is_polar,
        "is_brief_expected_prompt": bool(is_brief_expected or is_forced_choice),
        "is_open_ended_prompt": is_open_ended,
        "is_greeting_check": is_greeting_check,
        "is_audio_check": is_audio_check,
        "rapid_fire_context": rapid_fire_context,
        "next_turn_continues_question_sequence": next_turn_continues_question_sequence,
    }


def classify_short_response(
    question_meta: Dict[str, object],
    response_meta: Dict[str, object],
) -> Tuple[str, int, List[str]]:
    score = 0
    reasons: List[str] = []

    if bool(response_meta["has_stonewall_marker"]):
        score += 3
        reasons.append("response_contains_stonewall_marker")

    if bool(response_meta["is_ack_or_closing"]):
        score += 1
        reasons.append("response_is_ack_or_closing")

    if bool(question_meta["is_open_ended_prompt"]):
        score += 2
        reasons.append("prompt_is_open_ended")
        if int(response_meta["word_count"]) <= 3:
            score += 1
            reasons.append("very_short_reply_to_open_prompt")

    if bool(question_meta["is_brief_expected_prompt"]):
        score -= 3
        reasons.append("prompt_explicitly_expects_brief_answer")

    if bool(question_meta["rapid_fire_context"]):
        score -= 1
        reasons.append("rapid_fire_context")

    if bool(question_meta["is_greeting_check"]) or bool(question_meta["is_audio_check"]):
        score -= 2
        reasons.append("greeting_or_audio_check")

    if bool(question_meta["is_polar_question"]) and bool(response_meta["is_polar_answer"]):
        score -= 2
        reasons.append("direct_polar_answer")

    if bool(response_meta["looks_contentful"]):
        score -= 1
        reasons.append("contentful_short_answer")

    if bool(question_meta["next_turn_continues_question_sequence"]) and (
        bool(question_meta["is_brief_expected_prompt"]) or bool(question_meta["rapid_fire_context"])
    ):
        score -= 1
        reasons.append("next_turn_continues_question_sequence")

    if score >= 4:
        decision = "likely_stonewalling"
    elif score <= -2:
        decision = "likely_not_stonewalling"
    else:
        decision = "needs_manual_review"
    return decision, score, reasons


def audit_rows(
    short_rows: Sequence[Dict[str, str]],
    episode_index: Dict[str, Tuple[Path, List[Dict]]],
) -> List[Dict[str, object]]:
    audited: List[Dict[str, object]] = []

    for row in short_rows:
        episode_id = normalize_space(row.get("episode_id"))
        turn_idx = safe_int(row.get("turn_idx"), -1)
        episode_entry = episode_index.get(episode_id)

        source_path = ""
        previous_turn: Optional[Dict] = None
        current_turn: Optional[Dict] = None
        next_turn: Optional[Dict] = None
        previous_context: List[Dict] = []

        if episode_entry:
            source_path = str(episode_entry[0])
            turns = episode_entry[1]
            pos = find_turn_position(turns, turn_idx)
            if pos is not None:
                start = max(0, pos - 3)
                previous_context = turns[start:pos]
                previous_turn = turns[pos - 1] if pos > 0 else None
                current_turn = turns[pos]
                next_turn = turns[pos + 1] if pos + 1 < len(turns) else None

        current_text = turn_text(current_turn) or normalize_space(row.get("turn_text"))
        current_move = normalize_space((current_turn or {}).get("conversation_move_label") or row.get("conversation_move_label"))
        current_turn_type = normalize_space((current_turn or {}).get("turn_type_label") or "")
        next_text = turn_text(next_turn)
        next_move = normalize_space((next_turn or {}).get("conversation_move_label") or row.get("next_move"))
        next_turn_type = normalize_space((next_turn or {}).get("turn_type_label") or "")
        prev_text = turn_text(previous_turn)
        prev_move = normalize_space((previous_turn or {}).get("conversation_move_label") or "")
        prev_turn_type = normalize_space((previous_turn or {}).get("turn_type_label") or "")

        question_meta = question_features(previous_context, next_turn)
        response_meta = response_features(current_text)
        decision, score, reasons = classify_short_response(question_meta, response_meta)

        audited.append(
            {
                "episode_id": episode_id,
                "source_path": source_path,
                "turn_idx": turn_idx,
                "speaker_role": normalize_space(row.get("speaker_role")),
                "speaker_id": normalize_space(row.get("speaker_id")),
                "word_count": int(response_meta["word_count"]),
                "previous_turn_move": prev_move,
                "previous_turn_type": prev_turn_type,
                "previous_turn_text": prev_text,
                "current_turn_move": current_move,
                "current_turn_type": current_turn_type,
                "current_turn_text": current_text,
                "next_turn_move": next_move,
                "next_turn_type": next_turn_type,
                "next_turn_text": next_text,
                "question_type": str(question_meta["question_type"]),
                "is_polar_question": int(bool(question_meta["is_polar_question"])),
                "is_brief_expected_prompt": int(bool(question_meta["is_brief_expected_prompt"])),
                "is_open_ended_prompt": int(bool(question_meta["is_open_ended_prompt"])),
                "is_greeting_check": int(bool(question_meta["is_greeting_check"])),
                "is_audio_check": int(bool(question_meta["is_audio_check"])),
                "rapid_fire_context": int(bool(question_meta["rapid_fire_context"])),
                "next_turn_continues_question_sequence": int(bool(question_meta["next_turn_continues_question_sequence"])),
                "response_has_stonewall_marker": int(bool(response_meta["has_stonewall_marker"])),
                "response_is_polar_answer": int(bool(response_meta["is_polar_answer"])),
                "response_is_ack_or_closing": int(bool(response_meta["is_ack_or_closing"])),
                "response_content_word_count": int(response_meta["content_word_count"]),
                "response_looks_contentful": int(bool(response_meta["looks_contentful"])),
                "stonewalling_score": score,
                "audit_decision": decision,
                "audit_reasons": " | ".join(reasons),
                "manual_review_label": "",
                "manual_review_notes": "",
            }
        )

    audited.sort(
        key=lambda record: (
            {"likely_stonewalling": 0, "needs_manual_review": 1, "likely_not_stonewalling": 2}.get(
                str(record["audit_decision"]),
                9,
            ),
            -safe_int(record.get("stonewalling_score"), 0),
            str(record["episode_id"]),
            safe_int(record.get("turn_idx"), 0),
        )
    )
    return audited


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_summary(audited_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    decision_counts = Counter(str(row["audit_decision"]) for row in audited_rows)
    question_type_counts = Counter(str(row["question_type"]) for row in audited_rows)
    role_by_decision = Counter((str(row["speaker_role"]), str(row["audit_decision"])) for row in audited_rows)

    example_fields = [
        "episode_id",
        "turn_idx",
        "speaker_role",
        "question_type",
        "stonewalling_score",
        "current_turn_text",
        "previous_turn_text",
        "audit_reasons",
    ]

    examples: Dict[str, List[Dict[str, object]]] = {}
    for decision in ("likely_stonewalling", "needs_manual_review", "likely_not_stonewalling"):
        subset = [row for row in audited_rows if row["audit_decision"] == decision][:10]
        examples[decision] = [{field: row.get(field) for field in example_fields} for row in subset]

    return {
        "experiment": "exp7_social_power_short_response_stonewalling_audit",
        "target_triggers": sorted(TARGET_TRIGGERS),
        "n_rows_audited": len(audited_rows),
        "decision_counts": {key: int(val) for key, val in decision_counts.items()},
        "question_type_counts": {key: int(val) for key, val in question_type_counts.items()},
        "role_by_decision": {
            f"{role}:{decision}": int(count) for (role, decision), count in sorted(role_by_decision.items())
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the short-answer subset from Experiment 7 to check whether those rows "
            "look like genuine stonewalling or simply concise answers."
        )
    )
    parser.add_argument("--violation_csv", type=str, default=DEFAULT_VIOLATION_CSV)
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_rows", type=int, default=0, help="Optional debug cap on the number of short-answer rows.")
    args = parser.parse_args()

    violation_csv = Path(args.violation_csv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not violation_csv.exists():
        raise FileNotFoundError(f"Violation CSV does not exist: {violation_csv}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    short_rows = short_response_rows(violation_csv=violation_csv, max_rows=max(0, int(args.max_rows)))
    if not short_rows:
        raise RuntimeError(
            f"No rows with violation_trigger in {sorted(TARGET_TRIGGERS)!r} were found in {violation_csv}."
        )

    episode_ids = [normalize_space(row.get("episode_id")) for row in short_rows]
    episode_index = build_episode_index(input_dir=input_dir, target_episode_ids=episode_ids)
    audited_rows = audit_rows(short_rows=short_rows, episode_index=episode_index)
    summary = build_summary(audited_rows)

    audit_csv = output_dir / "exp7_short_response_stonewalling_audit.csv"
    summary_json = output_dir / "exp7_short_response_stonewalling_summary.json"

    write_csv(
        audit_csv,
        audited_rows,
        [
            "episode_id",
            "source_path",
            "turn_idx",
            "speaker_role",
            "speaker_id",
            "word_count",
            "previous_turn_move",
            "previous_turn_type",
            "previous_turn_text",
            "current_turn_move",
            "current_turn_type",
            "current_turn_text",
            "next_turn_move",
            "next_turn_type",
            "next_turn_text",
            "question_type",
            "is_polar_question",
            "is_brief_expected_prompt",
            "is_open_ended_prompt",
            "is_greeting_check",
            "is_audio_check",
            "rapid_fire_context",
            "next_turn_continues_question_sequence",
            "response_has_stonewall_marker",
            "response_is_polar_answer",
            "response_is_ack_or_closing",
            "response_content_word_count",
            "response_looks_contentful",
            "stonewalling_score",
            "audit_decision",
            "audit_reasons",
            "manual_review_label",
            "manual_review_notes",
        ],
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
