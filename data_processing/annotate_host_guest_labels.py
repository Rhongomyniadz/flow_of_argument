import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


HOST_CUE_RE = re.compile(
    r"\b(i\s*[' ]?m|i am|this is)\s+(?:your\s+)?host\b|"
    r"\bour guest\b|"
    r"\bjoining me\b|"
    r"\bjoined by\b|"
    r"\bwelcome (?:back )?to\b",
    re.IGNORECASE,
)
INTRO_RE = re.compile(r"\b(i\s*[' ]?m|i am|this is)\b", re.IGNORECASE)
GUEST_ACK_RE = re.compile(
    r"\b(thanks|thank you)\b.{0,20}\b(having me|invitation|having us)\b|"
    r"\bgood to be with you\b|"
    r"\bgreat to be (here|with you)\b",
    re.IGNORECASE,
)

PERSON_STOPWORDS = {
    "host",
    "podcast",
    "podcasts",
    "network",
    "radio",
    "news",
    "institute",
    "official",
    "show",
    "episode",
    "the",
    "and",
    "of",
    "none",
    "email",
    "com",
    "org",
    "www",
    "fm",
    "anchor",
    "itunes",
    "apple",
    "spotify",
    "audacy",
    "hudson",
}
TITLE_WORDS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "rep",
    "representative",
    "sen",
    "senator",
    "congressman",
    "congresswoman",
    "gov",
    "governor",
    "president",
    "fr",
    "father",
}


def safe_int_episode_id(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        return int(float(value))
    except Exception:
        return None


def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_word_mentions(text: str, token: str) -> int:
    if not token:
        return 0
    return len(re.findall(rf"\b{re.escape(token)}\b", text))


def split_name_tokens(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    words = re.findall(r"[a-z]+", text)
    out: List[str] = []
    for word in words:
        if word in TITLE_WORDS or word in PERSON_STOPWORDS:
            continue
        if len(word) <= 1:
            continue
        out.append(word)
    return out


def uniq_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_host_names_from_text(text: str) -> List[str]:
    text = text or ""
    patterns = [
        r"hosted by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"host\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"i\s*'?m your host\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
    ]
    names: List[str] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            names.append(m.group(1))
    return names


def load_metadata(path: Path) -> Dict[int, Dict]:
    metadata: Dict[int, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ep_id = safe_int_episode_id(row.get("dbId"))
            if ep_id is None:
                continue
            if ep_id not in metadata:
                metadata[ep_id] = row
    return metadata


def build_name_tokens(meta: Optional[Dict]) -> Tuple[List[str], List[str]]:
    if not meta:
        return [], []

    guest_tokens: List[str] = []
    for key in ("polGuestName", "name"):
        val = meta.get(key)
        if isinstance(val, str):
            guest_tokens.extend(split_name_tokens(val))
    guest_tokens = uniq_preserve(guest_tokens)

    host_sources: List[str] = []
    for key in ("author", "author_ep", "author_lower_x", "author_lower_y"):
        val = meta.get(key)
        if isinstance(val, str):
            host_sources.append(val)

    owner = meta.get("owner_pod")
    if isinstance(owner, dict):
        for k, v in owner.items():
            if isinstance(v, str) and ("name" in str(k).lower()):
                host_sources.append(v)

    for key in ("description_ep", "description_pod", "description", "title_pod"):
        val = meta.get(key)
        if isinstance(val, str):
            host_sources.extend(extract_host_names_from_text(val))

    host_tokens: List[str] = []
    for text in host_sources:
        host_tokens.extend(split_name_tokens(text))
    host_tokens = [t for t in host_tokens if t not in set(guest_tokens)]
    host_tokens = uniq_preserve(host_tokens)
    return guest_tokens, host_tokens


def speaker_feature_rows(
    turns: List[Dict],
    guest_tokens: List[str],
    host_tokens: List[str],
) -> Dict[str, Dict]:
    by_speaker: Dict[str, Dict] = {}
    for turn in turns:
        speaker = str(turn.get("speaker_id") or turn.get("speaker") or "").strip()
        if not speaker:
            continue
        row = by_speaker.setdefault(
            speaker,
            {
                "speaker_id": speaker,
                "n_turns": 0,
                "total_words": 0,
                "question_turns": 0,
                "host_cue_turns": 0,
                "intro_turns": 0,
                "guest_ack_turns": 0,
                "guest_name_mentions": 0,
                "host_name_mentions": 0,
            },
        )
        text = str(turn.get("turn_text") or turn.get("transcript") or "")
        text_norm = normalize_text(text)
        words = len(text.split())
        row["n_turns"] += 1
        row["total_words"] += words
        if "?" in text:
            row["question_turns"] += 1
        if HOST_CUE_RE.search(text):
            row["host_cue_turns"] += 1
        if INTRO_RE.search(text):
            row["intro_turns"] += 1
        if GUEST_ACK_RE.search(text):
            row["guest_ack_turns"] += 1
        row["guest_name_mentions"] += sum(count_word_mentions(text_norm, tok) for tok in guest_tokens if len(tok) >= 3)
        row["host_name_mentions"] += sum(count_word_mentions(text_norm, tok) for tok in host_tokens if len(tok) >= 3)

    for row in by_speaker.values():
        n = max(1, int(row["n_turns"]))
        row["avg_words_per_turn"] = float(row["total_words"]) / n
        row["question_ratio"] = float(row["question_turns"]) / n
    return by_speaker


def infer_host_speaker(
    turns: List[Dict],
    meta: Optional[Dict],
) -> Tuple[Optional[str], Dict]:
    guest_tokens, host_tokens = build_name_tokens(meta)
    feats = speaker_feature_rows(turns, guest_tokens, host_tokens)
    if not feats:
        return None, {
            "method": "no_speaker",
            "confidence": 0.0,
            "guest_tokens": guest_tokens,
            "host_tokens": host_tokens,
            "speaker_scores": {},
        }

    speakers = list(feats.keys())
    host_cue_speakers = [sp for sp in speakers if feats[sp]["host_cue_turns"] > 0]
    if len(host_cue_speakers) == 1:
        host_id = host_cue_speakers[0]
        method = "direct_host_cue"
        confidence = 1.0
        speaker_scores = {sp: float(feats[sp]["host_cue_turns"]) for sp in speakers}
        return host_id, {
            "method": method,
            "confidence": confidence,
            "guest_tokens": guest_tokens,
            "host_tokens": host_tokens,
            "speaker_scores": speaker_scores,
            "speaker_features": feats,
        }

    scores: Dict[str, float] = {}
    for sp, row in feats.items():
        n_turns = float(row["n_turns"])
        q_ratio = float(row["question_ratio"])
        avg_words = float(row["avg_words_per_turn"])
        guest_mentions = float(row["guest_name_mentions"])
        host_mentions = float(row["host_name_mentions"])
        host_cues = float(row["host_cue_turns"])
        intros = float(row["intro_turns"])
        guest_ack = float(row["guest_ack_turns"])

        score = 0.0
        score += 4.5 * host_cues
        score += 1.9 * q_ratio
        score += 0.9 * math.log1p(guest_mentions)
        score += 0.6 * intros
        score += 0.5 * math.log1p(host_mentions)
        score += 0.15 * math.log1p(n_turns)
        score -= 1.3 * math.log1p(guest_ack)
        score -= 0.002 * avg_words
        scores[sp] = float(score)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    host_id = sorted_scores[0][0]
    if len(sorted_scores) == 1:
        margin = 5.0
    else:
        margin = float(sorted_scores[0][1] - sorted_scores[1][1])

    if host_cue_speakers:
        method = "host_cue_tiebreak"
    else:
        method = "heuristic_score"
    confidence = max(0.05, min(0.99, 0.50 + 0.12 * margin))

    return host_id, {
        "method": method,
        "confidence": confidence,
        "guest_tokens": guest_tokens,
        "host_tokens": host_tokens,
        "speaker_scores": scores,
        "speaker_features": feats,
    }


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_input_files(input_dir: Path) -> List[Path]:
    direct_files = sorted(input_dir.glob("*.json"))
    if direct_files:
        return direct_files
    return sorted(input_dir.glob("*/*.json"))


def annotate_directory(
    input_dir: Path,
    metadata_by_ep: Dict[int, Dict],
    dry_run: bool = False,
    summary_out: Optional[Path] = None,
    mapping_csv_out: Optional[Path] = None,
) -> Dict:
    files = discover_input_files(input_dir)
    summary_rows: List[Dict] = []
    method_counts: Counter = Counter()
    confidence_vals: List[float] = []
    role_counts: Counter = Counter()
    missing_meta_eps: List[int] = []

    for fp in files:
        try:
            turns = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(turns, list) or not turns:
            continue

        ep_id = safe_int_episode_id(turns[0].get("episode_id")) or safe_int_episode_id(fp.stem)
        meta = metadata_by_ep.get(ep_id) if ep_id is not None else None
        if ep_id is not None and meta is None:
            missing_meta_eps.append(ep_id)

        host_id, detail = infer_host_speaker(turns, meta)
        method = str(detail.get("method", "unknown"))
        confidence = float(detail.get("confidence", 0.0))
        method_counts[method] += 1
        confidence_vals.append(confidence)

        speakers = sorted({str(t.get("speaker_id") or t.get("speaker") or "").strip() for t in turns if (t.get("speaker_id") or t.get("speaker"))})
        if host_id is None and speakers:
            host_id = speakers[0]
            method = "fallback_first_speaker"
            confidence = 0.05
            method_counts[method] += 1
            confidence_vals.append(confidence)

        for turn in turns:
            speaker = str(turn.get("speaker_id") or turn.get("speaker") or "").strip()
            role = "host" if host_id is not None and speaker == host_id else "guest"
            turn["speaker_role"] = role
            role_counts[role] += 1

        if not dry_run:
            write_json(fp, turns)

        summary_rows.append(
            {
                "episode_id": ep_id if ep_id is not None else fp.stem,
                "file": str(fp),
                "host_speaker_id": host_id,
                "method": method,
                "confidence": round(confidence, 4),
                "num_turns": len(turns),
                "num_speakers": len(speakers),
                "guest_tokens": " ".join(detail.get("guest_tokens", [])),
                "host_tokens": " ".join(detail.get("host_tokens", [])),
            }
        )

    summary = {
        "input_dir": str(input_dir),
        "num_files_processed": len(summary_rows),
        "num_missing_metadata": len(set(missing_meta_eps)),
        "method_counts": dict(method_counts),
        "avg_confidence": float(sum(confidence_vals) / max(1, len(confidence_vals))),
        "role_counts": dict(role_counts),
    }

    if summary_out is not None:
        write_json(summary_out, summary)

    if mapping_csv_out is not None:
        fieldnames = [
            "episode_id",
            "file",
            "host_speaker_id",
            "method",
            "confidence",
            "num_turns",
            "num_speakers",
            "guest_tokens",
            "host_tokens",
        ]
        with mapping_csv_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate each turn with speaker_role=host/guest.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/conversation_moves_labeled",
        help="Directory with episode JSON files to annotate.",
    )
    parser.add_argument(
        "--metadata_jsonl",
        type=str,
        default="/tmp/polEpsDataCleaned3_ids.jsonl",
        help="Path to polEpsDataCleaned3_ids.jsonl.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="data/conversation_moves_labeled_host_guest_summary.json",
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--episode_mapping_csv",
        type=str,
        default="data/conversation_moves_labeled_host_guest_episode_mapping.csv",
        help="Per-episode mapping CSV output.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Compute mapping but do not write label into turns.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata_path = Path(args.metadata_jsonl)
    summary_json = Path(args.summary_json)
    mapping_csv = Path(args.episode_mapping_csv)

    metadata_by_ep = load_metadata(metadata_path)
    summary = annotate_directory(
        input_dir=input_dir,
        metadata_by_ep=metadata_by_ep,
        dry_run=args.dry_run,
        summary_out=summary_json,
        mapping_csv_out=mapping_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
