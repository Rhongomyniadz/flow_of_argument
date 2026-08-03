from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .utils import list_episode_paths, stable_hash

_WORD_RE = re.compile(r"\w+")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def normalize_text_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            confidence = None
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            confidence = item.get("confidence")
        else:
            continue
        key = " ".join(text.casefold().split())
        if not text or key in seen:
            continue
        seen.add(key)
        try:
            numeric_confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            numeric_confidence = None
        rows.append({"text": text, "confidence": numeric_confidence})
    return rows


def turn_text(turn: dict[str, Any]) -> str:
    for key in ("turn_text", "transcript", "text"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    explicit = normalize_text_items(turn.get("explicit_propositions"))
    return " ".join(item["text"] for item in explicit).strip()


def load_turns(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        turns = value
    elif isinstance(value, dict) and isinstance(value.get("turns"), list):
        turns = value["turns"]
    else:
        raise ValueError(f"Unrecognized episode JSON structure: {path}")
    normalized = [turn for turn in turns if isinstance(turn, dict)]
    normalized.sort(key=lambda row: _safe_int(row.get("turn_idx"), len(normalized)))
    return normalized


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_show_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    mapping: dict[str, str] = {}
    if path.suffix.lower() in {".json", ".jsonl"}:
        value = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
        rows = value if isinstance(value, list) else []
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        for row in rows:
            if isinstance(row, dict) and row.get("episode_id") is not None and row.get("show_id") is not None:
                mapping[str(row["episode_id"])] = str(row["show_id"])
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            episode_id = row.get("episode_id")
            show_id = row.get("show_id") or row.get("rssKey") or row.get("rss_key")
            if episode_id and show_id:
                mapping[str(episode_id)] = str(show_id)
    return mapping


def resolve_show_id(
    turns: list[dict[str, Any]],
    episode_id: str,
    show_map: dict[str, str],
    allow_episode_fallback: bool,
) -> str:
    for turn in turns:
        for key in ("show_id", "rssKey", "rss_key", "podcast_id", "feed_id"):
            value = turn.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    if episode_id in show_map:
        return show_map[episode_id]
    if allow_episode_fallback:
        return f"episode:{episode_id}"
    raise ValueError(
        f"No show identity for episode {episode_id}. Supply --show-map or explicitly use --allow-episode-fallback."
    )


def normalize_episode(
    path: Path,
    *,
    show_map: dict[str, str],
    allow_episode_fallback: bool,
) -> dict[str, Any]:
    turns = load_turns(path)
    if not turns:
        raise ValueError(f"Episode has no usable turns: {path}")
    episode_id = str(turns[0].get("episode_id") or path.stem)
    category = str(turns[0].get("category") or path.parent.name or "unknown").strip().casefold()
    show_id = resolve_show_id(turns, episode_id, show_map, allow_episode_fallback)
    normalized_turns: list[dict[str, Any]] = []
    for fallback_idx, turn in enumerate(turns):
        index = _safe_int(turn.get("turn_idx"), fallback_idx)
        text = turn_text(turn)
        explicit = normalize_text_items(turn.get("explicit_propositions"))
        assumptions = normalize_text_items(turn.get("assumptions"))
        turn_id = f"{category}:{episode_id}:{index}"
        normalized_turns.append(
            {
                "turn_id": turn_id,
                "category": category,
                "show_id": show_id,
                "episode_id": episode_id,
                "turn_idx": index,
                "speaker_id": str(turn.get("speaker_id") or turn.get("speaker") or ""),
                "turn_text": text,
                "explicit": explicit,
                "assumptions": assumptions,
                "explicit_count": len(explicit),
                "assumption_count": len(assumptions),
                "explicit_token_count": sum(word_count(item["text"]) for item in explicit),
                "assumption_token_count": sum(word_count(item["text"]) for item in assumptions),
                "conversation_move_label": str(turn.get("conversation_move_label") or ""),
            }
        )
    normalized_turns.sort(key=lambda row: int(row["turn_idx"]))
    return {
        "episode_key": f"{category}:{episode_id}",
        "category": category,
        "show_id": show_id,
        "episode_id": episode_id,
        "source_path": str(path),
        "source_hash": stable_hash({"path": str(path), "size": path.stat().st_size, "mtime": path.stat().st_mtime_ns}),
        "turns": normalized_turns,
    }


def episode_input_hash(input_dir: Path) -> str:
    paths = list_episode_paths(input_dir)
    return stable_hash([(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths])

