from __future__ import annotations

"""Stage 00: prepare the show-disjoint pilot data."""

import argparse
import math
from pathlib import Path
from typing import Any


# Stage-local helper functions (utils).
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def list_episode_paths(input_dir: Path) -> list[Path]:
    direct = sorted(input_dir.glob("*.json"))
    return direct if direct else sorted(input_dir.glob("*/*.json"))


def patch_name(index: int, total: int) -> str:
    return f"patch_{index:04d}_of_{total:04d}"


def patch_directory(root: Path, index: int, total: int) -> Path:
    return root / "patches" / patch_name(index, total)


def shard_slice(total_items: int, patch_index: int, items_per_patch: int) -> slice:
    if patch_index < 0 or items_per_patch < 1:
        raise ValueError("patch_index must be non-negative and items_per_patch must be positive")
    start = patch_index * items_per_patch
    return slice(start, min(start + items_per_patch, total_items))


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "torch", "sentence_transformers", "sklearn"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "not-installed"
    return versions


def make_manifest(
    *,
    stage: str,
    patch_index: int,
    num_patches: int,
    row_count: int,
    input_hash: str,
    split_hash: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "patch_index": patch_index,
        "num_patches": num_patches,
        "row_count": row_count,
        "input_hash": input_hash,
        "split_hash": split_hash,
        "config": config,
        "config_hash": stable_hash(config),
        "complete": True,
        "runtime": runtime_versions(),
    }
    if extra:
        payload.update(extra)
    return payload


def manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        observed = read_json(path)
    except Exception:
        return False
    keys = ("stage", "patch_index", "num_patches", "input_hash", "split_hash", "config_hash")
    return bool(observed.get("complete")) and all(observed.get(key) == expected.get(key) for key in keys)


def validate_patch_manifests(root: Path, stage: str, num_patches: int) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for index in range(num_patches):
        path = patch_directory(root, index, num_patches) / "patch_manifest.json"
        if not path.exists():
            raise RuntimeError(f"Missing patch manifest: {path}")
        manifest = read_json(path)
        if not manifest.get("complete"):
            raise RuntimeError(f"Incomplete patch: {path}")
        if manifest.get("stage") != stage:
            raise RuntimeError(f"Stage mismatch in {path}: {manifest.get('stage')} != {stage}")
        if int(manifest.get("patch_index", -1)) != index:
            raise RuntimeError(f"Patch index mismatch in {path}")
        if int(manifest.get("num_patches", -1)) != num_patches:
            raise RuntimeError(f"Patch-count mismatch in {path}")
        manifests.append(manifest)
    for key in ("input_hash", "split_hash", "config_hash"):
        values = {manifest.get(key) for manifest in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Mixed {key} values across {stage} patches: {sorted(values)}")
    return manifests


# Stage-local progress display.
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


# Stage-local helper functions (data).
import csv
import json
import re
from pathlib import Path
from typing import Any


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


# Stage-local helper functions (splits).
from collections import defaultdict
from typing import Any

import numpy as np



def assign_show_splits(episodes: list[dict[str, Any]], seed: int) -> dict[str, str]:
    by_category: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        by_category[str(episode["category"])].add(str(episode["show_id"]))

    assignments: dict[str, str] = {}
    for category in sorted(by_category):
        unresolved = [show for show in sorted(by_category[category]) if show not in assignments]
        unresolved.sort(key=lambda show: stable_hash({"seed": seed, "category": category, "show": show}))
        count = len(unresolved)
        if count == 1:
            cuts = (1, 1)
        elif count == 2:
            cuts = (1, 2)
        else:
            # Keep at least one validation and one test show, including the
            # common three-show pilot case.
            train_end = min(max(1, int(np.floor(count * 0.8))), count - 2)
            validation_end = count - 1
            cuts = (train_end, validation_end)
        for index, show in enumerate(unresolved):
            if index < cuts[0]:
                split = "train"
            elif index < cuts[1]:
                split = "validation"
            else:
                split = "test"
            assignments[show] = split
    return assignments


def assert_show_disjoint(rows: list[dict[str, Any]]) -> None:
    observed: dict[str, str] = {}
    for row in rows:
        show = str(row["show_id"])
        split = str(row["split"])
        previous = observed.setdefault(show, split)
        if previous != split:
            raise ValueError(f"Show {show} appears in both {previous} and {split}")


def balanced_anchor_sample(
    anchors: list[dict[str, Any]],
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [row for row in anchors if row["split"] == split]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_category[str(row["category"])].append(row)
    rng = np.random.default_rng(seed)
    for rows in by_category.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    categories = sorted(by_category)
    while len(selected) < min(limit, len(eligible)):
        progressed = False
        for category in categories:
            if by_category[category]:
                selected.append(by_category[category].pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda row: row["anchor_id"])


# Stage-local helper functions (candidates).
from collections import defaultdict
from typing import Any

import numpy as np



def build_anchors(
    episodes: list[dict[str, Any]],
    splits: dict[str, str],
    candidate_count: int = 25,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    all_turns = [turn for episode in episodes for turn in episode["turns"]]
    lookup = {turn["turn_id"]: turn for turn in all_turns}
    by_episode: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    global_ids: list[str] = []
    for turn in all_turns:
        by_episode[str(turn["episode_id"])].append(turn["turn_id"])
        by_category[str(turn["category"])].append(turn["turn_id"])
        global_ids.append(turn["turn_id"])

    anchors: list[dict[str, Any]] = []
    def sample_pool(
        pool: list[str],
        *,
        rng: np.random.Generator,
        used: set[str],
        limit: int,
        predicate: Any | None = None,
    ) -> list[str]:
        if not pool or limit < 1:
            return []
        selected: list[str] = []
        attempts = min(max(limit * 50, 200), max(len(pool) * 2, 200))
        for _ in range(attempts):
            turn_id = pool[int(rng.integers(0, len(pool)))]
            if turn_id in used or (predicate is not None and not predicate(turn_id)):
                continue
            used.add(turn_id)
            selected.append(turn_id)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            for turn_id in pool:
                if turn_id in used or (predicate is not None and not predicate(turn_id)):
                    continue
                used.add(turn_id)
                selected.append(turn_id)
                if len(selected) >= limit:
                    break
        return selected

    episode_iterator = tqdm(episodes, desc="stage 00 anchors", unit="episode", dynamic_ncols=True) if show_progress else episodes
    for episode in episode_iterator:
        turns = episode["turns"]
        for position, turn in enumerate(turns[:-1]):
            target = turns[position + 1]
            if not turn["turn_text"] or not target["turn_text"]:
                continue
            anchor_id = turn["turn_id"]
            rng = np.random.default_rng(int(stable_hash(anchor_id, 16), 16))
            used = {anchor_id, target["turn_id"]}
            candidates: list[str] = [target["turn_id"]]

            same_episode = [
                turn_id
                for turn_id in by_episode[str(turn["episode_id"])]
                if turn_id not in used and abs(int(lookup[turn_id]["turn_idx"]) - int(turn["turn_idx"])) >= 3
            ]
            rng.shuffle(same_episode)
            for turn_id in same_episode[:8]:
                candidates.append(turn_id)
                used.add(turn_id)

            candidates.extend(
                sample_pool(
                    by_category[str(turn["category"])],
                    rng=rng,
                    used=used,
                    limit=max(0, min(candidate_count, 17) - len(candidates)),
                    predicate=lambda turn_id: lookup[turn_id]["show_id"] != turn["show_id"],
                )
            )
            candidates.extend(
                sample_pool(
                    global_ids,
                    rng=rng,
                    used=used,
                    limit=max(0, candidate_count - len(candidates)),
                )
            )
            if len(candidates) < 2:
                continue
            rng.shuffle(candidates)
            if candidates.count(target["turn_id"]) != 1 or len(candidates) != len(set(candidates)):
                raise AssertionError(f"Invalid candidate pool for {anchor_id}")
            history_ids = [item["turn_id"] for item in turns[max(0, position - 3):position]]
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "target_id": target["turn_id"],
                    "candidate_ids": candidates,
                    "history_ids": history_ids,
                    "category": turn["category"],
                    "show_id": turn["show_id"],
                    "episode_id": turn["episode_id"],
                    "turn_idx": turn["turn_idx"],
                    "split": splits[str(turn["show_id"])],
                    "assumption_count": turn["assumption_count"],
                    "assumption_token_count": turn["assumption_token_count"],
                }
            )
    return anchors


def validate_anchor(anchor: dict[str, Any]) -> None:
    candidates = list(anchor["candidate_ids"])
    target = anchor["target_id"]
    if candidates.count(target) != 1:
        raise ValueError(f"Anchor {anchor['anchor_id']} has {candidates.count(target)} true candidates")
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"Anchor {anchor['anchor_id']} has duplicate candidates")
    if anchor["anchor_id"] in candidates:
        raise ValueError(f"Anchor {anchor['anchor_id']} appears in its own candidates")


STAGE = "exp8_prepare_data"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Prepare the show-disjoint Exp8 pilot data.")
    value.add_argument("--input-dir", type=Path, default=Path("data/conversation_moves_labeled"))
    value.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"),
    )
    value.add_argument("--show-map", type=Path)
    value.add_argument("--allow-episode-fallback", action="store_true")
    value.add_argument("--candidate-count", type=int, default=25)
    value.add_argument("--development-limit", type=int, default=10000)
    value.add_argument("--seed", type=int, default=42)
    return value


def prepare(args: argparse.Namespace) -> None:
    paths = list_episode_paths(args.input_dir)
    if not paths:
        raise RuntimeError(f"No episode JSON files found under {args.input_dir}")

    show_map = load_show_map(args.show_map)
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in tqdm(paths, desc="stage 00 prepare", unit="episode", dynamic_ncols=True):
        try:
            episodes.append(
                normalize_episode(
                    path,
                    show_map=show_map,
                    allow_episode_fallback=args.allow_episode_fallback,
                )
            )
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})

    if not episodes:
        detail = errors[0]["error"] if errors else "unknown error"
        raise RuntimeError(f"No episodes could be normalized; first error: {detail}")

    assignments = assign_show_splits(episodes, args.seed)
    turns = [turn for episode in episodes for turn in episode["turns"]]
    anchors = build_anchors(episodes, assignments, args.candidate_count, show_progress=True)
    for anchor in anchors:
        validate_anchor(anchor)
    assert_show_disjoint(anchors)
    development = balanced_anchor_sample(
        anchors,
        split="validation",
        limit=args.development_limit,
        seed=args.seed,
    )
    split_rows = [
        {
            "show_id": show_id,
            "split": split,
            "split_hash_key": stable_hash({"show_id": show_id, "seed": args.seed}),
        }
        for show_id, split in sorted(assignments.items())
    ]
    split_hash = stable_hash(split_rows)
    input_hash = stable_hash(
        [
            {
                "path": str(path.relative_to(args.input_dir)),
                "size": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "episodes.jsonl", episodes)
    write_jsonl(args.output_dir / "turns.jsonl", turns)
    write_jsonl(args.output_dir / "anchors.jsonl", anchors)
    write_jsonl(args.output_dir / "development_anchors.jsonl", development)
    write_jsonl(args.output_dir / "split_manifest.jsonl", split_rows)
    write_jsonl(args.output_dir / "normalization_errors.jsonl", errors)
    write_json(
        args.output_dir / "summary.json",
        {
            "stage": STAGE,
            "input_hash": input_hash,
            "split_hash": split_hash,
            "config": {
                "input_dir": str(args.input_dir),
                "show_map": str(args.show_map) if args.show_map else None,
                "allow_episode_fallback": bool(args.allow_episode_fallback),
                "candidate_count": args.candidate_count,
                "development_limit": args.development_limit,
                "seed": args.seed,
            },
            "episode_count": len(episodes),
            "turn_count": len(turns),
            "anchor_count": len(anchors),
            "development_anchor_count": len(development),
            "show_count": len(assignments),
            "error_count": len(errors),
            "split_counts": {
                split: sum(1 for value in assignments.values() if value == split)
                for split in ("train", "validation", "test")
            },
        },
    )


def main() -> None:
    args = parser().parse_args()
    if args.candidate_count < 2 or args.development_limit < 1:
        raise ValueError("--candidate-count must be at least 2 and --development-limit must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
