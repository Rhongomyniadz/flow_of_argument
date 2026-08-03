from __future__ import annotations

"""Stage 06: train the nine mini-fusion GPU conditions."""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


# Stage-local helper functions (metrics).
from collections import defaultdict
from typing import Any

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def compose(vectors: list[np.ndarray], masks: list[bool] | None = None) -> np.ndarray:
    if masks is None:
        masks = [True] * len(vectors)
    selected = [normalize(vector) for vector, use in zip(vectors, masks) if use]
    if not selected:
        return np.zeros_like(np.asarray(vectors[0], dtype=np.float32))
    return normalize(np.mean(np.stack(selected), axis=0))


def rank_scores(candidate_ids: list[str], scores: np.ndarray, target_id: str) -> dict[str, float | int]:
    if candidate_ids.count(target_id) != 1:
        raise ValueError(f"Expected exactly one target {target_id}")
    array = np.asarray(scores, dtype=float)
    if array.shape != (len(candidate_ids),):
        raise ValueError("Score shape does not match candidate IDs")
    target_index = candidate_ids.index(target_id)
    order = sorted(range(len(candidate_ids)), key=lambda index: (-array[index], candidate_ids[index]))
    rank = order.index(target_index) + 1
    hardest_negative = max(array[index] for index in range(len(candidate_ids)) if index != target_index)
    return {
        "rank": rank,
        "reciprocal_rank": 1.0 / rank,
        "top1": int(rank == 1),
        "top5": int(rank <= 5),
        "margin": float(array[target_index] - hardest_negative),
    }


def aggregate_rows(rows: list[dict[str, Any]], condition_key: str = "condition") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[condition_key])].append(row)
    output: list[dict[str, Any]] = []
    for condition, group in sorted(groups.items()):
        output.append(
            {
                condition_key: condition,
                "n": len(group),
                "recall_at_1": float(np.mean([float(row["top1"]) for row in group])),
                "recall_at_5": float(np.mean([float(row["top5"]) for row in group])),
                "mrr": float(np.mean([float(row["reciprocal_rank"]) for row in group])),
                "mean_margin": float(np.mean([float(row["margin"]) for row in group])),
            }
        )
    return output


def clustered_delta_interval(
    rows: list[dict[str, Any]],
    first: str,
    second: str,
    metric: str = "reciprocal_rank",
    draws: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[str(row["anchor_id"])][str(row["condition"])] = row
    deltas_by_show: dict[str, list[float]] = defaultdict(list)
    for condition_rows in paired.values():
        if first in condition_rows and second in condition_rows:
            left = condition_rows[first]
            right = condition_rows[second]
            deltas_by_show[str(left["show_id"])].append(float(left[metric]) - float(right[metric]))
    if not deltas_by_show:
        return {"first": first, "second": second, "n_shows": 0, "mean_delta": float("nan")}
    show_ids = np.asarray(sorted(deltas_by_show), dtype=object)
    show_means = np.asarray([np.mean(deltas_by_show[show]) for show in show_ids], dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(1, draws)):
        indices = rng.integers(0, len(show_means), size=len(show_means))
        samples.append(float(np.mean(show_means[indices])))
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "n_shows": len(show_ids),
        "mean_delta": float(np.mean(show_means)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(np.asarray(samples) > 0)),
    }


# Stage-local helper functions (controls).
from collections import defaultdict
from bisect import bisect_left
from typing import Any



def _distance(source: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, int, str]:
    return (
        abs(int(source.get("assumption_count", 0)) - int(candidate.get("assumption_count", 0))),
        abs(int(source.get("assumption_token_count", 0)) - int(candidate.get("assumption_token_count", 0))),
        str(candidate["turn_id"]),
    )


def build_control_map(
    turns: list[dict[str, Any]],
    control_type: str,
    source_ids: set[str] | None = None,
) -> dict[str, str]:
    if control_type not in {"same_episode", "same_category", "explicit_matched"}:
        raise ValueError(f"Unknown control type: {control_type}")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        by_episode[str(turn["episode_id"])].append(turn)
        by_category[str(turn["category"])].append(turn)
    for rows in by_category.values():
        rows.sort(key=lambda row: (int(row.get("assumption_token_count", 0)), str(row["turn_id"])))
    category_tokens = {
        category: [int(row.get("assumption_token_count", 0)) for row in rows]
        for category, rows in by_category.items()
    }
    mapping: dict[str, str] = {}
    sources = turns if source_ids is None else [row for row in turns if str(row["turn_id"]) in source_ids]
    for source in sources:
        if control_type == "explicit_matched":
            if int(source.get("explicit_count", 0)) > 0:
                mapping[source["turn_id"]] = source["turn_id"]
            continue
        if control_type == "same_episode":
            candidates = [
                row for row in by_episode[str(source["episode_id"])]
                if row["turn_id"] != source["turn_id"]
                and abs(int(row["turn_idx"]) - int(source["turn_idx"])) >= 3
                and int(row.get("assumption_count", 0)) > 0
            ]
        else:
            rows = by_category[str(source["category"])]
            tokens = category_tokens[str(source["category"])]
            center = bisect_left(tokens, int(source.get("assumption_token_count", 0)))
            radius = 64
            candidates = [
                row for row in rows[max(0, center - radius):min(len(rows), center + radius)]
                if row["show_id"] != source["show_id"] and int(row.get("assumption_count", 0)) > 0
            ]
            if not candidates:
                candidates = [
                    row for row in rows
                    if row["show_id"] != source["show_id"] and int(row.get("assumption_count", 0)) > 0
                ][:128]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (_distance(source, row), stable_hash({"source": source["turn_id"], "candidate": row["turn_id"]})))
        mapping[source["turn_id"]] = candidates[0]["turn_id"]
    return mapping


# Stage-local helper functions (embeddings).
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_INSTRUCTION = "Represent this conversational state for retrieving the next speaker turn."


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str = "main",
        backend: str = "sentence_transformer",
        instruction: str = DEFAULT_INSTRUCTION,
        batch_size: int = 32,
        device: str | None = None,
        hash_dim: int = 64,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.backend = backend
        self.instruction = instruction
        self.batch_size = batch_size
        self.device = device
        self.hash_dim = hash_dim
        self._model: Any = None
        if backend not in {"sentence_transformer", "hash"}:
            raise ValueError(f"Unsupported embedding backend: {backend}")

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError("sentence-transformers is required for the production embedding backend") from error
            kwargs: dict[str, Any] = {"trust_remote_code": True, "revision": self.model_revision}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _hash_embedding(self, text: str, query: bool) -> np.ndarray:
        seed_text = f"{self.instruction if query else 'document'}\n{text}"
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        return normalize(rng.standard_normal(self.hash_dim).astype(np.float32))

    def encode(self, texts: Iterable[str], *, query: bool) -> np.ndarray:
        values = [str(text) for text in texts]
        if not values:
            dimension = self.hash_dim if self.backend == "hash" else 0
            return np.zeros((0, dimension), dtype=np.float32)
        if self.backend == "hash":
            return np.stack([self._hash_embedding(text, query) for text in values])
        model = self._load()
        prepared = [f"Instruct: {self.instruction}\nQuery: {text}" for text in values] if query else values
        array = model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(array, dtype=np.float32)

    def pool_statements(self, statements: list[dict[str, Any]]) -> tuple[np.ndarray, bool]:
        texts = [str(item["text"]) for item in statements if str(item.get("text") or "").strip()]
        if not texts:
            dimension = self.hash_dim if self.backend == "hash" else int(self._load().get_sentence_embedding_dimension())
            return np.zeros(dimension, dtype=np.float32), False
        return normalize(np.mean(self.encode(texts, query=True), axis=0)), True

    def matched_explicit(self, explicit: list[dict[str, Any]], target_words: int) -> tuple[np.ndarray, bool]:
        texts = [str(item["text"]).strip() for item in explicit if str(item.get("text") or "").strip()]
        if not texts or target_words < 1:
            dimension = self.hash_dim if self.backend == "hash" else int(self._load().get_sentence_embedding_dimension())
            return np.zeros(dimension, dtype=np.float32), False
        words: list[str] = []
        index = 0
        while len(words) < target_words:
            words.extend(texts[index % len(texts)].split())
            index += 1
        text = " ".join(words[:target_words])
        return self.encode([text], query=True)[0], True

    @staticmethod
    def matched_explicit_text(explicit: list[dict[str, Any]], target_words: int) -> str:
        texts = [str(item["text"]).strip() for item in explicit if str(item.get("text") or "").strip()]
        if not texts or target_words < 1:
            return ""
        words: list[str] = []
        index = 0
        while len(words) < target_words:
            words.extend(texts[index % len(texts)].split())
            index += 1
        return " ".join(words[:target_words])


class EmbeddingStore:
    def __init__(self, index_path: Path) -> None:
        index = read_json(index_path)
        self.dimension = int(index["dimension"])
        self.model_name = str(index["model_name"])
        self.backend = str(index["backend"])
        self.split_hash = str(index["split_hash"])
        self.vectors: dict[str, dict[str, Any]] = {}
        for raw_path in index["patch_files"]:
            path = Path(raw_path)
            if not path.is_absolute():
                path = index_path.parent / path
            with np.load(path, allow_pickle=False) as data:
                ids = [str(value) for value in data["turn_ids"].tolist()]
                for row_index, turn_id in enumerate(ids):
                    if turn_id in self.vectors:
                        raise ValueError(f"Duplicate embedding ID: {turn_id}")
                    self.vectors[turn_id] = {
                        "query": np.asarray(data["query"][row_index], dtype=np.float32),
                        "document": np.asarray(data["document"][row_index], dtype=np.float32),
                        "explicit": np.asarray(data["explicit"][row_index], dtype=np.float32),
                        "assumption": np.asarray(data["assumption"][row_index], dtype=np.float32),
                        "matched_explicit": np.asarray(data["matched_explicit"][row_index], dtype=np.float32),
                        "explicit_mask": bool(data["explicit_mask"][row_index]),
                        "assumption_mask": bool(data["assumption_mask"][row_index]),
                        "matched_explicit_mask": bool(data["matched_explicit_mask"][row_index]),
                    }

    def get(self, turn_id: str) -> dict[str, Any]:
        try:
            return self.vectors[turn_id]
        except KeyError as error:
            raise KeyError(f"No cached embeddings for {turn_id}") from error


def load_turn_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["turn_id"]): row for row in read_jsonl(path)}


def history_vector(anchor: dict[str, Any], store: EmbeddingStore, dimension: int | None = None) -> np.ndarray:
    ids = list(anchor.get("history_ids") or [])
    if not ids:
        base = store.get(str(anchor["anchor_id"]))["query"]
        return np.zeros_like(base[:dimension] if dimension else base)
    vectors = [store.get(str(turn_id))["query"] for turn_id in ids]
    value = normalize(np.mean(np.stack(vectors), axis=0))
    return value[:dimension] if dimension else value


def component_vectors(
    anchor: dict[str, Any],
    store: EmbeddingStore,
    *,
    assumption_source_id: str | None = None,
    explicit_as_assumption: bool = False,
    dimension: int | None = None,
) -> dict[str, np.ndarray | bool]:
    source = store.get(str(anchor["anchor_id"]))
    donor = store.get(assumption_source_id) if assumption_source_id else source
    assumption_key = "matched_explicit" if explicit_as_assumption else "assumption"
    assumption_mask_key = "matched_explicit_mask" if explicit_as_assumption else "assumption_mask"
    def trim(value: np.ndarray) -> np.ndarray:
        return value[:dimension] if dimension else value
    return {
        "current": trim(source["query"]),
        "history": history_vector(anchor, store, dimension),
        "explicit": trim(source["explicit"]),
        "assumption": trim(donor[assumption_key]),
        "explicit_mask": bool(source["explicit_mask"]),
        "assumption_mask": bool(donor[assumption_mask_key]),
    }


def save_embedding_patch(path: Path, rows: list[dict[str, Any]], embedder: TextEmbedder) -> int:
    if not rows:
        raise ValueError("Cannot write an empty embedding patch")
    turn_ids = [str(row["turn_id"]) for row in rows]
    texts = [str(row["turn_text"]) for row in rows]
    query = list(embedder.encode(texts, query=True))
    document = list(embedder.encode(texts, query=False))
    dimension = int(query[0].shape[0])

    def pooled_fields(field: str) -> tuple[list[np.ndarray], list[bool]]:
        owners: list[int] = []
        items: list[str] = []
        for row_index, row in enumerate(rows):
            for item in row[field]:
                text = str(item.get("text") or "").strip()
                if text:
                    owners.append(row_index)
                    items.append(text)
        encoded = embedder.encode(items, query=True) if items else np.zeros((0, dimension), dtype=np.float32)
        buckets: list[list[np.ndarray]] = [[] for _ in rows]
        for owner, vector in zip(owners, encoded):
            buckets[owner].append(vector)
        vectors = [
            normalize(np.mean(np.stack(bucket), axis=0)) if bucket else np.zeros(dimension, dtype=np.float32)
            for bucket in buckets
        ]
        return vectors, [bool(bucket) for bucket in buckets]

    explicit_vectors, explicit_masks = pooled_fields("explicit")
    assumption_vectors, assumption_masks = pooled_fields("assumptions")
    matched_texts = [
        embedder.matched_explicit_text(list(row["explicit"]), int(row.get("assumption_token_count", 0)))
        for row in rows
    ]
    matched_nonempty = [text for text in matched_texts if text]
    matched_encoded = iter(embedder.encode(matched_nonempty, query=True)) if matched_nonempty else iter(())
    matched_explicit_vectors = [next(matched_encoded) if text else np.zeros(dimension, dtype=np.float32) for text in matched_texts]
    matched_masks = [bool(text) for text in matched_texts]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        turn_ids=np.asarray(turn_ids, dtype=str),
        query=np.asarray(query, dtype=np.float16),
        document=np.asarray(document, dtype=np.float16),
        explicit=np.asarray(explicit_vectors, dtype=np.float16),
        assumption=np.asarray(assumption_vectors, dtype=np.float16),
        matched_explicit=np.asarray(matched_explicit_vectors, dtype=np.float16),
        explicit_mask=np.asarray(explicit_masks, dtype=np.bool_),
        assumption_mask=np.asarray(assumption_masks, dtype=np.bool_),
        matched_explicit_mask=np.asarray(matched_masks, dtype=np.bool_),
    )
    return len(rows)


# Stage-local helper functions (evaluation).
from typing import Any

import numpy as np



FROZEN_CONDITIONS = (
    "current",
    "current_history",
    "history_explicit",
    "history_assumption",
    "full",
    "shuffled",
)


def query_for_condition(
    anchor: dict[str, Any],
    store: EmbeddingStore,
    condition: str,
    *,
    donor_id: str | None = None,
    explicit_as_assumption: bool = False,
    dimension: int | None = None,
) -> np.ndarray:
    parts = component_vectors(
        anchor,
        store,
        assumption_source_id=donor_id,
        explicit_as_assumption=explicit_as_assumption,
        dimension=dimension,
    )
    current = parts["current"]
    history = parts["history"]
    explicit = parts["explicit"]
    assumption = parts["assumption"]
    history_mask = bool(anchor.get("history_ids"))
    explicit_mask = bool(parts["explicit_mask"])
    assumption_mask = bool(parts["assumption_mask"])
    if condition == "current":
        return normalize(current)
    if condition == "current_history":
        return compose([current, history], [True, history_mask])
    if condition == "history_explicit":
        return compose([current, history, explicit], [True, history_mask, explicit_mask])
    if condition == "history_assumption":
        return compose([current, history, assumption], [True, history_mask, assumption_mask])
    if condition in {"full", "shuffled", "control"}:
        return compose(
            [current, history, explicit, assumption],
            [True, history_mask, explicit_mask, assumption_mask],
        )
    raise ValueError(f"Unknown condition: {condition}")


def evaluate_anchor(
    anchor: dict[str, Any],
    store: EmbeddingStore,
    condition: str,
    *,
    donor_id: str | None = None,
    explicit_as_assumption: bool = False,
    dimension: int | None = None,
) -> dict[str, Any]:
    query = query_for_condition(
        anchor,
        store,
        condition,
        donor_id=donor_id,
        explicit_as_assumption=explicit_as_assumption,
        dimension=dimension,
    )
    candidate_ids = [str(value) for value in anchor["candidate_ids"]]
    documents = []
    for candidate_id in candidate_ids:
        vector = store.get(candidate_id)["document"]
        documents.append(normalize(vector[:dimension] if dimension else vector))
    scores = np.stack(documents) @ normalize(query)
    metrics = rank_scores(candidate_ids, scores, str(anchor["target_id"]))
    return {
        "anchor_id": anchor["anchor_id"],
        "condition": condition,
        "category": anchor["category"],
        "show_id": anchor["show_id"],
        "episode_id": anchor["episode_id"],
        "target_id": anchor["target_id"],
        "candidate_count": len(candidate_ids),
        "candidate_pool_hash": stable_hash(candidate_ids),
        **metrics,
    }


STAGE = "exp8_exp05_mini_fusion"
CONDITIONS = ("history", "full", "shuffled")
SEEDS = (42, 43, 44)
TASKS = tuple((condition, seed) for condition in CONDITIONS for seed in SEEDS)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Train or merge the nine Exp05 mini-fusion conditions.")
    value.add_argument("--mode", choices=("count", "worker", "merge"), required=True)
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--cache-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp05_results"))
    value.add_argument("--num-patches", type=int, default=len(TASKS))
    value.add_argument("--patch-index", type=int, default=0)
    value.add_argument("--condition", choices=CONDITIONS)
    value.add_argument("--seed", type=int)
    value.add_argument("--feature-dim", type=int, default=256)
    value.add_argument("--hidden-dim", type=int, default=256)
    value.add_argument("--max-train-anchors", type=int, default=50000)
    value.add_argument("--max-epochs", type=int, default=10)
    value.add_argument("--patience", type=int, default=2)
    value.add_argument("--batch-size", type=int, default=512)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--weight-decay", type=float, default=0.01)
    value.add_argument("--temperature", type=float, default=0.05)
    value.add_argument("--device")
    value.add_argument("--smoke", action="store_true", help="Use deterministic no-training scoring for CPU smoke tests.")
    value.add_argument("--force", action="store_true")
    return value


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_mapping": [{"patch_index": i, "condition": c, "seed": s} for i, (c, s) in enumerate(TASKS)],
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "max_train_anchors": args.max_train_anchors,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "smoke": args.smoke,
    }


def task_for(args: argparse.Namespace) -> tuple[str, int]:
    if args.num_patches != len(TASKS) or not 0 <= args.patch_index < len(TASKS):
        raise ValueError(f"Exp05 requires exactly {len(TASKS)} patches indexed 0..{len(TASKS) - 1}")
    expected = TASKS[args.patch_index]
    observed = (args.condition or expected[0], args.seed if args.seed is not None else expected[1])
    if observed != expected:
        raise ValueError(f"Patch {args.patch_index} must be {expected}, got {observed}")
    return expected


def select_anchors(anchors: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(anchors) <= limit:
        return anchors
    rng = np.random.default_rng(seed)
    selected = sorted(int(value) for value in rng.choice(len(anchors), size=limit, replace=False))
    return [anchors[index] for index in selected]


def feature_rows(
    anchors: list[dict[str, Any]],
    store: EmbeddingStore,
    condition: str,
    dimension: int,
    donors: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    kept = [anchor for anchor in anchors if condition != "shuffled" or str(anchor["anchor_id"]) in donors]
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for anchor in kept:
        parts = component_vectors(
            anchor,
            store,
            assumption_source_id=donors.get(str(anchor["anchor_id"])),
            dimension=dimension,
        )
        if condition == "history":
            parts["assumption"] = np.zeros(dimension, dtype=np.float32)
            parts["explicit"] = np.zeros(dimension, dtype=np.float32)
        features.append(np.concatenate([parts[name] for name in ("current", "history", "explicit", "assumption")]))
        targets.append(normalize(store.get(str(anchor["target_id"]))["document"][:dimension]))
    if not kept:
        raise RuntimeError(f"No anchors remain for Exp05 condition {condition}")
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32), kept


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, list[dict[str, Any]], str]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("Exp05 training requires torch; use --smoke only for the synthetic CPU pipeline") from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    class FusionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_dim, args.hidden_dim)
            self.gate = nn.Linear(input_dim, args.hidden_dim)
            self.output = nn.Linear(args.hidden_dim, output_dim)

        def forward(self, values: Any) -> Any:
            hidden = torch.tanh(self.projection(values)) * torch.sigmoid(self.gate(values))
            return functional.normalize(self.output(hidden), dim=-1)

    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train)
    dev_x = torch.from_numpy(x_dev).to(device)
    dev_y = torch.from_numpy(y_dev).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, Any] | None = None
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_x = train_x[indices].to(device)
            batch_y = train_y[indices].to(device)
            prediction = model(batch_x)
            logits = prediction @ batch_y.T / args.temperature
            labels = torch.arange(len(indices), device=device)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            cosine = float((model(dev_x) * dev_y).sum(dim=1).mean().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_cosine": cosine})
        if cosine > best_score + 1e-6:
            best_score = cosine
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Exp05 did not produce a model checkpoint")
    model.load_state_dict(best_state)
    return model, history, device


def score_model(model: Any, x_dev: np.ndarray, anchors: list[dict[str, Any]], store: EmbeddingStore, dimension: int, device: str) -> dict[str, float]:
    import torch

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_dev).to(device)).detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for anchor, prediction in zip(anchors, predictions):
        candidate_ids = [str(value) for value in anchor["candidate_ids"]]
        documents = np.stack([normalize(store.get(value)["document"][:dimension]) for value in candidate_ids])
        rows.append(rank_scores(candidate_ids, documents @ normalize(prediction), str(anchor["target_id"])))
    return {
        "recall_at_1": float(np.mean([row["top1"] for row in rows])),
        "recall_at_5": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
    }


def smoke_metrics(anchors: list[dict[str, Any]], store: EmbeddingStore, condition: str, donors: dict[str, str]) -> dict[str, float]:
    frozen = "current_history" if condition == "history" else ("shuffled" if condition == "shuffled" else "full")
    rows = [
        evaluate_anchor(anchor, store, frozen, donor_id=donors.get(str(anchor["anchor_id"])))
        for anchor in anchors
        if condition != "shuffled" or str(anchor["anchor_id"]) in donors
    ]
    if not rows:
        raise RuntimeError(f"No development anchors remain for Exp05 smoke condition {condition}")
    return {
        "recall_at_1": float(np.mean([row["top1"] for row in rows])),
        "recall_at_5": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
    }


def worker(args: argparse.Namespace) -> None:
    condition, seed = task_for(args)
    anchors_path = args.data_dir / "anchors.jsonl"
    dev_path = args.data_dir / "development_anchors.jsonl"
    cache_index = read_json(args.cache_dir / "cache_index.json")
    input_hash = stable_hash({"anchors": file_hash(anchors_path), "development": file_hash(dev_path), "cache": cache_index})
    config_value = configuration(args)
    patch_dir = patch_directory(args.output_dir, args.patch_index, len(TASKS))
    expected = make_manifest(
        stage=STAGE,
        patch_index=args.patch_index,
        num_patches=len(TASKS),
        row_count=0,
        input_hash=input_hash,
        split_hash=str(cache_index["split_hash"]),
        config=config_value,
    )
    if not args.force and manifest_matches(patch_dir / "patch_manifest.json", expected):
        print(f"Reusing completed Exp05 task {condition}/{seed}")
        return
    all_anchors = list(read_jsonl(anchors_path))
    train = select_anchors([value for value in all_anchors if value["split"] == "train"], args.max_train_anchors, seed)
    dev = list(read_jsonl(dev_path))
    store = EmbeddingStore(args.cache_dir / "cache_index.json")
    dimension = min(store.dimension, args.feature_dim)
    turns = list(read_jsonl(args.data_dir / "turns.jsonl"))
    sources = {str(value["anchor_id"]) for value in train + dev}
    donors = build_control_map(turns, "same_episode", sources) if condition == "shuffled" else {}
    patch_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if args.smoke:
        metrics = smoke_metrics(dev, store, condition, donors)
        epochs = 0
    else:
        x_train, y_train, train = feature_rows(train, store, condition, dimension, donors)
        x_dev, y_dev, dev = feature_rows(dev, store, condition, dimension, donors)
        model, history, device = train_model(x_train, y_train, x_dev, y_dev, args, seed)
        metrics = score_model(model, x_dev, dev, store, dimension, device)
        import torch

        torch.save(model.state_dict(), patch_dir / "model.pt")
        epochs = len(history)
    metric = {
        "condition": condition,
        "seed": seed,
        "n_train": len(train),
        "n_validation": len(dev),
        "epochs": epochs,
        **metrics,
    }
    pd.DataFrame([metric]).to_csv(patch_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(patch_dir / "training_history.csv", index=False)
    write_json(
        patch_dir / "patch_manifest.json",
        make_manifest(
            stage=STAGE,
            patch_index=args.patch_index,
            num_patches=len(TASKS),
            row_count=len(dev),
            input_hash=input_hash,
            split_hash=str(cache_index["split_hash"]),
            config=config_value,
            extra={"condition": condition, "seed": seed},
        ),
    )


def merge(args: argparse.Namespace) -> None:
    if args.num_patches != len(TASKS):
        raise ValueError(f"Exp05 merge requires --num-patches {len(TASKS)}")
    manifests = validate_patch_manifests(args.output_dir, STAGE, len(TASKS))
    observed = [(str(item["condition"]), int(item["seed"])) for item in manifests]
    if observed != list(TASKS):
        raise RuntimeError(f"Exp05 patch/seed mapping mismatch: {observed}")
    frames = [
        pd.read_csv(patch_directory(args.output_dir, index, len(TASKS)) / "metrics.csv")
        for index in range(len(TASKS))
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    aggregated = (
        metrics.groupby("condition")[["recall_at_1", "recall_at_5", "mrr"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = ["condition"] + [f"{metric}_{stat}" for metric, stat in aggregated.columns.tolist()[1:]]
    write_json(args.output_dir / "config.json", {**configuration(args), "config_hash": manifests[0]["config_hash"]})
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp05_mini_fusion",
            "status": "complete",
            "input_hash": manifests[0]["input_hash"],
            "split_hash": manifests[0]["split_hash"],
            "task_count": len(TASKS),
            "metrics": metrics.to_dict("records"),
            "aggregate": aggregated.to_dict("records"),
        },
    )


def main() -> None:
    args = parser().parse_args()
    if args.mode == "count":
        print(f"TOTAL_ITEMS={len(TASKS)}")
        print(f"NUM_PATCHES={len(TASKS)}")
    elif args.mode == "worker":
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
