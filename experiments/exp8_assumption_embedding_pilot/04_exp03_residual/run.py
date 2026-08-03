from __future__ import annotations

"""Stage 04: fit the three linear residual conditions."""

import argparse
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


# Stage-local progress display.
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


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


STAGE = "exp8_exp03_linear_residual"
CONDITIONS = ("baseline", "full", "shuffled")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Fit the three linear residual models.")
    value.add_argument("--data-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_data"))
    value.add_argument("--cache-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/shared_cache"))
    value.add_argument("--output-dir", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot/exp03_results"))
    value.add_argument("--feature-dim", type=int, default=256)
    value.add_argument("--max-train-anchors", type=int, default=50000)
    value.add_argument("--ridge-alpha", type=float, default=10.0)
    value.add_argument("--seed", type=int, default=42)
    return value


def features(
    anchor: dict[str, Any],
    store: EmbeddingStore,
    condition: str,
    dimension: int,
    donor: str | None,
) -> np.ndarray:
    parts = component_vectors(anchor, store, assumption_source_id=donor, dimension=dimension)
    values = [parts["current"], parts["history"], parts["explicit"]]
    if condition in {"full", "shuffled"}:
        values.append(parts["assumption"])
    return np.concatenate([np.asarray(value, dtype=np.float32) for value in values])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_x = (x - mean) / scale
    augmented = np.column_stack([np.ones(len(normalized_x), dtype=np.float32), normalized_x])
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
    return weights.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def predict(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([np.ones(len(x), dtype=np.float32), (x - mean) / scale])
    return augmented @ weights


def run(args: argparse.Namespace) -> None:
    anchors_path = args.data_dir / "anchors.jsonl"
    development_path = args.data_dir / "development_anchors.jsonl"
    all_anchors = list(read_jsonl(anchors_path))
    train = [anchor for anchor in all_anchors if anchor["split"] == "train"]
    validation = list(read_jsonl(development_path))
    if not train or not validation:
        raise RuntimeError("Stage 04 requires nonempty training and validation anchors")

    rng = np.random.default_rng(args.seed)
    if len(train) > args.max_train_anchors:
        indices = sorted(int(index) for index in rng.choice(len(train), size=args.max_train_anchors, replace=False))
        train = [train[index] for index in indices]

    cache_index = read_json(args.cache_dir / "cache_index.json")
    store = EmbeddingStore(args.cache_dir / "cache_index.json")
    dimension = min(args.feature_dim, store.dimension)
    turns = list(read_jsonl(args.data_dir / "turns.jsonl"))
    source_ids = {str(anchor["anchor_id"]) for anchor in train + validation}
    metrics: list[dict[str, Any]] = []
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        donors = build_control_map(turns, "same_episode", source_ids) if condition == "shuffled" else {}
        condition_train = [
            anchor for anchor in train
            if condition != "shuffled" or str(anchor["anchor_id"]) in donors
        ]
        condition_validation = [
            anchor for anchor in validation
            if condition != "shuffled" or str(anchor["anchor_id"]) in donors
        ]
        if not condition_train or not condition_validation:
            raise RuntimeError(f"Stage 04 condition {condition} has no usable anchors")

        x_train_rows: list[np.ndarray] = []
        y_train_rows: list[np.ndarray] = []
        for anchor in tqdm(
            condition_train,
            desc=f"stage 04 {condition} train",
            unit="anchor",
            dynamic_ncols=True,
        ):
            donor = donors.get(str(anchor["anchor_id"]))
            x_train_rows.append(features(anchor, store, condition, dimension, donor))
            y_train_rows.append(normalize(store.get(str(anchor["target_id"]))["document"][:dimension]))
        x_train = np.stack(x_train_rows)
        y_train = np.stack(y_train_rows)
        weights, mean, scale = fit_ridge(x_train, y_train, args.ridge_alpha)

        x_validation_rows: list[np.ndarray] = []
        for anchor in tqdm(
            condition_validation,
            desc=f"stage 04 {condition} validation",
            unit="anchor",
            dynamic_ncols=True,
        ):
            donor = donors.get(str(anchor["anchor_id"]))
            x_validation_rows.append(features(anchor, store, condition, dimension, donor))
        x_validation = np.stack(x_validation_rows)
        predictions = predict(x_validation, weights, mean, scale)
        targets = np.stack(
            [normalize(store.get(str(anchor["target_id"]))["document"][:dimension]) for anchor in condition_validation]
        )
        normalized_predictions = np.stack([normalize(row) for row in predictions])
        cosines = np.sum(normalized_predictions * targets, axis=1)
        denominator = float(np.sum((targets - targets.mean(axis=0)) ** 2))
        r2 = 1.0 - float(np.sum((targets - predictions) ** 2)) / denominator if denominator > 0 else float("nan")

        retrieval_rows: list[dict[str, Any]] = []
        for anchor, prediction_vector in tqdm(
            zip(condition_validation, normalized_predictions),
            total=len(condition_validation),
            desc=f"stage 04 {condition} retrieval",
            unit="anchor",
            dynamic_ncols=True,
        ):
            candidate_ids = [str(value) for value in anchor["candidate_ids"]]
            documents = np.stack(
                [normalize(store.get(candidate_id)["document"][:dimension]) for candidate_id in candidate_ids]
            )
            retrieval_rows.append(
                rank_scores(candidate_ids, documents @ prediction_vector, str(anchor["target_id"]))
            )

        metrics.append(
            {
                "condition": condition,
                "n_train": len(condition_train),
                "n_validation": len(condition_validation),
                "mean_target_cosine": float(np.mean(cosines)),
                "embedding_r2": r2,
                "recall_at_1": float(np.mean([row["top1"] for row in retrieval_rows])),
                "recall_at_5": float(np.mean([row["top5"] for row in retrieval_rows])),
                "mrr": float(np.mean([row["reciprocal_rank"] for row in retrieval_rows])),
            }
        )
        np.savez_compressed(
            model_dir / f"{condition}.npz",
            weights=weights,
            mean=mean,
            scale=scale,
        )

    configuration = {
        "conditions": list(CONDITIONS),
        "feature_dim": args.feature_dim,
        "max_train_anchors": args.max_train_anchors,
        "ridge_alpha": args.ridge_alpha,
        "seed": args.seed,
    }
    input_hash = stable_hash(
        {
            "anchors": file_hash(anchors_path),
            "development": file_hash(development_path),
            "cache": cache_index,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    write_json(args.output_dir / "config.json", configuration)
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": "exp03_linear_residual",
            "status": "complete",
            "input_hash": input_hash,
            "split_hash": str(cache_index["split_hash"]),
            "metrics": metrics,
        },
    )


def main() -> None:
    args = parser().parse_args()
    if args.feature_dim < 1 or args.max_train_anchors < 1 or args.ridge_alpha < 0:
        raise ValueError("Feature dimensions and train limits must be positive; ridge alpha cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
