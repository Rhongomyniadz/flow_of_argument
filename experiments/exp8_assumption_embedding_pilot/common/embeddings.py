from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metrics import normalize
from .utils import read_json, read_jsonl

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
