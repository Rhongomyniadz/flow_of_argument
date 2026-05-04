import argparse
import hashlib
import json
import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_INPUT_DIR = Path("data/conversation_moves_labeled")
DEFAULT_OUTPUT_DIR = Path("experiments/exp1_relevance_bridge/results")
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QWEN_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_TARGET_EMBEDDING_MAX_LENGTH = 1024
DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS = 20
WHITENING_EPSILON = 1e-6
SIM_TIE_EPSILON = 1e-12
ZERO_NORM_EPSILON = 1e-12
UNRELATED_SENTENCES_PATH = Path(__file__).with_name("unrelated_sentences.json")
UNRELATED_SENTENCE_POOL_SIZE = 100
BASELINE_SENTENCE_SAMPLE_SIZE = 10
HARD_NEGATIVE_TARGET_COUNT = 24
HARD_NEGATIVE_LAYER_TARGET = 8
GREEDY_MAX_ASSUMPTIONS = 3
PAIR_EXPORT_COLUMNS = [
    "category",
    "episode_id",
    "turn_a_idx",
    "turn_b_idx",
    "pair_id",
    "turn_b_move_label",
    "analysis_bucket",
    "eligible",
    "eligibility_exclusion_reason",
    "canonical_retained",
    "coverage_drop_reason",
    "turn_a_text",
    "turn_b_claim_text",
    "candidate_assumption_count",
    "turn_b_has_assumptions",
    "selected_assumption_count",
    "selected_assumption_ids",
    "selected_context_text",
    "negative_sample_count_actual",
    "negative_pool_complete",
    "random_context_pool_complete",
    "negative_count_layer1_exact",
    "negative_count_layer2_gap3",
    "negative_count_layer2_gap2_backfill",
    "negative_count_layer3_exact",
    "negative_count_backfill_same_category",
    "negative_count_backfill_global",
    "win_rate_claim",
    "win_rate_random_context",
    "win_rate_context",
    "bridge_lift",
    "random_context_lift",
    "bridge_advantage_over_random",
    "win_rate_full_bag_context",
    "full_bag_context_valid",
    "ablation_drop_reason",
    "legacy_sim_claim_raw",
    "legacy_sim_context_raw",
    "legacy_sim_unrelated_only_raw",
    "legacy_bridge_delta_raw",
]
BOOLEAN_COLUMNS = [
    "eligible",
    "canonical_retained",
]
NULLABLE_STATUS_BOOLEAN_COLUMNS = [
    "negative_pool_complete",
    "random_context_pool_complete",
    "full_bag_context_valid",
]
HEADLINE_CONSTRUCTIVE_MOVES = {
    "Assert / Elaborate",
    "Answer",
    "Agree / Align",
}
STRESS_TEST_MOVES = {
    "Clarification Request (Generic)",
    "Clarification Request (Specific)",
    "Correction / Challenge",
    "Self-Correction",
    "Topic Shift",
    "Stonewalling / Non-Response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max_episodes_per_category", type=int, default=None)
    parser.add_argument("--num_patches", type=int, default=1)
    parser.add_argument("--patch_index", type=int, default=0)
    parser.add_argument("--episodes_per_patch", type=int, default=None)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument(
        "--embedding_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_EMBEDDING_DEVICE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--prepare_whitening_only", action="store_true")
    parser.add_argument("--prepare_whitening_patch_only", action="store_true")
    parser.add_argument("--merge_whitening_patches_only", action="store_true")
    return parser.parse_args()


def resolve_output_dir(base_output_dir: Path, embedding_model_name: str) -> Path:
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embedding_model_name.replace("/", "__").strip())
    if not model_slug:
        raise ValueError("embedding_model_name must not be empty.")
    return base_output_dir / model_slug


def resolve_patch_output_dir(base_output_dir: Path, num_patches: int, patch_index: int) -> Path:
    if num_patches == 1:
        return base_output_dir
    return base_output_dir / "patches" / f"patch_{patch_index:04d}_of_{num_patches:04d}"


def normalize_categories(input_dir: Path, requested: list[str] | None) -> list[str]:
    available = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    if not requested or any(str(item).lower() == "all" for item in requested):
        return available
    lookup = {name.lower(): name for name in available}
    chosen: list[str] = []
    for raw_name in requested:
        match = lookup.get(str(raw_name).lower())
        if match is None:
            raise ValueError(f"Unknown category: {raw_name}. Available: {', '.join(available)}")
        if match not in chosen:
            chosen.append(match)
    return chosen


def collect_category_files(
    input_dir: Path,
    categories: list[str],
    max_episodes_per_category: int | None,
) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for category in categories:
        category_files = sorted((input_dir / category).glob("*.json"))
        if max_episodes_per_category is not None:
            category_files = category_files[:max_episodes_per_category]
        files.extend((category, path) for path in category_files)
    return files


def select_patch_files(
    category_files: list[tuple[str, Path]],
    num_patches: int,
    patch_index: int,
    episodes_per_patch: int | None,
) -> list[tuple[str, Path]]:
    if episodes_per_patch is not None:
        start = patch_index * episodes_per_patch
        end = min(start + episodes_per_patch, len(category_files))
        return category_files[start:end]
    return [item for idx, item in enumerate(category_files) if idx % num_patches == patch_index]


def validate_patch_args(num_patches: int, patch_index: int, episodes_per_patch: int | None) -> None:
    if num_patches < 1:
        raise ValueError(f"num_patches must be >= 1, got {num_patches}")
    if patch_index < 0 or patch_index >= num_patches:
        raise ValueError(f"patch_index must be in [0, {num_patches - 1}], got {patch_index}")
    if episodes_per_patch is not None and episodes_per_patch < 1:
        raise ValueError(f"episodes_per_patch must be >= 1, got {episodes_per_patch}")


def load_turns(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("turns", [])


def load_unrelated_sentences(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing unrelated sentences file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("unrelated_sentences.json must contain a top-level JSON array of strings.")
    sentences: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise TypeError(
                f"unrelated_sentences.json item {index} must be a string, got {type(item).__name__}."
            )
        sentence = item.strip()
        if not sentence:
            raise ValueError(f"unrelated_sentences.json item {index} is empty after trimming whitespace.")
        if sentence in seen:
            raise ValueError(f"unrelated_sentences.json contains a duplicate sentence at item {index}.")
        seen.add(sentence)
        sentences.append(sentence)
    if len(sentences) != UNRELATED_SENTENCE_POOL_SIZE:
        raise ValueError(
            "unrelated_sentences.json must contain exactly "
            f"{UNRELATED_SENTENCE_POOL_SIZE} unique sentences, got {len(sentences)}."
        )
    return sentences


def normalize_text_list(raw_items: Any) -> tuple[list[str], bool]:
    if raw_items is None:
        return [], False
    if not isinstance(raw_items, list):
        return [], True
    texts: list[str] = []
    corrupt = False
    for item in raw_items:
        if isinstance(item, dict):
            raw_text = item.get("text")
        else:
            raw_text = item
        if not isinstance(raw_text, str):
            corrupt = True
            continue
        text = raw_text.strip()
        if not text:
            corrupt = True
            continue
        texts.append(text)
    return texts, corrupt


def item_text(items: Any) -> str:
    texts, _ = normalize_text_list(items)
    return " ".join(texts).strip()


def build_context_text(claim_text: str, assumption_texts: list[str]) -> str:
    return " ".join([claim_text, *assumption_texts]).strip()


def turn_time(turn: dict[str, Any]) -> float:
    raw = turn.get("start_time", turn.get("startTime", turn.get("end_time", turn.get("endTime", 0.0))))
    return float(raw if raw is not None else 0.0)


def resolve_analysis_bucket(move_label: str) -> str | None:
    if move_label in HEADLINE_CONSTRUCTIVE_MOVES:
        return "headline_constructive"
    if move_label in STRESS_TEST_MOVES:
        return "stress_test"
    return None


def compute_file_list_hash(category_files: list[tuple[str, Path]]) -> str:
    payload = "\n".join(f"{category}:{path.as_posix()}" for category, path in category_files)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_seed_int(seed_text: str) -> int:
    return int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big", signed=False)


def build_pair_rng(pair_id: str, sample_label: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(build_seed_int(f"{pair_id}:{sample_label}:{seed}"))


def serialize_json_field(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def coerce_nullable_bool(series: pd.Series) -> pd.Series:
    def _convert(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        return None
    return series.map(_convert)


def resolve_embedding_max_length(tokenizer: Any) -> int:
    raw_model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(raw_model_max_length, int):
        return DEFAULT_TARGET_EMBEDDING_MAX_LENGTH
    if raw_model_max_length <= 0:
        return DEFAULT_TARGET_EMBEDDING_MAX_LENGTH
    if raw_model_max_length > 100000:
        return DEFAULT_TARGET_EMBEDDING_MAX_LENGTH
    return min(DEFAULT_TARGET_EMBEDDING_MAX_LENGTH, raw_model_max_length)


def resolve_embedding_device(requested_device: str) -> torch.device:
    normalized_device = requested_device.strip().lower()
    if normalized_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_device == "cpu":
        return torch.device("cpu")
    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("embedding_device='cuda' was requested, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported embedding device: {requested_device}")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def iter_embedded_text_batches(
    texts: list[str],
    batch_size: int,
    use_tqdm: bool,
    embedding_model_name: str,
    embedding_device: str,
) -> Iterator[tuple[list[str], np.ndarray]]:
    device = resolve_embedding_device(embedding_device)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True).to(device).eval()
    except torch.OutOfMemoryError as error:
        if device.type != "cuda":
            raise
        raise RuntimeError(
            "CUDA ran out of memory while loading the embedding model. "
            f"model={embedding_model_name}, embedding_device={device.type}. "
            "Free GPU memory or rerun with --embedding_device cpu."
        ) from error
    embedding_max_length = resolve_embedding_max_length(tokenizer)
    unique_texts = list(dict.fromkeys(texts))
    logger.info(
        "Embedding %d unique texts with model=%s on device=%s and batch_size=%d.",
        len(unique_texts),
        embedding_model_name,
        device.type,
        batch_size,
    )
    iterator = tqdm(
        range(0, len(unique_texts), batch_size),
        desc="Embedding texts",
        disable=not use_tqdm,
    )
    with torch.inference_mode():
        for start in iterator:
            batch = unique_texts[start:start + batch_size]
            try:
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=embedding_max_length,
                    return_tensors="pt",
                )
                tokens = {name: value.to(device) for name, value in tokens.items()}
                output = model(**tokens)
            except torch.OutOfMemoryError as error:
                if device.type != "cuda":
                    raise
                raise RuntimeError(
                    "CUDA ran out of memory while embedding texts. "
                    f"model={embedding_model_name}, embedding_device={device.type}, batch_size={batch_size}. "
                    "Reduce --embedding_batch_size, free GPU memory, or rerun with --embedding_device cpu."
                ) from error
            pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"]).cpu().numpy()
            yield batch, pooled.astype(np.float32, copy=False)


def embed_texts(
    texts: list[str],
    batch_size: int,
    use_tqdm: bool,
    embedding_model_name: str,
    embedding_device: str,
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for batch, pooled in iter_embedded_text_batches(
        texts=texts,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
    ):
        for text, vector in zip(batch, pooled):
            vectors[text] = vector
    return vectors


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= ZERO_NORM_EPSILON:
        return None
    return (vector / norm).astype(np.float32, copy=False)


def fit_whitening_artifact(text_to_raw_vec: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.vstack([text_to_raw_vec[text] for text in sorted(text_to_raw_vec)]).astype(np.float64, copy=False)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
    basis, singular_values, _ = np.linalg.svd(covariance, full_matrices=False)
    scales = 1.0 / np.sqrt(singular_values + WHITENING_EPSILON)
    return mean.astype(np.float32), basis.astype(np.float32), scales.astype(np.float32)


def collect_whitening_moments_from_texts(
    texts: list[str],
    batch_size: int,
    use_tqdm: bool,
    embedding_model_name: str,
    embedding_device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = 0
    sum_vector: np.ndarray | None = None
    cross_product: np.ndarray | None = None
    for _, pooled in iter_embedded_text_batches(
        texts=texts,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
    ):
        pooled64 = pooled.astype(np.float64, copy=False)
        if sum_vector is None:
            embedding_dim = int(pooled64.shape[1])
            sum_vector = np.zeros(embedding_dim, dtype=np.float64)
            cross_product = np.zeros((embedding_dim, embedding_dim), dtype=np.float64)
        sum_vector += pooled64.sum(axis=0)
        cross_product += pooled64.T @ pooled64
        count += int(pooled64.shape[0])

    if count < 1 or sum_vector is None or cross_product is None:
        raise RuntimeError("Cannot fit whitening artifact because no texts were embedded.")

    return sum_vector, cross_product, count


def fit_whitening_artifact_from_moments(
    sum_vector: np.ndarray,
    cross_product: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    mean = sum_vector / float(count)
    covariance = (cross_product - (float(count) * np.outer(mean, mean))) / max(count - 1, 1)
    covariance = (covariance + covariance.T) * 0.5
    basis, singular_values, _ = np.linalg.svd(covariance, full_matrices=False)
    scales = 1.0 / np.sqrt(singular_values + WHITENING_EPSILON)
    return mean.astype(np.float32), basis.astype(np.float32), scales.astype(np.float32)


def fit_whitening_artifact_from_texts(
    texts: list[str],
    batch_size: int,
    use_tqdm: bool,
    embedding_model_name: str,
    embedding_device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    sum_vector, cross_product, count = collect_whitening_moments_from_texts(
        texts=texts,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
    )
    mean, basis, scales = fit_whitening_artifact_from_moments(
        sum_vector=sum_vector,
        cross_product=cross_product,
        count=count,
    )
    return mean, basis, scales, count


def apply_whitening(
    text_to_raw_vec: dict[str, np.ndarray],
    mean: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
) -> dict[str, np.ndarray | None]:
    transformed: dict[str, np.ndarray | None] = {}
    basis64 = basis.astype(np.float64, copy=False)
    scales64 = scales.astype(np.float64, copy=False)
    mean64 = mean.astype(np.float64, copy=False)
    for text, raw_vec in text_to_raw_vec.items():
        centered = raw_vec.astype(np.float64, copy=False) - mean64
        whitened = (centered @ basis64) * scales64
        whitened32 = whitened.astype(np.float32, copy=False)
        transformed[text] = normalize_vector(whitened32)
    return transformed


def whitening_paths(model_output_dir: Path) -> tuple[Path, Path]:
    return model_output_dir / "exp1_whitening_params.npz", model_output_dir / "exp1_whitening_manifest.json"


def whitening_patch_dir(model_output_dir: Path) -> Path:
    return model_output_dir / "whitening_patches"


def whitening_patch_paths(model_output_dir: Path, num_patches: int, patch_index: int) -> tuple[Path, Path]:
    patch_dir = whitening_patch_dir(model_output_dir)
    patch_name = f"patch_{patch_index:04d}_of_{num_patches:04d}"
    return patch_dir / f"{patch_name}.npz", patch_dir / f"{patch_name}.json"


def text_partition_index(text: str, num_patches: int) -> int:
    if num_patches < 1:
        raise ValueError(f"num_patches must be >= 1, got {num_patches}")
    return build_seed_int(f"exp1_whitening:{text}") % num_patches


def select_whitening_partition_texts(texts: list[str], num_patches: int, patch_index: int) -> list[str]:
    validate_patch_args(num_patches, patch_index, None)
    unique_texts = list(dict.fromkeys(texts))
    return [
        text
        for text in unique_texts
        if text_partition_index(text, num_patches) == patch_index
    ]


def build_whitening_manifest(
    args: argparse.Namespace,
    categories: list[str],
    category_files: list[tuple[str, Path]],
    selected_episode_file_count: int,
) -> dict[str, Any]:
    return {
        "artifact_version": 2,
        "input_dir": str(args.input_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "embedding_device": str(resolve_embedding_device(args.embedding_device)),
        "categories": categories,
        "max_episodes_per_category": int(args.max_episodes_per_category) if args.max_episodes_per_category is not None else None,
        "selected_episode_file_count": int(selected_episode_file_count),
        "category_files_hash": compute_file_list_hash(category_files),
        "whitening_method": "pca_svd",
        "whitening_epsilon": WHITENING_EPSILON,
    }


def manifest_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_whitening_artifact(
    params_path: Path,
    manifest_path: Path,
    mean: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    manifest_payload: dict[str, Any],
) -> str:
    manifest_with_hash = dict(manifest_payload)
    manifest_with_hash["manifest_hash"] = manifest_hash(manifest_payload)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(params_path, mean=mean, basis=basis, scales=scales)
    manifest_path.write_text(json.dumps(manifest_with_hash, indent=2))
    return str(manifest_with_hash["manifest_hash"])


def write_whitening_patch_moments(
    params_path: Path,
    manifest_path: Path,
    sum_vector: np.ndarray,
    cross_product: np.ndarray,
    count: int,
    manifest_payload: dict[str, Any],
    num_patches: int,
    patch_index: int,
) -> str:
    manifest_with_hash = dict(manifest_payload)
    manifest_with_hash["manifest_hash"] = manifest_hash(manifest_payload)
    manifest_with_hash["num_patches"] = int(num_patches)
    manifest_with_hash["patch_index"] = int(patch_index)
    manifest_with_hash["text_count"] = int(count)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        params_path,
        sum_vector=sum_vector,
        cross_product=cross_product,
        count=np.array(count, dtype=np.int64),
    )
    manifest_path.write_text(json.dumps(manifest_with_hash, indent=2), encoding="utf-8")
    return str(manifest_with_hash["manifest_hash"])


def load_and_validate_whitening_patch_moments(
    params_path: Path,
    manifest_path: Path,
    expected_payload: dict[str, Any],
    num_patches: int,
    patch_index: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    if not params_path.exists() or not manifest_path.exists():
        raise RuntimeError(
            "Missing whitening patch moments. "
            f"params_path={params_path}, manifest_path={manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest_hash(expected_payload)
    observed_hash = manifest.get("manifest_hash")
    if observed_hash != expected_hash:
        raise RuntimeError(
            "Whitening patch manifest hash mismatch. "
            f"patch_index={patch_index}, expected={expected_hash}, observed={observed_hash}"
        )
    observed_num_patches = int(manifest.get("num_patches", -1))
    observed_patch_index = int(manifest.get("patch_index", -1))
    if observed_num_patches != num_patches or observed_patch_index != patch_index:
        raise RuntimeError(
            "Whitening patch metadata mismatch. "
            f"expected_num_patches={num_patches}, observed_num_patches={observed_num_patches}, "
            f"expected_patch_index={patch_index}, observed_patch_index={observed_patch_index}"
        )
    with np.load(params_path) as params:
        sum_vector = params["sum_vector"].astype(np.float64, copy=True)
        cross_product = params["cross_product"].astype(np.float64, copy=True)
        count = int(params["count"].item())
    return sum_vector, cross_product, count, manifest


def load_and_validate_whitening_artifact(
    params_path: Path,
    manifest_path: Path,
    expected_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not params_path.exists() or not manifest_path.exists():
        raise RuntimeError(
            "Missing whitening artifact. Run Exp 1 whitening preparation before patch mode. "
            f"params_path={params_path}, manifest_path={manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_hash = manifest_hash(expected_payload)
    observed_hash = manifest.get("manifest_hash")
    if observed_hash != expected_hash:
        raise RuntimeError(
            "Whitening artifact manifest hash mismatch. "
            f"expected={expected_hash}, observed={observed_hash}"
        )
    params = np.load(params_path)
    return params["mean"], params["basis"], params["scales"], manifest


def build_episode_records(category: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    turns = load_turns(path)
    ordered_turns = sorted(turns, key=turn_time)
    substantive_position = 0
    turn_records: list[dict[str, Any]] = []
    for turn in ordered_turns:
        turn_type_label = str(turn.get("turn_type_label") or "").strip()
        move_label = str(turn.get("conversation_move_label") or "").strip()
        claim_text = item_text(turn.get("explicit_propositions")) or str(turn.get("turn_text") or "").strip()
        assumption_texts, assumption_field_corrupt = normalize_text_list(turn.get("assumptions"))
        episode_id = str(turn.get("episode_id") or path.stem)
        turn_idx = int(turn.get("turn_idx", -1))
        assumption_ids = [
            f"{category}:{episode_id}:{turn_idx}:{assumption_index}"
            for assumption_index, _ in enumerate(assumption_texts)
        ]
        if turn_type_label == "Substantive":
            substantive_position += 1
            turn_records.append(
                {
                    "turn_id": f"{category}:{episode_id}:{turn_idx}",
                    "category": category,
                    "episode_id": episode_id,
                    "turn_idx": turn_idx,
                    "substantive_position": substantive_position,
                    "move_label": move_label,
                    "analysis_bucket": resolve_analysis_bucket(move_label),
                    "claim_text": claim_text,
                    "assumption_texts": assumption_texts,
                    "assumption_ids": assumption_ids,
                    "assumption_field_corrupt": bool(assumption_field_corrupt),
                }
            )

    pair_rows: list[dict[str, Any]] = []
    for first_turn, second_turn in zip(ordered_turns, ordered_turns[1:]):
        episode_id = str(second_turn.get("episode_id") or path.stem)
        turn_a_idx = int(first_turn.get("turn_idx", -1))
        turn_b_idx = int(second_turn.get("turn_idx", -1))
        turn_a_type = str(first_turn.get("turn_type_label") or "").strip()
        turn_b_type = str(second_turn.get("turn_type_label") or "").strip()
        turn_b_move_label = str(second_turn.get("conversation_move_label") or "").strip()
        turn_a_text = item_text(first_turn.get("explicit_propositions")) or str(first_turn.get("turn_text") or "").strip()
        turn_b_claim_text = item_text(second_turn.get("explicit_propositions")) or str(second_turn.get("turn_text") or "").strip()
        turn_b_assumption_texts, assumption_field_corrupt = normalize_text_list(second_turn.get("assumptions"))
        turn_b_assumption_ids = [
            f"{category}:{episode_id}:{turn_b_idx}:{assumption_index}"
            for assumption_index, _ in enumerate(turn_b_assumption_texts)
        ]
        pair_id = f"{category}:{episode_id}:{turn_a_idx}:{turn_b_idx}"
        if turn_a_type != "Substantive" or turn_b_type != "Substantive":
            eligible = False
            analysis_bucket = None
            eligibility_exclusion_reason = "non_substantive_pair"
        else:
            analysis_bucket = resolve_analysis_bucket(turn_b_move_label)
            eligible = analysis_bucket is not None
            eligibility_exclusion_reason = None if eligible else "unsupported_move_bucket"
        pair_rows.append(
            {
                "category": category,
                "episode_id": episode_id,
                "turn_a_idx": turn_a_idx,
                "turn_b_idx": turn_b_idx,
                "pair_id": pair_id,
                "turn_b_move_label": turn_b_move_label,
                "analysis_bucket": analysis_bucket,
                "eligible": eligible,
                "eligibility_exclusion_reason": eligibility_exclusion_reason,
                "turn_a_text": turn_a_text,
                "turn_b_claim_text": turn_b_claim_text,
                "turn_b_assumption_texts": turn_b_assumption_texts,
                "turn_b_assumption_ids": turn_b_assumption_ids,
                "candidate_assumption_count": len(turn_b_assumption_texts),
                "turn_b_has_assumptions": int(bool(turn_b_assumption_texts)),
                "turn_b_assumption_field_corrupt": bool(assumption_field_corrupt),
            }
        )
    return turn_records, pair_rows


def pair_sort_key(pair_row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(pair_row["category"]),
        str(pair_row["episode_id"]),
        int(pair_row["turn_a_idx"]),
        int(pair_row["turn_b_idx"]),
    )


def build_global_records(
    category_files: list[tuple[str, Path]],
    selected_files: list[tuple[str, Path]],
    use_tqdm: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_lookup = {path: category for category, path in selected_files}
    global_turn_records: list[dict[str, Any]] = []
    selected_pair_rows: list[dict[str, Any]] = []
    selected_turn_records: list[dict[str, Any]] = []
    iterator = tqdm(category_files, desc="Loading Exp 1 episodes", disable=not use_tqdm)
    for category, path in iterator:
        turn_records, pair_rows = build_episode_records(category, path)
        global_turn_records.extend(turn_records)
        if path in selected_lookup:
            selected_turn_records.extend(turn_records)
            selected_pair_rows.extend(pair_rows)
    selected_turn_records = sorted(
        selected_turn_records,
        key=lambda item: (
            str(item["category"]),
            str(item["episode_id"]),
            int(item["turn_idx"]),
        ),
    )
    selected_pair_rows = sorted(selected_pair_rows, key=pair_sort_key)
    return global_turn_records, selected_turn_records, selected_pair_rows


def assumption_pool_records(turn_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for turn_record in turn_records:
        for assumption_index, (assumption_id, assumption_text) in enumerate(
            zip(turn_record["assumption_ids"], turn_record["assumption_texts"])
        ):
            records.append(
                {
                    "assumption_id": assumption_id,
                    "category": turn_record["category"],
                    "episode_id": turn_record["episode_id"],
                    "turn_idx": int(turn_record["turn_idx"]),
                    "assumption_index": assumption_index,
                    "move_label": turn_record["move_label"],
                    "analysis_bucket": turn_record["analysis_bucket"],
                    "text": assumption_text,
                }
            )
    return records


def collect_canonical_text_pool(
    global_turn_records: list[dict[str, Any]],
    selected_pair_rows: list[dict[str, Any]],
) -> list[str]:
    texts = [
        record["claim_text"]
        for record in global_turn_records
        if record["claim_text"]
    ]
    for record in global_turn_records:
        texts.extend(record["assumption_texts"])
    for pair_row in selected_pair_rows:
        if bool(pair_row["eligible"]):
            texts.append(pair_row["turn_a_text"])
            texts.append(pair_row["turn_b_claim_text"])
    return [text for text in texts if isinstance(text, str) and text.strip()]


def collect_candidate_context_texts(selected_pair_rows: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for pair_row in selected_pair_rows:
        if not bool(pair_row["eligible"]):
            continue
        claim_text = str(pair_row["turn_b_claim_text"])
        assumption_texts = [str(text) for text in pair_row["turn_b_assumption_texts"] if str(text).strip()]
        max_subset_size = min(GREEDY_MAX_ASSUMPTIONS, len(assumption_texts))
        for subset_size in range(1, max_subset_size + 1):
            for subset_indices in combinations(range(len(assumption_texts)), subset_size):
                selected_texts = [assumption_texts[index] for index in subset_indices]
                texts.append(build_context_text(claim_text, selected_texts))
        if len(assumption_texts) > GREEDY_MAX_ASSUMPTIONS:
            texts.append(build_context_text(claim_text, assumption_texts))
    return [text for text in texts if text.strip()]


def prepare_whitening_artifact(
    args: argparse.Namespace,
    model_output_dir: Path,
    categories: list[str],
    category_files: list[tuple[str, Path]],
    use_tqdm: bool,
) -> dict[str, Any]:
    global_turn_records, _, selected_pair_rows = build_global_records(
        category_files=category_files,
        selected_files=category_files,
        use_tqdm=use_tqdm,
    )
    texts_to_embed = collect_canonical_text_pool(global_turn_records, selected_pair_rows)
    mean, basis, scales, text_count = fit_whitening_artifact_from_texts(
        texts=texts_to_embed,
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
    )
    params_path, manifest_path = whitening_paths(model_output_dir)
    payload = build_whitening_manifest(
        args=args,
        categories=categories,
        category_files=category_files,
        selected_episode_file_count=len(category_files),
    )
    observed_hash = write_whitening_artifact(
        params_path=params_path,
        manifest_path=manifest_path,
        mean=mean,
        basis=basis,
        scales=scales,
        manifest_payload=payload,
    )
    logger.info("Prepared Exp 1 whitening artifact at %s", params_path)
    return {
        "params_path": str(params_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": observed_hash,
        "text_count": int(text_count),
    }


def prepare_whitening_patch_moments(
    args: argparse.Namespace,
    model_output_dir: Path,
    categories: list[str],
    category_files: list[tuple[str, Path]],
    use_tqdm: bool,
) -> dict[str, Any]:
    global_turn_records, _, selected_pair_rows = build_global_records(
        category_files=category_files,
        selected_files=category_files,
        use_tqdm=use_tqdm,
    )
    texts_to_embed = select_whitening_partition_texts(
        texts=collect_canonical_text_pool(global_turn_records, selected_pair_rows),
        num_patches=args.num_patches,
        patch_index=args.patch_index,
    )
    if not texts_to_embed:
        raise RuntimeError(
            "No texts selected for whitening patch. "
            f"num_patches={args.num_patches}, patch_index={args.patch_index}"
        )
    sum_vector, cross_product, text_count = collect_whitening_moments_from_texts(
        texts=texts_to_embed,
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
    )
    params_path, manifest_path = whitening_patch_paths(
        model_output_dir=model_output_dir,
        num_patches=args.num_patches,
        patch_index=args.patch_index,
    )
    payload = build_whitening_manifest(
        args=args,
        categories=categories,
        category_files=category_files,
        selected_episode_file_count=len(category_files),
    )
    observed_hash = write_whitening_patch_moments(
        params_path=params_path,
        manifest_path=manifest_path,
        sum_vector=sum_vector,
        cross_product=cross_product,
        count=text_count,
        manifest_payload=payload,
        num_patches=args.num_patches,
        patch_index=args.patch_index,
    )
    logger.info("Prepared Exp 1 whitening patch moments at %s", params_path)
    return {
        "params_path": str(params_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": observed_hash,
        "text_count": int(text_count),
        "num_patches": int(args.num_patches),
        "patch_index": int(args.patch_index),
    }


def merge_whitening_patch_moments(
    args: argparse.Namespace,
    model_output_dir: Path,
    categories: list[str],
    category_files: list[tuple[str, Path]],
) -> dict[str, Any]:
    payload = build_whitening_manifest(
        args=args,
        categories=categories,
        category_files=category_files,
        selected_episode_file_count=len(category_files),
    )
    merged_sum_vector: np.ndarray | None = None
    merged_cross_product: np.ndarray | None = None
    merged_count = 0
    patch_manifests: list[dict[str, Any]] = []
    for patch_index in range(args.num_patches):
        params_path, manifest_path = whitening_patch_paths(
            model_output_dir=model_output_dir,
            num_patches=args.num_patches,
            patch_index=patch_index,
        )
        sum_vector, cross_product, count, patch_manifest = load_and_validate_whitening_patch_moments(
            params_path=params_path,
            manifest_path=manifest_path,
            expected_payload=payload,
            num_patches=args.num_patches,
            patch_index=patch_index,
        )
        if merged_sum_vector is None:
            merged_sum_vector = np.zeros_like(sum_vector, dtype=np.float64)
            merged_cross_product = np.zeros_like(cross_product, dtype=np.float64)
        if merged_sum_vector.shape != sum_vector.shape:
            raise RuntimeError(
                "Whitening patch sum vector shape mismatch. "
                f"patch_index={patch_index}, expected_shape={merged_sum_vector.shape}, observed_shape={sum_vector.shape}"
            )
        if merged_cross_product is None or merged_cross_product.shape != cross_product.shape:
            expected_shape = None if merged_cross_product is None else merged_cross_product.shape
            raise RuntimeError(
                "Whitening patch cross-product shape mismatch. "
                f"patch_index={patch_index}, expected_shape={expected_shape}, observed_shape={cross_product.shape}"
            )
        merged_sum_vector += sum_vector
        merged_cross_product += cross_product
        merged_count += count
        patch_manifests.append(patch_manifest)

    if merged_sum_vector is None or merged_cross_product is None:
        raise RuntimeError("No whitening patch moments were found to merge.")

    mean, basis, scales = fit_whitening_artifact_from_moments(
        sum_vector=merged_sum_vector,
        cross_product=merged_cross_product,
        count=merged_count,
    )
    params_path, manifest_path = whitening_paths(model_output_dir)
    observed_hash = write_whitening_artifact(
        params_path=params_path,
        manifest_path=manifest_path,
        mean=mean,
        basis=basis,
        scales=scales,
        manifest_payload=payload,
    )
    logger.info("Merged Exp 1 whitening artifact at %s", params_path)
    return {
        "params_path": str(params_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": observed_hash,
        "text_count": int(merged_count),
        "merged_patch_count": int(len(patch_manifests)),
        "num_patches": int(args.num_patches),
    }


def ensure_whitening_artifact(
    args: argparse.Namespace,
    model_output_dir: Path,
    categories: list[str],
    category_files: list[tuple[str, Path]],
    use_tqdm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    params_path, manifest_path = whitening_paths(model_output_dir)
    expected_payload = build_whitening_manifest(
        args=args,
        categories=categories,
        category_files=category_files,
        selected_episode_file_count=len(category_files),
    )
    if args.num_patches == 1 and not params_path.exists():
        prepare_whitening_artifact(
            args=args,
            model_output_dir=model_output_dir,
            categories=categories,
            category_files=category_files,
            use_tqdm=use_tqdm,
        )
    return load_and_validate_whitening_artifact(
        params_path=params_path,
        manifest_path=manifest_path,
        expected_payload=expected_payload,
    )


def sample_unrelated_sentences(pair_id: str, unrelated_sentences: list[str], seed: int) -> list[str]:
    if len(unrelated_sentences) < BASELINE_SENTENCE_SAMPLE_SIZE:
        raise ValueError(
            "Unrelated sentence pool is too small for the baseline sample size. "
            f"pool_size={len(unrelated_sentences)}, sample_size={BASELINE_SENTENCE_SAMPLE_SIZE}"
        )
    rng = build_pair_rng(pair_id, "legacy_unrelated_only", seed)
    sampled_indices = rng.choice(len(unrelated_sentences), size=BASELINE_SENTENCE_SAMPLE_SIZE, replace=False)
    return [unrelated_sentences[int(idx)] for idx in np.asarray(sampled_indices).tolist()]


def build_turn_indexes(turn_records: list[dict[str, Any]]) -> dict[str, Any]:
    turn_lookup = {record["turn_id"]: record for record in turn_records}
    by_category_headline: dict[str, list[str]] = {}
    by_episode_substantive: dict[tuple[str, str], list[str]] = {}
    by_category_move_headline: dict[tuple[str, str], list[str]] = {}
    for record in sorted(
        turn_records,
        key=lambda item: (item["category"], item["move_label"], item["episode_id"], int(item["turn_idx"])),
    ):
        turn_id = record["turn_id"]
        if record["analysis_bucket"] == "headline_constructive":
            by_category_headline.setdefault(record["category"], []).append(turn_id)
            by_category_move_headline.setdefault((record["category"], record["move_label"]), []).append(turn_id)
        episode_key = (str(record["category"]), str(record["episode_id"]))
        by_episode_substantive.setdefault(episode_key, []).append(turn_id)
    assumption_records = assumption_pool_records(turn_records)
    assumption_lookup = {record["assumption_id"]: record for record in assumption_records}
    assumptions_by_category_move: dict[tuple[str, str], list[str]] = {}
    assumptions_by_category: dict[str, list[str]] = {}
    assumptions_global_headline: list[str] = []
    for record in sorted(
        assumption_records,
        key=lambda item: (
            item["category"],
            item["move_label"],
            item["episode_id"],
            int(item["turn_idx"]),
            int(item["assumption_index"]),
        ),
    ):
        assumptions_by_category_move.setdefault((record["category"], record["move_label"]), []).append(record["assumption_id"])
        assumptions_by_category.setdefault(record["category"], []).append(record["assumption_id"])
        if record["analysis_bucket"] == "headline_constructive":
            assumptions_global_headline.append(record["assumption_id"])
    return {
        "turn_lookup": turn_lookup,
        "by_category_headline": by_category_headline,
        "by_episode_substantive": by_episode_substantive,
        "by_category_move_headline": by_category_move_headline,
        "assumption_lookup": assumption_lookup,
        "assumptions_by_category_move": assumptions_by_category_move,
        "assumptions_by_category": assumptions_by_category,
        "assumptions_global_headline": assumptions_global_headline,
    }


def sample_unique_ids(candidate_ids: list[str], sample_size: int, pair_id: str, sample_label: str, seed: int) -> list[str]:
    if sample_size <= 0 or not candidate_ids:
        return []
    if len(candidate_ids) <= sample_size:
        return list(candidate_ids)
    rng = build_pair_rng(pair_id, sample_label, seed)
    sampled_indices = rng.choice(len(candidate_ids), size=sample_size, replace=False)
    return [candidate_ids[int(idx)] for idx in np.asarray(sampled_indices).tolist()]


def select_negative_ids(
    pair_row: dict[str, Any],
    turn_indexes: dict[str, Any],
    seed: int,
) -> tuple[list[str], dict[str, int], bool]:
    if not bool(pair_row["eligible"]):
        return [], {
            "negative_count_layer1_exact": 0,
            "negative_count_layer2_gap3": 0,
            "negative_count_layer2_gap2_backfill": 0,
            "negative_count_layer3_exact": 0,
            "negative_count_backfill_same_category": 0,
            "negative_count_backfill_global": 0,
        }, False

    pair_id = str(pair_row["pair_id"])
    category = str(pair_row["category"])
    episode_id = str(pair_row["episode_id"])
    episode_key = (category, episode_id)
    move_label = str(pair_row["turn_b_move_label"])
    turn_lookup = turn_indexes["turn_lookup"]
    by_category_headline = turn_indexes["by_category_headline"]
    by_episode_substantive = turn_indexes["by_episode_substantive"]
    by_category_move_headline = turn_indexes["by_category_move_headline"]

    turn_b_turn_id = f"{category}:{episode_id}:{int(pair_row['turn_b_idx'])}"
    turn_b_record = turn_lookup.get(turn_b_turn_id)
    turn_b_position = int(turn_b_record["substantive_position"]) if turn_b_record is not None else -1
    used_ids: set[str] = set()
    selected_ids: list[str] = []
    counts = {
        "negative_count_layer1_exact": 0,
        "negative_count_layer2_gap3": 0,
        "negative_count_layer2_gap2_backfill": 0,
        "negative_count_layer3_exact": 0,
        "negative_count_backfill_same_category": 0,
        "negative_count_backfill_global": 0,
    }

    def available(ids: list[str]) -> list[str]:
        return [candidate_id for candidate_id in ids if candidate_id not in used_ids]

    layer1_exact = [
        candidate_id
        for candidate_id in by_category_headline.get(category, [])
        if turn_lookup[candidate_id]["episode_id"] != episode_id
    ]
    chosen = sample_unique_ids(available(layer1_exact), HARD_NEGATIVE_LAYER_TARGET, pair_id, "negative_layer1_exact", seed)
    selected_ids.extend(chosen)
    used_ids.update(chosen)
    counts["negative_count_layer1_exact"] = len(chosen)

    same_episode_candidates = [
        candidate_id
        for candidate_id in by_episode_substantive.get(episode_key, [])
        if abs(int(turn_lookup[candidate_id]["substantive_position"]) - turn_b_position) >= 3
    ]
    chosen = sample_unique_ids(available(same_episode_candidates), HARD_NEGATIVE_LAYER_TARGET, pair_id, "negative_layer2_gap3", seed)
    selected_ids.extend(chosen)
    used_ids.update(chosen)
    counts["negative_count_layer2_gap3"] = len(chosen)

    same_episode_gap2_candidates = [
        candidate_id
        for candidate_id in by_episode_substantive.get(episode_key, [])
        if abs(int(turn_lookup[candidate_id]["substantive_position"]) - turn_b_position) >= 2
    ]
    remaining_layer2 = HARD_NEGATIVE_LAYER_TARGET - counts["negative_count_layer2_gap3"]
    if remaining_layer2 > 0:
        chosen = sample_unique_ids(
            available(same_episode_gap2_candidates),
            remaining_layer2,
            pair_id,
            "negative_layer2_gap2_backfill",
            seed,
        )
        selected_ids.extend(chosen)
        used_ids.update(chosen)
        counts["negative_count_layer2_gap2_backfill"] = len(chosen)

    layer3_exact = [
        candidate_id
        for candidate_id in by_category_move_headline.get((category, move_label), [])
        if turn_lookup[candidate_id]["episode_id"] != episode_id
    ]
    chosen = sample_unique_ids(available(layer3_exact), HARD_NEGATIVE_LAYER_TARGET, pair_id, "negative_layer3_exact", seed)
    selected_ids.extend(chosen)
    used_ids.update(chosen)
    counts["negative_count_layer3_exact"] = len(chosen)

    if len(selected_ids) < HARD_NEGATIVE_TARGET_COUNT:
        same_category_backfill = [
            candidate_id
            for candidate_id, record in turn_lookup.items()
            if record["category"] == category and record["episode_id"] != episode_id
        ]
        needed = HARD_NEGATIVE_TARGET_COUNT - len(selected_ids)
        chosen = sample_unique_ids(
            available(same_category_backfill),
            needed,
            pair_id,
            "negative_backfill_same_category",
            seed,
        )
        selected_ids.extend(chosen)
        used_ids.update(chosen)
        counts["negative_count_backfill_same_category"] = len(chosen)

    if len(selected_ids) < HARD_NEGATIVE_TARGET_COUNT:
        global_backfill = [
            candidate_id
            for candidate_id in turn_lookup
            if turn_lookup[candidate_id]["analysis_bucket"] == "headline_constructive"
            and turn_lookup[candidate_id]["episode_id"] != episode_id
        ]
        needed = HARD_NEGATIVE_TARGET_COUNT - len(selected_ids)
        chosen = sample_unique_ids(
            available(global_backfill),
            needed,
            pair_id,
            "negative_backfill_global",
            seed,
        )
        selected_ids.extend(chosen)
        used_ids.update(chosen)
        counts["negative_count_backfill_global"] = len(chosen)

    return selected_ids, counts, len(selected_ids) == HARD_NEGATIVE_TARGET_COUNT


def select_random_context_assumption_ids(
    pair_row: dict[str, Any],
    turn_indexes: dict[str, Any],
    selected_assumption_count: int,
    seed: int,
) -> tuple[list[str], bool]:
    if selected_assumption_count == 0:
        return [], True

    category = str(pair_row["category"])
    move_label = str(pair_row["turn_b_move_label"])
    episode_id = str(pair_row["episode_id"])
    assumption_lookup = turn_indexes["assumption_lookup"]
    assumptions_by_category_move = turn_indexes["assumptions_by_category_move"]
    assumptions_by_category = turn_indexes["assumptions_by_category"]
    assumptions_global_headline = turn_indexes["assumptions_global_headline"]
    excluded_assumption_ids = set(pair_row["turn_b_assumption_ids"])

    def assumption_candidates(ids: list[str]) -> list[str]:
        return [candidate_id for candidate_id in ids if candidate_id not in excluded_assumption_ids]

    random_pool = [
        candidate_id
        for candidate_id in assumptions_by_category_move.get((category, move_label), [])
        if assumption_lookup[candidate_id]["episode_id"] != episode_id
    ]
    random_ids = sample_unique_ids(
        assumption_candidates(random_pool),
        selected_assumption_count,
        str(pair_row["pair_id"]),
        "random_context_same_category_same_move",
        seed,
    )
    if len(random_ids) < selected_assumption_count:
        needed = selected_assumption_count - len(random_ids)
        backfill_pool = [
            candidate_id
            for candidate_id in assumptions_by_category.get(category, [])
            if assumption_lookup[candidate_id]["episode_id"] != episode_id
            and candidate_id not in random_ids
            and candidate_id not in excluded_assumption_ids
        ]
        random_ids.extend(
            sample_unique_ids(
                backfill_pool,
                needed,
                str(pair_row["pair_id"]),
                "random_context_same_category_backfill",
                seed,
            )
        )
    if len(random_ids) < selected_assumption_count:
        needed = selected_assumption_count - len(random_ids)
        global_pool = [
            candidate_id
            for candidate_id in assumptions_global_headline
            if assumption_lookup[candidate_id]["episode_id"] != episode_id
            and candidate_id not in random_ids
            and candidate_id not in excluded_assumption_ids
        ]
        random_ids.extend(
            sample_unique_ids(
                global_pool,
                needed,
                str(pair_row["pair_id"]),
                "random_context_global_backfill",
                seed,
            )
        )
    return random_ids, len(random_ids) == selected_assumption_count


def collect_random_context_texts(
    selected_pair_rows: list[dict[str, Any]],
    turn_indexes: dict[str, Any],
    seed: int,
) -> list[str]:
    texts: list[str] = []
    assumption_lookup = turn_indexes["assumption_lookup"]
    for pair_row in selected_pair_rows:
        if not bool(pair_row["eligible"]):
            continue
        claim_text = str(pair_row["turn_b_claim_text"])
        for selected_assumption_count in range(1, GREEDY_MAX_ASSUMPTIONS + 1):
            random_ids, random_pool_complete = select_random_context_assumption_ids(
                pair_row=pair_row,
                turn_indexes=turn_indexes,
                selected_assumption_count=selected_assumption_count,
                seed=seed,
            )
            if not random_pool_complete:
                continue
            random_texts = [str(assumption_lookup[random_id]["text"]) for random_id in random_ids]
            texts.append(build_context_text(claim_text, random_texts))
    return [text for text in texts if text.strip()]


def compose_normalized_mean(vectors: list[np.ndarray]) -> np.ndarray | None:
    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    composed = matrix.mean(axis=0)
    return normalize_vector(composed)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))


def win_rate_from_scores(positive_score: float, negative_scores: list[float]) -> float:
    if not negative_scores:
        return float("nan")
    wins = []
    for negative_score in negative_scores:
        delta = positive_score - negative_score
        if delta > SIM_TIE_EPSILON:
            wins.append(1.0)
        elif abs(delta) <= SIM_TIE_EPSILON:
            wins.append(0.5)
        else:
            wins.append(0.0)
    return float(np.mean(wins))


def greedy_select_assumptions(
    vec_a: np.ndarray,
    vec_claim: np.ndarray,
    claim_text: str,
    candidate_records: list[dict[str, Any]],
    negative_scores: list[float],
    whitened_text_to_vec: dict[str, np.ndarray | None],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidate_records)
    current_vector = vec_claim
    while remaining and len(selected) < GREEDY_MAX_ASSUMPTIONS:
        best_record: dict[str, Any] | None = None
        best_vector: np.ndarray | None = None
        best_rank_gain = 0.0
        best_similarity_gain = 0.0
        current_similarity = cosine_similarity(vec_a, current_vector)
        current_win_rate = win_rate_from_scores(current_similarity, negative_scores)
        for candidate in remaining:
            trial_ids = {record["id"] for record in selected}
            trial_ids.add(candidate["id"])
            trial_texts = [record["text"] for record in candidate_records if record["id"] in trial_ids]
            candidate_context_text = build_context_text(claim_text, trial_texts)
            candidate_vector = whitened_text_to_vec.get(candidate_context_text)
            if candidate_vector is None:
                continue
            candidate_similarity = cosine_similarity(vec_a, candidate_vector)
            candidate_win_rate = win_rate_from_scores(candidate_similarity, negative_scores)
            rank_gain = candidate_win_rate - current_win_rate
            similarity_gain = candidate_similarity - current_similarity
            improves_rank = rank_gain > best_rank_gain + SIM_TIE_EPSILON
            ties_rank = abs(rank_gain - best_rank_gain) <= SIM_TIE_EPSILON
            improves_similarity = similarity_gain > best_similarity_gain + SIM_TIE_EPSILON
            if improves_rank or (ties_rank and improves_similarity):
                best_record = candidate
                best_vector = candidate_vector
                best_rank_gain = rank_gain
                best_similarity_gain = similarity_gain
        if best_record is None:
            break
        if best_rank_gain <= SIM_TIE_EPSILON and best_similarity_gain <= SIM_TIE_EPSILON:
            break
        selected.append(best_record)
        remaining = [record for record in remaining if record["id"] != best_record["id"]]
        current_vector = best_vector if best_vector is not None else current_vector
    return selected, current_vector


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    value_column: str,
    seed_label: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    valid = df[df[value_column].notna()].copy()
    if valid.empty:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci_unstable": True,
            "cluster_count": 0,
        }
    valid["_cluster_key"] = valid["category"].astype(str) + "||" + valid["episode_id"].astype(str)
    clusters = [group[value_column].to_numpy(dtype=np.float64) for _, group in valid.groupby("_cluster_key", sort=False)]
    cluster_count = len(clusters)
    point_estimate = float(valid[value_column].mean())
    if cluster_count < DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS:
        return {
            "mean": point_estimate,
            "ci95_low": None,
            "ci95_high": None,
            "ci_unstable": True,
            "cluster_count": cluster_count,
        }
    rng = np.random.default_rng(build_seed_int(f"{seed_label}:{seed}"))
    draws_values = []
    for _ in range(draws):
        sampled_indices = rng.integers(0, cluster_count, size=cluster_count)
        sampled_values = np.concatenate([clusters[int(idx)] for idx in sampled_indices])
        draws_values.append(float(sampled_values.mean()))
    alpha = 1.0 - DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL
    return {
        "mean": point_estimate,
        "ci95_low": float(np.quantile(draws_values, alpha / 2.0)),
        "ci95_high": float(np.quantile(draws_values, 1.0 - alpha / 2.0)),
        "ci_unstable": False,
        "cluster_count": cluster_count,
    }


def build_group_summary(
    df: pd.DataFrame,
    group_column: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    retained = df[df["canonical_retained"] == True].copy()
    if retained.empty:
        return pd.DataFrame(columns=[group_column])
    for group_value, group_df in retained.groupby(group_column, sort=False, observed=False):
        row = {
            group_column: group_value,
            "pair_count": int(len(group_df)),
            "cluster_count": int(group_df[["category", "episode_id"]].drop_duplicates().shape[0]),
            "ci_unstable": bool(
                group_df[["category", "episode_id"]].drop_duplicates().shape[0]
                < DEFAULT_CLUSTER_BOOTSTRAP_MIN_CLUSTERS
            ),
        }
        metric_columns = [
            ("mean_bridge_lift", "bridge_lift"),
            ("mean_win_rate_claim", "win_rate_claim"),
            ("mean_win_rate_random_context", "win_rate_random_context"),
            ("mean_win_rate_context", "win_rate_context"),
            ("mean_bridge_advantage_over_random", "bridge_advantage_over_random"),
            ("mean_selected_assumption_count", "selected_assumption_count"),
        ]
        for output_column, metric_column in metric_columns:
            boot = cluster_bootstrap_mean(group_df, metric_column, f"{group_column}:{group_value}:{metric_column}", seed, DEFAULT_BOOTSTRAP_DRAWS)
            row[output_column] = boot["mean"]
            row[f"{output_column}_ci95_low"] = boot["ci95_low"]
            row[f"{output_column}_ci95_high"] = boot["ci95_high"]
            row[f"{output_column}_ci_unstable"] = boot["ci_unstable"]

        full_bag_valid = group_df[group_df["full_bag_context_valid"] == True].copy()
        row["full_bag_valid_pair_count"] = int(len(full_bag_valid))
        if full_bag_valid.empty:
            row["mean_win_rate_full_bag_context"] = None
            row["mean_win_rate_full_bag_context_ci95_low"] = None
            row["mean_win_rate_full_bag_context_ci95_high"] = None
            row["mean_win_rate_full_bag_context_ci_unstable"] = True
        else:
            boot = cluster_bootstrap_mean(
                full_bag_valid,
                "win_rate_full_bag_context",
                f"{group_column}:{group_value}:win_rate_full_bag_context",
                seed,
                DEFAULT_BOOTSTRAP_DRAWS,
            )
            row["mean_win_rate_full_bag_context"] = boot["mean"]
            row["mean_win_rate_full_bag_context_ci95_low"] = boot["ci95_low"]
            row["mean_win_rate_full_bag_context_ci95_high"] = boot["ci95_high"]
            row["mean_win_rate_full_bag_context_ci_unstable"] = boot["ci_unstable"]

        row["legacy_mean_sim_claim_raw"] = float(group_df["legacy_sim_claim_raw"].dropna().mean()) if group_df["legacy_sim_claim_raw"].notna().any() else None
        row["legacy_mean_sim_context_raw"] = float(group_df["legacy_sim_context_raw"].dropna().mean()) if group_df["legacy_sim_context_raw"].notna().any() else None
        row["legacy_mean_sim_unrelated_only_raw"] = float(group_df["legacy_sim_unrelated_only_raw"].dropna().mean()) if group_df["legacy_sim_unrelated_only_raw"].notna().any() else None
        row["legacy_mean_bridge_delta_raw"] = float(group_df["legacy_bridge_delta_raw"].dropna().mean()) if group_df["legacy_bridge_delta_raw"].notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bridge_lift_by_category(summary_df: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ordered = summary_df.sort_values("mean_bridge_lift", ascending=False).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    means = ordered["mean_bridge_lift"].astype(float).to_numpy()
    lower = ordered["mean_bridge_lift_ci95_low"].astype(float).to_numpy()
    upper = ordered["mean_bridge_lift_ci95_high"].astype(float).to_numpy()
    lower_err = np.where(np.isnan(lower), 0.0, means - lower)
    upper_err = np.where(np.isnan(upper), 0.0, upper - means)
    ax.errorbar(
        means,
        y_positions,
        xerr=np.vstack([lower_err, upper_err]),
        fmt="o",
        color="#1b9e77",
        ecolor="#1b9e77",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["category"].astype(str).tolist())
    ax.set_xlabel("Bridge Lift")
    ax.set_ylabel("Category")
    ax.set_title("Exp 1 v2 Bridge Lift by Category")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_ablation_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        comparisons = [
            ("Claim Only", row.mean_win_rate_claim, row.mean_win_rate_claim_ci95_low, row.mean_win_rate_claim_ci95_high),
            ("Random Context", row.mean_win_rate_random_context, row.mean_win_rate_random_context_ci95_low, row.mean_win_rate_random_context_ci95_high),
            ("Greedy Context", row.mean_win_rate_context, row.mean_win_rate_context_ci95_low, row.mean_win_rate_context_ci95_high),
            ("Full-Bag Context", row.mean_win_rate_full_bag_context, row.mean_win_rate_full_bag_context_ci95_low, row.mean_win_rate_full_bag_context_ci95_high),
        ]
        for comparison, mean_value, ci_low, ci_high in comparisons:
            rows.append(
                {
                    "category": row.category,
                    "comparison": comparison,
                    "mean_value": mean_value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def plot_ablation_by_category(summary_df: pd.DataFrame, path: Path) -> None:
    long_df = build_ablation_summary(summary_df)
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    category_order = summary_df.sort_values("mean_bridge_lift", ascending=False)["category"].astype(str).tolist()
    comparison_order = ["Claim Only", "Random Context", "Greedy Context", "Full-Bag Context"]
    palette = {
        "Claim Only": "#d95f02",
        "Random Context": "#7570b3",
        "Greedy Context": "#1b9e77",
        "Full-Bag Context": "#4c78a8",
    }
    for comparison in comparison_order:
        part = long_df[long_df["comparison"] == comparison].set_index("category").reindex(category_order).reset_index()
        y_positions = np.arange(len(category_order)) + (comparison_order.index(comparison) - 1.5) * 0.16
        means = part["mean_value"].astype(float).to_numpy()
        lower = part["ci95_low"].astype(float).to_numpy()
        upper = part["ci95_high"].astype(float).to_numpy()
        lower_err = np.where(np.isnan(lower), 0.0, means - lower)
        upper_err = np.where(np.isnan(upper), 0.0, upper - means)
        ax.errorbar(
            means,
            y_positions,
            xerr=np.vstack([lower_err, upper_err]),
            fmt="o",
            color=palette[comparison],
            ecolor=palette[comparison],
            capsize=2.5,
            label=comparison,
        )
    ax.set_yticks(np.arange(len(category_order)))
    ax.set_yticklabels(category_order)
    ax.set_xlabel("Win Rate")
    ax.set_ylabel("Category")
    ax.set_title("Exp 1 v2 Ablation by Category")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_legacy_cosine_distribution(df: pd.DataFrame, path: Path) -> None:
    retained = df[df["canonical_retained"] == True].copy()
    if retained.empty:
        empty_fig, empty_ax = plt.subplots(figsize=(11.5, 7))
        empty_ax.set_title("Exp 1 v2 Legacy Raw Cosine Diagnostics")
        empty_ax.set_xlabel("Raw Cosine Similarity")
        empty_ax.set_ylabel("Density")
        empty_fig.tight_layout()
        empty_fig.savefig(path, dpi=220)
        plt.close(empty_fig)
        return
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    bins = np.linspace(0, 1, 60)
    ax.hist(retained["legacy_sim_claim_raw"].dropna(), bins=bins, density=True, alpha=0.42, color="#d95f02", label="Claim Only")
    ax.hist(retained["legacy_sim_context_raw"].dropna(), bins=bins, density=True, alpha=0.48, color="#1b9e77", label="Assumption Context")
    ax.hist(retained["legacy_sim_unrelated_only_raw"].dropna(), bins=bins, density=True, alpha=0.42, color="#4c78a8", label="Unrelated Baseline")
    ax.set_title("Exp 1 v2 Legacy Raw Cosine Diagnostics")
    ax.set_xlabel("Raw Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_segment_summary(df: pd.DataFrame, analysis_bucket: str, seed: int) -> dict[str, Any]:
    eligible = df[(df["eligible"] == True) & (df["analysis_bucket"] == analysis_bucket)].copy()
    retained = eligible[eligible["canonical_retained"] == True].copy()
    summary: dict[str, Any] = {
        "eligible_pair_count": int(len(eligible)),
        "retained_pair_count": int(len(retained)),
        "retained_pair_rate": float(len(retained) / len(eligible)) if len(eligible) > 0 else None,
        "eligible_zero_assumption_pair_count": int((eligible["selected_assumption_count"].fillna(-1) == 0).sum()),
        "retained_zero_assumption_pair_count": int((retained["selected_assumption_count"].fillna(-1) == 0).sum()),
    }
    for metric_name, column_name in [
        ("mean_bridge_lift", "bridge_lift"),
        ("mean_win_rate_claim", "win_rate_claim"),
        ("mean_win_rate_random_context", "win_rate_random_context"),
        ("mean_win_rate_context", "win_rate_context"),
        ("mean_bridge_advantage_over_random", "bridge_advantage_over_random"),
    ]:
        boot = cluster_bootstrap_mean(retained, column_name, f"{analysis_bucket}:{column_name}", seed, DEFAULT_BOOTSTRAP_DRAWS)
        summary[metric_name] = boot["mean"]
        summary[f"{metric_name}_ci95_low"] = boot["ci95_low"]
        summary[f"{metric_name}_ci95_high"] = boot["ci95_high"]
        summary[f"{metric_name}_ci_unstable"] = boot["ci_unstable"]
    full_bag_valid = retained[retained["full_bag_context_valid"] == True].copy()
    if full_bag_valid.empty:
        summary["mean_win_rate_full_bag_context"] = None
        summary["mean_win_rate_full_bag_context_ci95_low"] = None
        summary["mean_win_rate_full_bag_context_ci95_high"] = None
        summary["mean_win_rate_full_bag_context_ci_unstable"] = True
    else:
        boot = cluster_bootstrap_mean(
            full_bag_valid,
            "win_rate_full_bag_context",
            f"{analysis_bucket}:win_rate_full_bag_context",
            seed,
            DEFAULT_BOOTSTRAP_DRAWS,
        )
        summary["mean_win_rate_full_bag_context"] = boot["mean"]
        summary["mean_win_rate_full_bag_context_ci95_low"] = boot["ci95_low"]
        summary["mean_win_rate_full_bag_context_ci95_high"] = boot["ci95_high"]
        summary["mean_win_rate_full_bag_context_ci_unstable"] = boot["ci_unstable"]
    return summary


def build_summary_payload(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    category_summary: pd.DataFrame,
    move_summary: pd.DataFrame,
    pair_csv: Path,
    category_csv: Path,
    move_csv: Path,
    main_plot_png: Path,
    ablation_plot_png: Path,
    diagnostic_plot_png: Path,
    analysis_stage: str,
    categories: list[str],
    selected_files: list[tuple[str, Path]],
    category_files: list[tuple[str, Path]],
    whitening_manifest: dict[str, Any],
    selected_episode_file_count: int | None,
    candidate_episode_file_count: int | None,
    extra_sections: dict[str, Any] | None,
) -> dict[str, Any]:
    headline_summary = build_segment_summary(df, "headline_constructive", args.seed)
    stress_summary = build_segment_summary(df, "stress_test", args.seed)
    eligibility_summary = {
        "eligible_pair_count": int((df["eligible"] == True).sum()),
        "excluded_pair_count": int((df["eligible"] != True).sum()),
        "excluded_non_substantive_pair_count": int((df["eligibility_exclusion_reason"] == "non_substantive_pair").sum()),
        "excluded_unsupported_move_bucket_count": int((df["eligibility_exclusion_reason"] == "unsupported_move_bucket").sum()),
    }
    eligible = df[df["eligible"] == True].copy()
    coverage = {
        "coverage_retained_pair_count": int((eligible["canonical_retained"] == True).sum()),
        "coverage_dropped_pair_count": int((eligible["canonical_retained"] != True).sum()),
        "coverage_retained_pair_rate": float((eligible["canonical_retained"] == True).mean()) if len(eligible) > 0 else None,
        "coverage_dropped_insufficient_unique_negatives_count": int((eligible["coverage_drop_reason"] == "insufficient_unique_negatives").sum()),
        "coverage_dropped_insufficient_random_context_assumptions_count": int((eligible["coverage_drop_reason"] == "insufficient_random_context_assumptions").sum()),
        "coverage_dropped_zero_norm_whitened_vector_count": int((eligible["coverage_drop_reason"] == "zero_norm_whitened_vector").sum()),
        "coverage_dropped_corrupt_or_unreadable_assumption_field_count": int((eligible["coverage_drop_reason"] == "corrupt_or_unreadable_assumption_field").sum()),
    }
    full_bag_valid = df[df["full_bag_context_valid"] == True]
    legacy_source = df[df["canonical_retained"] == True].copy()
    payload = {
        "experiment": "Experiment 1: The Relevance Bridge",
        "analysis_stage": analysis_stage,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "embedding_model_name": str(args.embedding_model_name),
        "embedding_batch_size": int(args.embedding_batch_size),
        "embedding_device": str(resolve_embedding_device(args.embedding_device)),
        "default_embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
        "recommended_qwen_embedding_model_name": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
        "default_embedding_device": DEFAULT_EMBEDDING_DEVICE,
        "target_embedding_max_length": DEFAULT_TARGET_EMBEDDING_MAX_LENGTH,
        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "bootstrap_confidence_level": DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
        "bootstrap_cluster_key": ["category", "episode_id"],
        "categories": categories,
        "selected_episode_file_count": int(selected_episode_file_count) if selected_episode_file_count is not None else int(len(selected_files)),
        "candidate_episode_file_count": int(candidate_episode_file_count) if candidate_episode_file_count is not None else int(len(category_files)),
        "headline_constructive": headline_summary,
        "stress_test": stress_summary,
        "legacy_diagnostics": {
            "legacy_mean_sim_claim_raw": float(legacy_source["legacy_sim_claim_raw"].dropna().mean()) if legacy_source["legacy_sim_claim_raw"].notna().any() else None,
            "legacy_mean_sim_context_raw": float(legacy_source["legacy_sim_context_raw"].dropna().mean()) if legacy_source["legacy_sim_context_raw"].notna().any() else None,
            "legacy_mean_sim_unrelated_only_raw": float(legacy_source["legacy_sim_unrelated_only_raw"].dropna().mean()) if legacy_source["legacy_sim_unrelated_only_raw"].notna().any() else None,
            "legacy_mean_bridge_delta_raw": float(legacy_source["legacy_bridge_delta_raw"].dropna().mean()) if legacy_source["legacy_bridge_delta_raw"].notna().any() else None,
        },
        "eligibility_summary": eligibility_summary,
        "coverage": coverage,
        "full_bag_ablation": {
            "full_bag_ablation_valid_pair_count": int(len(full_bag_valid)),
            "full_bag_ablation_invalid_pair_count": int(len(df[df["full_bag_context_valid"] == False])),
            "full_bag_ablation_invalid_zero_norm_count": int((df["ablation_drop_reason"] == "zero_norm_full_bag_context_vector").sum()),
        },
        "whitening_artifact": {
            "params_path": str(whitening_paths(resolve_output_dir(args.output_dir, args.embedding_model_name))[0]),
            "manifest_path": str(whitening_paths(resolve_output_dir(args.output_dir, args.embedding_model_name))[1]),
            "manifest_hash": whitening_manifest.get("manifest_hash"),
        },
        "outputs": {
            "pair_csv": str(pair_csv),
            "category_csv": str(category_csv),
            "move_csv": str(move_csv),
            "bridge_lift_by_category_png": str(main_plot_png),
            "ablation_by_category_png": str(ablation_plot_png),
            "legacy_cosine_distribution_png": str(diagnostic_plot_png),
        },
    }
    if extra_sections:
        payload.update(extra_sections)
    return payload


def serialize_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    serialized = df.copy()
    for column_name in ["selected_assumption_ids"]:
        if column_name in serialized.columns:
            serialized[column_name] = serialized[column_name].map(serialize_json_field)
    return serialized[PAIR_EXPORT_COLUMNS]


def score_pair_rows(
    pair_rows: list[dict[str, Any]],
    turn_indexes: dict[str, Any],
    raw_text_to_vec: dict[str, np.ndarray],
    whitened_text_to_vec: dict[str, np.ndarray | None],
    unrelated_sentences: list[str],
    seed: int,
    use_tqdm: bool,
) -> pd.DataFrame:
    scored_rows: list[dict[str, Any]] = []
    turn_lookup = turn_indexes["turn_lookup"]
    assumption_lookup = turn_indexes["assumption_lookup"]
    iterator = tqdm(pair_rows, desc="Scoring Exp 1 v2 pairs", disable=not use_tqdm)
    for original_row in iterator:
        row = dict(original_row)
        row["canonical_retained"] = None
        row["coverage_drop_reason"] = None
        row["negative_pool_complete"] = None
        row["random_context_pool_complete"] = None
        row["full_bag_context_valid"] = None
        row["ablation_drop_reason"] = None
        row["selected_assumption_count"] = None
        row["selected_assumption_ids"] = None
        row["selected_context_text"] = None
        row["negative_sample_count_actual"] = None
        row["negative_count_layer1_exact"] = None
        row["negative_count_layer2_gap3"] = None
        row["negative_count_layer2_gap2_backfill"] = None
        row["negative_count_layer3_exact"] = None
        row["negative_count_backfill_same_category"] = None
        row["negative_count_backfill_global"] = None
        row["win_rate_claim"] = None
        row["win_rate_random_context"] = None
        row["win_rate_context"] = None
        row["bridge_lift"] = None
        row["random_context_lift"] = None
        row["bridge_advantage_over_random"] = None
        row["win_rate_full_bag_context"] = None
        row["legacy_sim_claim_raw"] = None
        row["legacy_sim_context_raw"] = None
        row["legacy_sim_unrelated_only_raw"] = None
        row["legacy_bridge_delta_raw"] = None

        if not bool(row["eligible"]):
            scored_rows.append(row)
            continue

        negative_ids, negative_counts, negative_pool_complete = select_negative_ids(row, turn_indexes, seed)
        row["negative_sample_count_actual"] = int(len(negative_ids))
        for key, value in negative_counts.items():
            row[key] = int(value)
        row["negative_pool_complete"] = bool(negative_pool_complete)
        if not negative_pool_complete:
            row["canonical_retained"] = False
            row["random_context_pool_complete"] = None
            row["full_bag_context_valid"] = None
            row["coverage_drop_reason"] = "insufficient_unique_negatives"
            scored_rows.append(row)
            continue

        if bool(row["turn_b_assumption_field_corrupt"]):
            row["canonical_retained"] = False
            row["random_context_pool_complete"] = None
            row["full_bag_context_valid"] = None
            row["coverage_drop_reason"] = "corrupt_or_unreadable_assumption_field"
            scored_rows.append(row)
            continue

        vec_a_raw = normalize_vector(raw_text_to_vec[row["turn_a_text"]])
        vec_claim_raw = normalize_vector(raw_text_to_vec[row["turn_b_claim_text"]])
        vec_a = whitened_text_to_vec.get(row["turn_a_text"])
        vec_claim = whitened_text_to_vec.get(row["turn_b_claim_text"])
        negative_vecs = [whitened_text_to_vec.get(turn_lookup[negative_id]["claim_text"]) for negative_id in negative_ids]
        if vec_a is None or vec_claim is None or any(negative_vec is None for negative_vec in negative_vecs):
            row["canonical_retained"] = False
            row["random_context_pool_complete"] = None
            row["full_bag_context_valid"] = None
            row["coverage_drop_reason"] = "zero_norm_whitened_vector"
            scored_rows.append(row)
            continue

        negative_scores = [cosine_similarity(vec_a, negative_vec) for negative_vec in negative_vecs if negative_vec is not None]
        claim_score = cosine_similarity(vec_a, vec_claim)
        win_rate_claim = win_rate_from_scores(claim_score, negative_scores)

        assumption_records: list[dict[str, Any]] = []
        for assumption_id, assumption_text in zip(row["turn_b_assumption_ids"], row["turn_b_assumption_texts"]):
            if assumption_id not in assumption_lookup:
                continue
            assumption_records.append(
                {
                    "id": assumption_id,
                    "text": assumption_text,
                }
            )

        selected_assumptions, greedy_vec = greedy_select_assumptions(
            vec_a=vec_a,
            vec_claim=vec_claim,
            claim_text=str(row["turn_b_claim_text"]),
            candidate_records=assumption_records,
            negative_scores=negative_scores,
            whitened_text_to_vec=whitened_text_to_vec,
        )
        selected_assumption_ids = [record["id"] for record in selected_assumptions]
        selected_assumption_texts_by_id = {record["id"]: record["text"] for record in selected_assumptions}
        selected_assumption_texts = [
            assumption_text
            for assumption_id, assumption_text in zip(row["turn_b_assumption_ids"], row["turn_b_assumption_texts"])
            if assumption_id in selected_assumption_texts_by_id
        ]
        row["selected_assumption_count"] = int(len(selected_assumptions))
        row["selected_assumption_ids"] = selected_assumption_ids
        row["selected_context_text"] = build_context_text(str(row["turn_b_claim_text"]), selected_assumption_texts)

        if selected_assumptions and greedy_vec is None:
            row["canonical_retained"] = False
            row["random_context_pool_complete"] = None
            row["full_bag_context_valid"] = None
            row["coverage_drop_reason"] = "zero_norm_whitened_vector"
            scored_rows.append(row)
            continue

        context_vec = greedy_vec if selected_assumptions else vec_claim
        context_score = cosine_similarity(vec_a, context_vec)
        win_rate_context = win_rate_from_scores(context_score, negative_scores)

        if row["selected_assumption_count"] == 0:
            row["random_context_pool_complete"] = True
            win_rate_random_context = win_rate_claim
            random_context_score = claim_score
        else:
            random_ids, random_pool_complete = select_random_context_assumption_ids(
                pair_row=row,
                turn_indexes=turn_indexes,
                selected_assumption_count=int(row["selected_assumption_count"]),
                seed=seed,
            )
            if not random_pool_complete:
                row["canonical_retained"] = False
                row["random_context_pool_complete"] = False
                row["full_bag_context_valid"] = None
                row["coverage_drop_reason"] = "insufficient_random_context_assumptions"
                scored_rows.append(row)
                continue
            random_assumption_texts = [str(assumption_lookup[random_id]["text"]) for random_id in random_ids]
            random_context_text = build_context_text(str(row["turn_b_claim_text"]), random_assumption_texts)
            random_context_vec = whitened_text_to_vec.get(random_context_text)
            if random_context_vec is None:
                row["canonical_retained"] = False
                row["random_context_pool_complete"] = False
                row["full_bag_context_valid"] = None
                row["coverage_drop_reason"] = "zero_norm_whitened_vector"
                scored_rows.append(row)
                continue
            row["random_context_pool_complete"] = True
            random_context_score = cosine_similarity(vec_a, random_context_vec)
            win_rate_random_context = win_rate_from_scores(random_context_score, negative_scores)

        full_bag_context_valid = True
        ablation_drop_reason = None
        if row["candidate_assumption_count"] == 0:
            win_rate_full_bag_context = win_rate_claim
        else:
            full_bag_context_text = build_context_text(
                str(row["turn_b_claim_text"]),
                [str(text) for text in row["turn_b_assumption_texts"]],
            )
            full_bag_context_vec = whitened_text_to_vec.get(full_bag_context_text)
            if full_bag_context_vec is None:
                full_bag_context_valid = False
                ablation_drop_reason = "zero_norm_full_bag_context_vector"
                win_rate_full_bag_context = float("nan")
            else:
                win_rate_full_bag_context = win_rate_from_scores(
                    cosine_similarity(vec_a, full_bag_context_vec),
                    negative_scores,
                )

        if vec_a_raw is not None and vec_claim_raw is not None:
            baseline_texts = sample_unrelated_sentences(row["pair_id"], unrelated_sentences, seed)
            baseline_raw_vecs = [normalize_vector(raw_text_to_vec[text]) for text in baseline_texts]
            baseline_raw_vecs = [vec for vec in baseline_raw_vecs if vec is not None]
            if len(baseline_raw_vecs) == BASELINE_SENTENCE_SAMPLE_SIZE:
                baseline_raw = compose_normalized_mean(baseline_raw_vecs)
            else:
                baseline_raw = None
            legacy_context_raw = normalize_vector(raw_text_to_vec[row["selected_context_text"]])
            row["legacy_sim_claim_raw"] = cosine_similarity(vec_a_raw, vec_claim_raw)
            row["legacy_sim_context_raw"] = cosine_similarity(vec_a_raw, legacy_context_raw) if legacy_context_raw is not None else None
            row["legacy_sim_unrelated_only_raw"] = cosine_similarity(vec_a_raw, baseline_raw) if baseline_raw is not None else None
            if row["legacy_sim_context_raw"] is not None:
                row["legacy_bridge_delta_raw"] = row["legacy_sim_context_raw"] - row["legacy_sim_claim_raw"]

        row["canonical_retained"] = True
        row["coverage_drop_reason"] = None
        row["win_rate_claim"] = win_rate_claim
        row["win_rate_context"] = win_rate_context
        row["win_rate_random_context"] = win_rate_random_context
        row["bridge_lift"] = win_rate_context - win_rate_claim
        row["random_context_lift"] = win_rate_random_context - win_rate_claim
        row["bridge_advantage_over_random"] = win_rate_context - win_rate_random_context
        row["win_rate_full_bag_context"] = win_rate_full_bag_context
        row["full_bag_context_valid"] = full_bag_context_valid
        row["ablation_drop_reason"] = ablation_drop_reason
        scored_rows.append(row)
    return pd.DataFrame(scored_rows)


def finalize_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    finalized = df.copy()
    for column_name in NULLABLE_STATUS_BOOLEAN_COLUMNS:
        if column_name in finalized.columns:
            finalized[column_name] = finalized[column_name].where(finalized["eligible"] == True, None)
    if "eligible" in finalized.columns:
        finalized["eligible"] = finalized["eligible"].fillna(False).astype(bool)
    finalized["canonical_retained"] = finalized["canonical_retained"].fillna(False)
    finalized = finalized.sort_values(
        by=["category", "episode_id", "turn_a_idx", "turn_b_idx"],
        kind="stable",
    ).reset_index(drop=True)
    return finalized


def write_pair_csv(df: pd.DataFrame, path: Path) -> None:
    serialized = serialize_pair_frame(df)
    serialized.to_csv(path, index=False)


def coerce_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for column_name in NULLABLE_STATUS_BOOLEAN_COLUMNS:
        if column_name in coerced.columns:
            coerced[column_name] = coerce_nullable_bool(coerced[column_name])
    if "eligible" in coerced.columns:
        coerced["eligible"] = coerce_nullable_bool(coerced["eligible"]).fillna(False).astype(bool)
    if "canonical_retained" in coerced.columns:
        coerced["canonical_retained"] = coerce_nullable_bool(coerced["canonical_retained"]).fillna(False).astype(bool)
    for column_name in [
        "selected_assumption_count",
        "negative_sample_count_actual",
        "negative_count_layer1_exact",
        "negative_count_layer2_gap3",
        "negative_count_layer2_gap2_backfill",
        "negative_count_layer3_exact",
        "negative_count_backfill_same_category",
        "negative_count_backfill_global",
    ]:
        if column_name in coerced.columns:
            coerced[column_name] = pd.to_numeric(coerced[column_name], errors="coerce")
    for column_name in [
        "win_rate_claim",
        "win_rate_random_context",
        "win_rate_context",
        "bridge_lift",
        "random_context_lift",
        "bridge_advantage_over_random",
        "win_rate_full_bag_context",
        "legacy_sim_claim_raw",
        "legacy_sim_context_raw",
        "legacy_sim_unrelated_only_raw",
        "legacy_bridge_delta_raw",
    ]:
        if column_name in coerced.columns:
            coerced[column_name] = pd.to_numeric(coerced[column_name], errors="coerce")
    return coerced


def main() -> None:
    args = parse_args()
    mode_count = sum(
        int(enabled)
        for enabled in [
            args.prepare_whitening_only,
            args.prepare_whitening_patch_only,
            args.merge_whitening_patches_only,
        ]
    )
    if mode_count > 1:
        raise ValueError(
            "Choose only one Exp 1 mode flag: --prepare_whitening_only, "
            "--prepare_whitening_patch_only, or --merge_whitening_patches_only."
        )
    validate_patch_args(args.num_patches, args.patch_index, args.episodes_per_patch)
    model_output_dir = resolve_output_dir(args.output_dir, args.embedding_model_name)
    output_dir = resolve_patch_output_dir(model_output_dir, args.num_patches, args.patch_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = normalize_categories(args.input_dir, args.categories)
    use_tqdm = not args.no_tqdm
    unrelated_sentences = load_unrelated_sentences(UNRELATED_SENTENCES_PATH)
    category_files = collect_category_files(args.input_dir, categories, args.max_episodes_per_category)
    selected_files = select_patch_files(
        category_files=category_files,
        num_patches=args.num_patches,
        patch_index=args.patch_index,
        episodes_per_patch=args.episodes_per_patch,
    )
    if not selected_files:
        raise RuntimeError(
            f"No episode files selected for patch {args.patch_index} out of {args.num_patches}. "
            f"candidate_file_count={len(category_files)}"
        )
    if args.prepare_whitening_only:
        artifact_info = prepare_whitening_artifact(
            args=args,
            model_output_dir=model_output_dir,
            categories=categories,
            category_files=category_files,
            use_tqdm=use_tqdm,
        )
        print(json.dumps({"analysis_stage": "prepare_whitening_only", "artifact": artifact_info}, indent=2))
        return
    if args.prepare_whitening_patch_only:
        artifact_info = prepare_whitening_patch_moments(
            args=args,
            model_output_dir=model_output_dir,
            categories=categories,
            category_files=category_files,
            use_tqdm=use_tqdm,
        )
        print(json.dumps({"analysis_stage": "prepare_whitening_patch_only", "artifact": artifact_info}, indent=2))
        return
    if args.merge_whitening_patches_only:
        artifact_info = merge_whitening_patch_moments(
            args=args,
            model_output_dir=model_output_dir,
            categories=categories,
            category_files=category_files,
        )
        print(json.dumps({"analysis_stage": "merge_whitening_patches_only", "artifact": artifact_info}, indent=2))
        return

    mean, basis, scales, whitening_manifest = ensure_whitening_artifact(
        args=args,
        model_output_dir=model_output_dir,
        categories=categories,
        category_files=category_files,
        use_tqdm=use_tqdm,
    )
    global_turn_records, _, selected_pair_rows = build_global_records(
        category_files=category_files,
        selected_files=selected_files,
        use_tqdm=use_tqdm,
    )
    turn_indexes = build_turn_indexes(global_turn_records)
    eligible_selected_rows = [row for row in selected_pair_rows if bool(row["eligible"])]
    negative_texts: list[str] = []
    for pair_row in eligible_selected_rows:
        negative_ids, _, _ = select_negative_ids(pair_row, turn_indexes, args.seed)
        for negative_id in negative_ids:
            negative_texts.append(turn_indexes["turn_lookup"][negative_id]["claim_text"])
    candidate_context_texts = collect_candidate_context_texts(eligible_selected_rows)
    random_context_texts = collect_random_context_texts(
        selected_pair_rows=eligible_selected_rows,
        turn_indexes=turn_indexes,
        seed=args.seed,
    )
    texts_to_embed = list(
        dict.fromkeys(
            [
                row["turn_a_text"]
                for row in selected_pair_rows
                if row["turn_a_text"]
            ]
            + [
                row["turn_b_claim_text"]
                for row in selected_pair_rows
                if row["turn_b_claim_text"]
            ]
            + [
                assumption_text
                for row in eligible_selected_rows
                for assumption_text in row["turn_b_assumption_texts"]
            ]
            + candidate_context_texts
            + random_context_texts
            + negative_texts
            + unrelated_sentences
        )
    )
    raw_text_to_vec = embed_texts(
        texts=texts_to_embed,
        batch_size=args.embedding_batch_size,
        use_tqdm=use_tqdm,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
    )
    whitened_text_to_vec = apply_whitening(
        text_to_raw_vec=raw_text_to_vec,
        mean=mean,
        basis=basis,
        scales=scales,
    )
    scored_df = score_pair_rows(
        pair_rows=selected_pair_rows,
        turn_indexes=turn_indexes,
        raw_text_to_vec=raw_text_to_vec,
        whitened_text_to_vec=whitened_text_to_vec,
        unrelated_sentences=unrelated_sentences,
        seed=args.seed,
        use_tqdm=use_tqdm,
    )
    scored_df = finalize_pair_frame(scored_df)
    pair_csv = output_dir / "exp1_bridge_pairs.csv"
    summary_json = output_dir / "exp1_summary.json"
    write_pair_csv(scored_df, pair_csv)

    if args.num_patches > 1:
        patch_summary = build_summary_payload(
            args=args,
            output_dir=output_dir,
            df=scored_df,
            category_summary=pd.DataFrame(),
            move_summary=pd.DataFrame(),
            pair_csv=pair_csv,
            category_csv=output_dir / "exp1_bridge_by_category.csv",
            move_csv=output_dir / "exp1_bridge_by_move.csv",
            main_plot_png=output_dir / "exp1_bridge_lift_by_category.png",
            ablation_plot_png=output_dir / "exp1_ablation_by_category.png",
            diagnostic_plot_png=output_dir / "exp1_legacy_cosine_distribution.png",
            analysis_stage="patch_pair_scoring_only",
            categories=categories,
            selected_files=selected_files,
            category_files=category_files,
            whitening_manifest=whitening_manifest,
            selected_episode_file_count=len(selected_files),
            candidate_episode_file_count=len(category_files),
            extra_sections={
                "num_patches": int(args.num_patches),
                "patch_index": int(args.patch_index),
                "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
            },
        )
        summary_json.write_text(json.dumps(patch_summary, indent=2))
        logger.info("Done. Wrote patch-level Exp 1 v2 results to %s", output_dir)
        return

    category_summary = build_group_summary(
        scored_df[(scored_df["analysis_bucket"] == "headline_constructive") & (scored_df["canonical_retained"] == True)].copy(),
        "category",
        args.seed,
    )
    move_summary = build_group_summary(
        scored_df[scored_df["canonical_retained"] == True].copy(),
        "turn_b_move_label",
        args.seed,
    )
    category_csv = output_dir / "exp1_bridge_by_category.csv"
    move_csv = output_dir / "exp1_bridge_by_move.csv"
    main_plot_png = output_dir / "exp1_bridge_lift_by_category.png"
    ablation_plot_png = output_dir / "exp1_ablation_by_category.png"
    diagnostic_plot_png = output_dir / "exp1_legacy_cosine_distribution.png"
    category_summary.to_csv(category_csv, index=False)
    move_summary.to_csv(move_csv, index=False)
    plot_bridge_lift_by_category(category_summary, main_plot_png)
    plot_ablation_by_category(category_summary, ablation_plot_png)
    plot_legacy_cosine_distribution(scored_df, diagnostic_plot_png)
    summary = build_summary_payload(
        args=args,
        output_dir=output_dir,
        df=scored_df,
        category_summary=category_summary,
        move_summary=move_summary,
        pair_csv=pair_csv,
        category_csv=category_csv,
        move_csv=move_csv,
        main_plot_png=main_plot_png,
        ablation_plot_png=ablation_plot_png,
        diagnostic_plot_png=diagnostic_plot_png,
        analysis_stage="full_analysis",
        categories=categories,
        selected_files=selected_files,
        category_files=category_files,
        whitening_manifest=whitening_manifest,
        selected_episode_file_count=len(selected_files),
        candidate_episode_file_count=len(category_files),
        extra_sections=None,
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    logger.info("Done. Wrote Exp 1 v2 results to %s", output_dir)


if __name__ == "__main__":
    main()
