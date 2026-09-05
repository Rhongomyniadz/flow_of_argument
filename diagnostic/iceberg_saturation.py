from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence, cast

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("diagnostic/results")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SAMPLE_SIZE = 1_000
DECILE_COUNT = 10
SAMPLE_PER_DECILE = 100
SAMPLE_SEED = 42
ITEM_CAP = 10
WORD_PATTERN: re.Pattern[str] = re.compile(r"\w+")
CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)


UNCAPPED_PROMPT = """You are analyzing the content of different turns in a conversation. Your task is to separate and extract the explicit and implicit information in one turn.

CRITICAL DEFINITIONS:
- Explicit propositions: Direct statements or factual claims clearly expressed in the text of the turn.
- Assumptions: The premises that must hold for the speaker's stance to make sense or be coherent. These assumptions can include categories such as
  - causal assumptions
  - normative assumptions
  - epistemic assumptions
  - beliefs about the audience or world
  - goals
  - beliefs about what counts as knowledge/evidence/trustworthy sources
  - social/affective beliefs (trust, respect, authority, identity, morality) that justify the speaker's stance

TASK:
- Extract a list of propositions and a list of assumptions for the given turn.

RULES:
- Order each list from most to least salient for the turn's communicative intent
- Each explicit proposition must be an atomic statement with a numeric confidence score
- Each assumption must be an implicit belief, not a paraphrase of explicit propositions
- Do not duplicate content across explicit propositions and implicit assumptions.
- Generate results in JSON only. Do not include commentary, markdown, or extra keys. Double quotes only. No trailing commas.
- If there are no propositions or beliefs for a category, output an empty list.

OUTPUT FORMAT (strict JSON with exactly these keys):
{{
  "explicit_propositions": [
    {{"text": "...", "confidence": 0.95}},
    {{"text": "...", "confidence": 0.90}}
  ],
  "assumptions": [
    {{"text": "...", "confidence": 0.93}},
    {{"text": "...", "confidence": 0.88}}
  ]
}}

TASK:
Extract the list of propositions and list of assumptions for this speaker turn:
"{turn_text}"

"""


class DiagnosticError(RuntimeError):
    """Base error for an invalid diagnostic run."""


class InputValidationError(DiagnosticError):
    """Raised when an extraction-pipeline input artifact is invalid."""


class ExtractionParseError(DiagnosticError):
    """Raised when a model response does not satisfy the extraction schema."""


class GenerationTruncatedError(DiagnosticError):
    """Raised when a model response reaches the generation-token limit."""


@dataclass(frozen=True)
class ExtractionItem:
    text: str
    confidence: float


@dataclass(frozen=True)
class ExtractionResult:
    explicit_propositions: tuple[ExtractionItem, ...]
    assumptions: tuple[ExtractionItem, ...]


@dataclass(frozen=True)
class SourceTurn:
    turn_id: str
    category: str
    episode_id: str
    turn_idx: int
    source_path: str
    turn_text: str
    word_count: int
    original_explicit_count: int | None
    original_implicit_count: int | None


@dataclass(frozen=True)
class AssignedTurn:
    source: SourceTurn
    length_decile: int


@dataclass(frozen=True)
class GenerationRecord:
    turn_id: str
    raw_output: str
    finish_reason: str
    prompt_token_count: int
    output_token_count: int
    run_signature: str


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    download_dir: str | None
    tensor_parallel_size: int
    gpu_memory_utilization: float
    batch_size: int
    max_tokens: int
    max_model_len: int
    temperature: float
    top_p: float
    min_p: float
    top_k: int
    repetition_penalty: float
    seed: int


class CompletionOutput(Protocol):
    text: str
    token_ids: list[int]
    finish_reason: str | None


class RequestOutput(Protocol):
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]


class GenerateModel(Protocol):
    def generate(
        self,
        prompts: list[str],
        sampling_params: object,
        use_tqdm: bool,
    ) -> list[RequestOutput]: ...


class LLMConnector:
    """Typed connector around the optional vLLM runtime."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError(
                "Iceberg saturation inference requires vllm. Install the project's llm optional "
                "dependencies before running the diagnostic."
            ) from error

        self._model: object = LLM(
            model=config.model_name,
            download_dir=config.download_dir,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
        )
        self._sampling_params: object = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            seed=config.seed,
        )

    def generate(self, prompts: list[str], show_progress: bool) -> list[RequestOutput]:
        model = cast(GenerateModel, self._model)
        return model.generate(prompts, self._sampling_params, use_tqdm=show_progress)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def require_nonempty_string(value: object, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{context}: field '{field}' must be a nonempty string")
    return value.strip()


def require_integer(value: object, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(f"{context}: field '{field}' must be an integer")
    return value


def optional_list_count(row: dict[str, object], field: str, context: str) -> int | None:
    if field not in row:
        return None
    value = row[field]
    if not isinstance(value, list):
        raise InputValidationError(
            f"{context}: optional field '{field}' must be a list when present"
        )
    return len(value)


def normalize_episode_id(value: object, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InputValidationError(
            f"{context}: field 'episode_id' must be a nonempty string or integer"
        )
    normalized = str(value).strip()
    if not normalized:
        raise InputValidationError(f"{context}: field 'episode_id' must not be empty")
    return normalized


def parse_source_turn(
    value: object,
    category_hint: str,
    source_path: str,
    row_index: int,
) -> SourceTurn | None:
    context = f"{source_path}: row {row_index}"
    if not isinstance(value, dict):
        raise InputValidationError(f"{context}: expected an object, found {type(value).__name__}")
    row = cast(dict[str, object], value)

    # The directory layout (<category>/parsed/*.json) is the authoritative
    # source of category. Some original extraction rows omit "category" or
    # contain an empty value, so fall back to category_hint in that case.
    # If a row does provide a category, keep the strict mismatch check so a
    # genuinely misplaced/corrupted file still fails loudly.
    raw_category = row.get("category")
    if raw_category is None or (isinstance(raw_category, str) and not raw_category.strip()):
        category = category_hint
    else:
        category = require_nonempty_string(raw_category, "category", context)
        if category != category_hint:
            raise InputValidationError(
                f"{context}: category '{category}' does not match directory category '{category_hint}'"
            )
    episode_id = normalize_episode_id(row.get("episode_id"), context)
    turn_idx = require_integer(row.get("turn_idx"), "turn_idx", context)
    raw_text = row.get("turn_text")
    if not isinstance(raw_text, str):
        raise InputValidationError(f"{context}: field 'turn_text' must be a string")
    turn_text = raw_text.strip()
    if not turn_text:
        return None
    turn_id = f"{category}:{episode_id}:{turn_idx}"
    return SourceTurn(
        turn_id=turn_id,
        category=category,
        episode_id=episode_id,
        turn_idx=turn_idx,
        source_path=source_path,
        turn_text=turn_text,
        word_count=count_words(turn_text),
        original_explicit_count=optional_list_count(
            row,
            "explicit_propositions",
            context,
        ),
        original_implicit_count=optional_list_count(row, "assumptions", context),
    )


def discover_input_files(input_dir: Path) -> list[tuple[str, Path]]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Extraction output root does not exist or is not a directory: {input_dir}")
    discovered: list[tuple[str, Path]] = []
    for category_dir in sorted(input_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        parsed_dir = category_dir / "parsed"
        if not parsed_dir.is_dir():
            continue
        discovered.extend((category_dir.name, path) for path in sorted(parsed_dir.glob("*.json")))
    if not discovered:
        raise FileNotFoundError(
            f"No original extraction outputs matched '<category>/parsed/*.json' under {input_dir}"
        )
    return discovered


def progress_iterator(
    values: list[tuple[str, Path]],
    show_progress: bool,
) -> Iterable[tuple[str, Path]]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError(
            "Progress display requires tqdm. Install the project dependencies or pass --no_tqdm."
        ) from error
    return tqdm(values, desc="Loading original extraction turns")


def load_source_turns(input_dir: Path, show_progress: bool) -> tuple[list[SourceTurn], dict[str, object]]:
    files = discover_input_files(input_dir)
    turns: list[SourceTurn] = []
    blank_turn_count = 0
    category_turn_counts: Counter[str] = Counter()
    input_manifest: list[dict[str, str]] = []
    for category, path in progress_iterator(files, show_progress):
        relative_path = path.relative_to(input_dir).as_posix()
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise InputValidationError(
                f"{relative_path}: invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
            ) from error
        if not isinstance(payload, list):
            raise InputValidationError(
                f"{relative_path}: expected a JSON list, found {type(payload).__name__}"
            )
        for row_index, value in enumerate(payload):
            turn = parse_source_turn(value, category, relative_path, row_index)
            if turn is None:
                blank_turn_count += 1
                continue
            turns.append(turn)
            category_turn_counts[category] += 1
        input_manifest.append({"path": relative_path, "sha256": file_sha256(path)})
    ensure_unique_turn_ids(turns)
    if len(turns) < SAMPLE_SIZE:
        raise InputValidationError(
            f"The original extraction outputs contain {len(turns)} nonempty unique turns; "
            f"at least {SAMPLE_SIZE} are required"
        )
    audit: dict[str, object] = {
        "input_dir": str(input_dir),
        "input_file_count": len(files),
        "eligible_turn_count": len(turns),
        "blank_turn_count": blank_turn_count,
        "category_eligible_turn_counts": dict(sorted(category_turn_counts.items())),
        "input_manifest_sha256": canonical_sha256(input_manifest),
        "input_manifest_hash_method": "sha256(canonical JSON of sorted relative paths and file sha256 values)",
    }
    return turns, audit


def ensure_unique_turn_ids(turns: Sequence[SourceTurn]) -> None:
    counts = Counter(turn.turn_id for turn in turns)
    duplicates = sorted(turn_id for turn_id, count in counts.items() if count > 1)
    if duplicates:
        raise InputValidationError(
            "Duplicate original-extraction turn IDs were found; expected category, episode_id, and "
            f"turn_idx to be unique. First duplicates: {duplicates[:5]}"
        )


def stable_tie_key(turn_id: str, seed: int) -> str:
    return sha256_bytes(f"{seed}:{turn_id}".encode("utf-8"))


def assign_length_deciles(
    turns: Sequence[SourceTurn],
    decile_count: int,
    seed: int,
) -> list[AssignedTurn]:
    if decile_count < 2:
        raise ValueError(f"decile_count must be at least 2; received {decile_count}")
    if len(turns) < decile_count:
        raise ValueError(
            f"Cannot divide {len(turns)} turns into {decile_count} nonempty length strata"
        )
    ordered = sorted(
        turns,
        key=lambda turn: (turn.word_count, stable_tie_key(turn.turn_id, seed)),
    )
    assignments: list[AssignedTurn] = []
    for index, turn in enumerate(ordered):
        length_decile = min(decile_count, (index * decile_count) // len(ordered) + 1)
        assignments.append(AssignedTurn(source=turn, length_decile=length_decile))
    return assignments


def sample_from_deciles(
    assignments: Sequence[AssignedTurn],
    decile_count: int,
    sample_per_decile: int,
    seed: int,
) -> list[AssignedTurn]:
    if sample_per_decile < 1:
        raise ValueError(f"sample_per_decile must be positive; received {sample_per_decile}")
    sampled: list[AssignedTurn] = []
    for length_decile in range(1, decile_count + 1):
        stratum = [turn for turn in assignments if turn.length_decile == length_decile]
        if len(stratum) < sample_per_decile:
            raise InputValidationError(
                f"Length decile {length_decile} contains {len(stratum)} turns, fewer than the "
                f"required sample of {sample_per_decile}"
            )
        rng = random.Random(seed + length_decile)
        sampled.extend(rng.sample(stratum, sample_per_decile))
    sampled.sort(key=lambda turn: (turn.length_decile, turn.source.word_count, turn.source.turn_id))
    sampled_ids = [turn.source.turn_id for turn in sampled]
    if len(sampled_ids) != len(set(sampled_ids)):
        raise InputValidationError("The stratified sample contains duplicate turn IDs")
    expected_size = decile_count * sample_per_decile
    if len(sampled) != expected_size:
        raise InputValidationError(
            f"Stratified sampling produced {len(sampled)} turns; expected {expected_size}"
        )
    return sampled


def sample_payload(sample: Sequence[AssignedTurn]) -> list[dict[str, object]]:
    return [
        {
            "turn_id": turn.source.turn_id,
            "category": turn.source.category,
            "episode_id": turn.source.episode_id,
            "turn_idx": turn.source.turn_idx,
            "source_path": turn.source.source_path,
            "word_count": turn.source.word_count,
            "length_decile": turn.length_decile,
            "original_explicit_count": turn.source.original_explicit_count,
            "original_implicit_count": turn.source.original_implicit_count,
            "turn_text": turn.source.turn_text,
        }
        for turn in sample
    ]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(path, text + "\n")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def ensure_sample_checkpoint(path: Path, sample: Sequence[AssignedTurn]) -> str:
    payload = sample_payload(sample)
    sample_hash = canonical_sha256(payload)
    contents = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in payload
    )
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != contents:
            raise InputValidationError(
                f"Existing sample checkpoint does not match the deterministic sample: {path}. "
                "Move or remove it before starting a different diagnostic run."
            )
        return sample_hash
    atomic_write_text(path, contents)
    return sample_hash


def prompt_for_turn(turn_text: str) -> str:
    return UNCAPPED_PROMPT.format(turn_text=turn_text)


def run_signature(config: ModelConfig, sample_hash: str) -> str:
    return canonical_sha256(
        {
            "model_config": asdict(config),
            "prompt_sha256": sha256_bytes(UNCAPPED_PROMPT.encode("utf-8")),
            "sample_sha256": sample_hash,
        }
    )


def parse_checkpoint_integer(value: object, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputValidationError(f"{context}: field '{field}' must be a nonnegative integer")
    return value


def parse_generation_checkpoint_row(
    value: object,
    context: str,
) -> GenerationRecord:
    if not isinstance(value, dict):
        raise InputValidationError(f"{context}: expected a JSON object")
    row = cast(dict[str, object], value)
    return GenerationRecord(
        turn_id=require_nonempty_string(row.get("turn_id"), "turn_id", context),
        raw_output=require_nonempty_string(row.get("raw_output"), "raw_output", context),
        finish_reason=require_nonempty_string(
            row.get("finish_reason"),
            "finish_reason",
            context,
        ),
        prompt_token_count=parse_checkpoint_integer(
            row.get("prompt_token_count"),
            "prompt_token_count",
            context,
        ),
        output_token_count=parse_checkpoint_integer(
            row.get("output_token_count"),
            "output_token_count",
            context,
        ),
        run_signature=require_nonempty_string(
            row.get("run_signature"),
            "run_signature",
            context,
        ),
    )


def validate_generation_record(record: GenerationRecord) -> None:
    if not record.raw_output.strip():
        raise DiagnosticError(f"Turn {record.turn_id} returned an empty model response")
    if record.finish_reason == "length":
        raise GenerationTruncatedError(
            f"Turn {record.turn_id} reached the generation-token limit after "
            f"{record.output_token_count} output tokens. Increase --max_tokens, remove the raw "
            "checkpoint, and rerun; a truncated response cannot enter the saturation verdict."
        )
    if record.finish_reason != "stop":
        raise DiagnosticError(
            f"Turn {record.turn_id} has unsupported finish_reason={record.finish_reason!r}; "
            "expected 'stop'"
        )


def load_generation_checkpoint(
    path: Path,
    expected_turn_ids: set[str],
    expected_signature: str,
) -> dict[str, GenerationRecord]:
    if not path.exists():
        return {}
    records: dict[str, GenerationRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            context = f"{path}: line {line_number}"
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise InputValidationError(
                    f"{context}: invalid checkpoint JSON: {error.msg}"
                ) from error
            record = parse_generation_checkpoint_row(value, context)
            if record.turn_id not in expected_turn_ids:
                raise InputValidationError(
                    f"{context}: turn_id '{record.turn_id}' is not in the current sample"
                )
            if record.run_signature != expected_signature:
                raise InputValidationError(
                    f"{context}: run signature does not match the current sample, prompt, model, "
                    "and decoding settings"
                )
            if record.turn_id in records:
                raise InputValidationError(
                    f"{context}: duplicate generation checkpoint for turn_id '{record.turn_id}'"
                )
            validate_generation_record(record)
            records[record.turn_id] = record
    return records


def append_generation_records(path: Path, records: Sequence[GenerationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
        handle.flush()


def generation_records_from_outputs(
    batch: Sequence[AssignedTurn],
    outputs: Sequence[RequestOutput],
    signature: str,
) -> list[GenerationRecord]:
    if len(outputs) != len(batch):
        raise DiagnosticError(
            f"vLLM returned {len(outputs)} responses for a batch of {len(batch)} prompts"
        )
    records: list[GenerationRecord] = []
    for turn, output in zip(batch, outputs):
        if len(output.outputs) != 1:
            raise DiagnosticError(
                f"Turn {turn.source.turn_id} returned {len(output.outputs)} completions; expected 1"
            )
        completion = output.outputs[0]
        finish_reason = completion.finish_reason or ""
        record = GenerationRecord(
            turn_id=turn.source.turn_id,
            raw_output=completion.text.strip(),
            finish_reason=finish_reason,
            prompt_token_count=len(output.prompt_token_ids),
            output_token_count=len(completion.token_ids),
            run_signature=signature,
        )
        records.append(record)
    return records


def pending_batches(
    sample: Sequence[AssignedTurn],
    completed_turn_ids: set[str],
    batch_size: int,
) -> Iterator[list[AssignedTurn]]:
    pending = [turn for turn in sample if turn.source.turn_id not in completed_turn_ids]
    for start in range(0, len(pending), batch_size):
        yield pending[start : start + batch_size]


def generate_missing_records(
    sample: Sequence[AssignedTurn],
    completed: dict[str, GenerationRecord],
    checkpoint_path: Path,
    config: ModelConfig,
    signature: str,
    show_progress: bool,
) -> dict[str, GenerationRecord]:
    missing_count = len(sample) - len(completed)
    if missing_count == 0:
        return completed
    logger.info("Starting uncapped extraction inference", extra={"pending_turn_count": missing_count})
    connector = LLMConnector(config)
    for batch in pending_batches(sample, set(completed), config.batch_size):
        prompts = [prompt_for_turn(turn.source.turn_text) for turn in batch]
        outputs = connector.generate(prompts, show_progress)
        records = generation_records_from_outputs(batch, outputs, signature)
        append_generation_records(checkpoint_path, records)
        for record in records:
            validate_generation_record(record)
            completed[record.turn_id] = record
    return completed


def json_object_candidates(text: str) -> Iterator[dict[str, object]]:
    seen: set[str] = set()
    candidates: list[str] = [text.strip()]
    candidates.extend(match.group(1).strip() for match in CODE_FENCE_PATTERN.finditer(text))
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            _, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        candidates.append(text[index:end])
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            value: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield cast(dict[str, object], value)


def normalize_extraction_item(value: object, context: str) -> ExtractionItem:
    if not isinstance(value, dict):
        raise ExtractionParseError(f"{context}: each extraction item must be an object")
    item = cast(dict[str, object], value)
    text = item.get("text")
    confidence = item.get("confidence")
    if not isinstance(text, str) or not text.strip():
        raise ExtractionParseError(f"{context}: extraction item text must be a nonempty string")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ExtractionParseError(f"{context}: extraction item confidence must be numeric")
    normalized_confidence = float(confidence)
    if not math.isfinite(normalized_confidence) or not 0.0 <= normalized_confidence <= 1.0:
        raise ExtractionParseError(
            f"{context}: extraction item confidence must be finite and between 0 and 1"
        )
    return ExtractionItem(text=text.strip(), confidence=normalized_confidence)


def normalize_extraction_list(value: object, context: str) -> tuple[ExtractionItem, ...]:
    if not isinstance(value, list):
        raise ExtractionParseError(f"{context}: expected a JSON list")
    normalized: list[ExtractionItem] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(value):
        item = normalize_extraction_item(raw_item, f"{context}[{index}]")
        key = item.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return tuple(normalized)


def parse_uncapped_extraction(text: str, context: str) -> ExtractionResult:
    if not text.strip():
        raise ExtractionParseError(f"{context}: model returned an empty response")
    schema_errors: list[str] = []
    for candidate in json_object_candidates(text):
        if "explicit_propositions" not in candidate or "assumptions" not in candidate:
            continue
        try:
            explicit = normalize_extraction_list(
                candidate["explicit_propositions"],
                f"{context}.explicit_propositions",
            )
            assumptions = normalize_extraction_list(
                candidate["assumptions"],
                f"{context}.assumptions",
            )
        except ExtractionParseError as error:
            schema_errors.append(str(error))
            continue
        explicit_keys = {item.text.casefold() for item in explicit}
        deduplicated_assumptions = tuple(
            item for item in assumptions if item.text.casefold() not in explicit_keys
        )
        return ExtractionResult(
            explicit_propositions=explicit,
            assumptions=deduplicated_assumptions,
        )
    detail = schema_errors[0] if schema_errors else "no JSON object contained both required lists"
    raise ExtractionParseError(f"{context}: could not parse extraction response: {detail}")


def capped_count(value: int, cap: int) -> int:
    if value < 0:
        raise ValueError(f"Count must be nonnegative; received {value}")
    if cap < 1:
        raise ValueError(f"Cap must be positive; received {cap}")
    return min(value, cap)


def iceberg_ratio(explicit_count: int, implicit_count: int) -> float:
    if explicit_count < 0 or implicit_count < 0:
        raise ValueError(
            "Iceberg-ratio counts must be nonnegative; "
            f"received explicit={explicit_count}, implicit={implicit_count}"
        )
    return float(explicit_count) / float(implicit_count + 1)


def optional_original_ratio(turn: SourceTurn) -> float | None:
    if turn.original_explicit_count is None or turn.original_implicit_count is None:
        return None
    return iceberg_ratio(turn.original_explicit_count, turn.original_implicit_count)


def build_turn_result(
    turn: AssignedTurn,
    generation: GenerationRecord,
    extraction: ExtractionResult,
    cap: int,
) -> dict[str, object]:
    explicit_count = len(extraction.explicit_propositions)
    implicit_count = len(extraction.assumptions)
    uncapped_ratio = iceberg_ratio(explicit_count, implicit_count)
    recapped_ratio = iceberg_ratio(
        capped_count(explicit_count, cap),
        capped_count(implicit_count, cap),
    )
    return {
        "turn_id": turn.source.turn_id,
        "category": turn.source.category,
        "episode_id": turn.source.episode_id,
        "turn_idx": turn.source.turn_idx,
        "source_path": turn.source.source_path,
        "word_count": turn.source.word_count,
        "length_decile": turn.length_decile,
        "original_explicit_count": turn.source.original_explicit_count,
        "original_implicit_count": turn.source.original_implicit_count,
        "original_iceberg_ratio": optional_original_ratio(turn.source),
        "uncapped_explicit_count": explicit_count,
        "uncapped_implicit_count": implicit_count,
        "explicit_ge_11": explicit_count >= cap + 1,
        "implicit_ge_11": implicit_count >= cap + 1,
        "either_ge_11": explicit_count >= cap + 1 or implicit_count >= cap + 1,
        "recapped_explicit_count": capped_count(explicit_count, cap),
        "recapped_implicit_count": capped_count(implicit_count, cap),
        "uncapped_iceberg_ratio": uncapped_ratio,
        "recapped_iceberg_ratio": recapped_ratio,
        "ratio_difference": uncapped_ratio - recapped_ratio,
        "absolute_ratio_difference": abs(uncapped_ratio - recapped_ratio),
        "ratio_changed_by_cap": not math.isclose(
            uncapped_ratio,
            recapped_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "prompt_token_count": generation.prompt_token_count,
        "output_token_count": generation.output_token_count,
        "finish_reason": generation.finish_reason,
    }


def build_turn_results(
    sample: Sequence[AssignedTurn],
    generations: dict[str, GenerationRecord],
    cap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for turn in sample:
        generation = generations.get(turn.source.turn_id)
        if generation is None:
            raise DiagnosticError(f"Missing generation for sampled turn {turn.source.turn_id}")
        validate_generation_record(generation)
        extraction = parse_uncapped_extraction(
            generation.raw_output,
            f"turn {turn.source.turn_id}",
        )
        rows.append(build_turn_result(turn, generation, extraction, cap))
    frame = pd.DataFrame(rows)
    if len(frame) != len(sample):
        raise DiagnosticError(f"Built {len(frame)} result rows for {len(sample)} sampled turns")
    if frame["turn_id"].duplicated().any():
        duplicates = frame.loc[frame["turn_id"].duplicated(), "turn_id"].tolist()
        raise DiagnosticError(f"Duplicate turn IDs in diagnostic results: {duplicates[:5]}")
    return frame.sort_values(
        ["length_decile", "word_count", "turn_id"],
        kind="stable",
    ).reset_index(drop=True)


def finite_series_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise DiagnosticError("Summary values must be nonempty and finite")
    return {
        "mean": float(np.mean(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def original_at_cap_rate(values: pd.Series, cap: int) -> tuple[int, int, float | None]:
    valid = values.dropna().astype(int)
    if valid.empty:
        return 0, 0, None
    count = int((valid == cap).sum())
    return count, int(len(valid)), float(count / len(valid))


def build_decile_summary_frame(
    turns: pd.DataFrame,
    decile_count: int,
    cap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for length_decile in range(1, decile_count + 1):
        group = turns.loc[turns["length_decile"] == length_decile].copy()
        if group.empty:
            raise DiagnosticError(f"No sampled turns were found in length decile {length_decile}")
        explicit = finite_series_summary(group["uncapped_explicit_count"])
        implicit = finite_series_summary(group["uncapped_implicit_count"])
        uncapped_ratio = finite_series_summary(group["uncapped_iceberg_ratio"])
        recapped_ratio = finite_series_summary(group["recapped_iceberg_ratio"])
        original_explicit_at_cap, original_explicit_valid, original_explicit_rate = (
            original_at_cap_rate(group["original_explicit_count"], cap)
        )
        original_implicit_at_cap, original_implicit_valid, original_implicit_rate = (
            original_at_cap_rate(group["original_implicit_count"], cap)
        )
        rows.append(
            {
                "length_decile": length_decile,
                "sample_size": int(len(group)),
                "word_count_min": int(group["word_count"].min()),
                "word_count_median": float(group["word_count"].median()),
                "word_count_max": int(group["word_count"].max()),
                "original_explicit_valid_count": original_explicit_valid,
                "original_explicit_at_10_count": original_explicit_at_cap,
                "original_explicit_at_10_rate": original_explicit_rate,
                "original_implicit_valid_count": original_implicit_valid,
                "original_implicit_at_10_count": original_implicit_at_cap,
                "original_implicit_at_10_rate": original_implicit_rate,
                "uncapped_explicit_mean": explicit["mean"],
                "uncapped_explicit_q25": explicit["q25"],
                "uncapped_explicit_median": explicit["median"],
                "uncapped_explicit_q75": explicit["q75"],
                "uncapped_explicit_max": explicit["max"],
                "uncapped_implicit_mean": implicit["mean"],
                "uncapped_implicit_q25": implicit["q25"],
                "uncapped_implicit_median": implicit["median"],
                "uncapped_implicit_q75": implicit["q75"],
                "uncapped_implicit_max": implicit["max"],
                "explicit_ge_11_count": int(group["explicit_ge_11"].sum()),
                "explicit_ge_11_rate": float(group["explicit_ge_11"].mean()),
                "implicit_ge_11_count": int(group["implicit_ge_11"].sum()),
                "implicit_ge_11_rate": float(group["implicit_ge_11"].mean()),
                "either_ge_11_count": int(group["either_ge_11"].sum()),
                "either_ge_11_rate": float(group["either_ge_11"].mean()),
                "uncapped_ratio_q25": uncapped_ratio["q25"],
                "uncapped_ratio_median": uncapped_ratio["median"],
                "uncapped_ratio_q75": uncapped_ratio["q75"],
                "recapped_ratio_q25": recapped_ratio["q25"],
                "recapped_ratio_median": recapped_ratio["median"],
                "recapped_ratio_q75": recapped_ratio["q75"],
                "ratio_changed_by_cap_count": int(group["ratio_changed_by_cap"].sum()),
                "ratio_changed_by_cap_rate": float(group["ratio_changed_by_cap"].mean()),
                "mean_absolute_ratio_difference": float(
                    group["absolute_ratio_difference"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def category_composition(turns: pd.DataFrame) -> dict[str, object]:
    overall = turns["category"].value_counts().sort_index()
    by_decile = pd.crosstab(turns["length_decile"], turns["category"])
    return {
        "overall": {str(key): int(value) for key, value in overall.items()},
        "by_length_decile": {
            str(int(index)): {str(key): int(value) for key, value in row.items()}
            for index, row in by_decile.iterrows()
        },
    }


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return cast(list[dict[str, object]], normalized.to_dict(orient="records"))


def diagnostic_verdict(turns: pd.DataFrame, cap: int, top_decile: int) -> dict[str, object]:
    top = turns.loc[turns["length_decile"] == top_decile]
    if top.empty:
        raise DiagnosticError(f"No turns are available in the top length decile {top_decile}")
    explicit_count = int(top["explicit_ge_11"].sum())
    implicit_count = int(top["implicit_ge_11"].sum())
    either_count = int(top["either_ge_11"].sum())
    ratio_changed_count = int(top["ratio_changed_by_cap"].sum())
    above_cap = either_count > 0
    cap_changes_ratio = ratio_changed_count > 0
    supports_cap_contribution = above_cap and cap_changes_ratio
    if supports_cap_contribution:
        interpretation = (
            "Top-decile uncapped extraction exceeds 10 items and synthetically recapping the same "
            "outputs changes their iceberg ratios. This supports the max-10 limit as a mechanical "
            "contributor to flattening, without establishing that it is the only cause."
        )
    elif above_cap:
        interpretation = (
            "Top-decile extraction exceeds 10 items, but recapping the sampled outputs does not "
            "change their iceberg ratios; the count ceiling is present without a demonstrated "
            "ratio effect in this sample."
        )
    else:
        interpretation = (
            "No top-decile sampled output exceeds 10 explicit propositions or implicit assumptions; "
            "this sample does not demonstrate that the max-10 limit causes the observed flattening."
        )
    return {
        "item_cap_tested": cap,
        "top_length_decile": top_decile,
        "top_decile_sample_size": int(len(top)),
        "top_decile_word_count_min": int(top["word_count"].min()),
        "top_decile_word_count_median": float(top["word_count"].median()),
        "top_decile_word_count_max": int(top["word_count"].max()),
        "explicit_ge_11_observed": explicit_count > 0,
        "explicit_ge_11_count": explicit_count,
        "explicit_ge_11_rate": float(explicit_count / len(top)),
        "explicit_max": int(top["uncapped_explicit_count"].max()),
        "implicit_ge_11_observed": implicit_count > 0,
        "implicit_ge_11_count": implicit_count,
        "implicit_ge_11_rate": float(implicit_count / len(top)),
        "implicit_max": int(top["uncapped_implicit_count"].max()),
        "either_ge_11_observed": above_cap,
        "either_ge_11_count": either_count,
        "either_ge_11_rate": float(either_count / len(top)),
        "ratio_changed_by_cap_observed": cap_changes_ratio,
        "ratio_changed_by_cap_count": ratio_changed_count,
        "ratio_changed_by_cap_rate": float(ratio_changed_count / len(top)),
        "cap_contribution_to_flattening_supported": supports_cap_contribution,
        "interpretation": interpretation,
    }


def build_summary(
    turns: pd.DataFrame,
    deciles: pd.DataFrame,
    input_audit: dict[str, object],
    config: ModelConfig,
    sample_hash: str,
    signature: str,
) -> dict[str, object]:
    finish_reasons = turns["finish_reason"].value_counts().sort_index()
    return {
        "diagnostic": "Iceberg extraction saturation",
        "population": "Every nonempty merged turn written under data/*/parsed by the first extraction pipeline",
        "length_definition": r"Number of Unicode regex matches for \w+ in turn_text",
        "sampling": {
            "sample_size": SAMPLE_SIZE,
            "length_decile_count": DECILE_COUNT,
            "sample_per_decile": SAMPLE_PER_DECILE,
            "seed": SAMPLE_SEED,
            "tie_policy": "sort by word_count, then sha256(seed:turn_id), before equal-frequency assignment",
            "sample_sha256": sample_hash,
        },
        "extraction": {
            "numerical_item_limit": None,
            "prompt_sha256": sha256_bytes(UNCAPPED_PROMPT.encode("utf-8")),
            "model_config": asdict(config),
            "run_signature": signature,
            "parsed_response_count": int(len(turns)),
            "finish_reason_counts": {
                str(key): int(value) for key, value in finish_reasons.items()
            },
            "total_prompt_tokens": int(turns["prompt_token_count"].sum()),
            "total_output_tokens": int(turns["output_token_count"].sum()),
            "token_truncated_response_count": 0,
        },
        "input_audit": input_audit,
        "sample_category_composition": category_composition(turns),
        "decile_summary_records": json_records(deciles),
        "verdict": diagnostic_verdict(turns, ITEM_CAP, DECILE_COUNT),
        "outputs": {
            "turns": "iceberg_saturation_turns.csv",
            "by_decile": "iceberg_saturation_by_decile.csv",
            "summary": "iceberg_saturation_summary.json",
            "plot_png": "iceberg_saturation.png",
            "plot_pdf": "iceberg_saturation.pdf",
        },
    }


def save_plot(deciles: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Diagnostic plotting requires matplotlib. Install the project dependencies."
        ) from error

    x = deciles["length_decile"].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    count_axis = axes[0]
    for prefix, label, color in (
        ("uncapped_explicit", "Explicit propositions", "#1f77b4"),
        ("uncapped_implicit", "Implicit assumptions", "#d95f02"),
    ):
        median = deciles[f"{prefix}_median"].to_numpy(dtype=float)
        lower = deciles[f"{prefix}_q25"].to_numpy(dtype=float)
        upper = deciles[f"{prefix}_q75"].to_numpy(dtype=float)
        count_axis.plot(x, median, marker="o", linewidth=2.0, color=color, label=label)
        count_axis.fill_between(x, lower, upper, color=color, alpha=0.18)
    count_axis.axhline(ITEM_CAP, color="#444444", linestyle="--", linewidth=1.2, label="Old cap = 10")
    count_axis.set_title("Uncapped extraction counts")
    count_axis.set_xlabel("Word-length decile")
    count_axis.set_ylabel("Median count (IQR)")
    count_axis.legend(frameon=False, fontsize=8)

    rate_axis = axes[1]
    width = 0.34
    rate_axis.bar(
        x - width / 2.0,
        100.0 * deciles["explicit_ge_11_rate"].to_numpy(dtype=float),
        width=width,
        color="#1f77b4",
        label="Explicit",
    )
    rate_axis.bar(
        x + width / 2.0,
        100.0 * deciles["implicit_ge_11_rate"].to_numpy(dtype=float),
        width=width,
        color="#d95f02",
        label="Implicit",
    )
    rate_axis.set_title("Turns exceeding the old cap")
    rate_axis.set_xlabel("Word-length decile")
    rate_axis.set_ylabel("Turns with at least 11 items (%)")
    rate_axis.legend(frameon=False, fontsize=8)

    ratio_axis = axes[2]
    for prefix, label, color in (
        ("uncapped_ratio", "Uncapped", "#2ca02c"),
        ("recapped_ratio", "Same outputs recapped at 10", "#6a3d9a"),
    ):
        median = deciles[f"{prefix}_median"].to_numpy(dtype=float)
        lower = deciles[f"{prefix}_q25"].to_numpy(dtype=float)
        upper = deciles[f"{prefix}_q75"].to_numpy(dtype=float)
        ratio_axis.plot(x, median, marker="o", linewidth=2.0, color=color, label=label)
        ratio_axis.fill_between(x, lower, upper, color=color, alpha=0.15)
    ratio_axis.set_title("Mechanical effect on iceberg ratio")
    ratio_axis.set_xlabel("Word-length decile")
    ratio_axis.set_ylabel("Median explicit / (implicit + 1) (IQR)")
    ratio_axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(np.arange(1, DECILE_COUNT + 1))
        axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle(
        "Iceberg extraction saturation diagnostic: 100 turns per word-length decile",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "iceberg_saturation.png"
    pdf_path = output_dir / "iceberg_saturation.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def validate_model_config(config: ModelConfig) -> None:
    if config.tensor_parallel_size < 1:
        raise ValueError("--tensor_parallel_size must be positive")
    if not 0.0 < config.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu_memory_utilization must be in (0, 1]")
    if config.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if config.max_tokens < 1:
        raise ValueError("--max_tokens must be positive")
    if config.max_model_len <= config.max_tokens:
        raise ValueError("--max_model_len must be greater than --max_tokens")
    if config.temperature < 0.0:
        raise ValueError("--temperature must be nonnegative")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError("--top_p must be in (0, 1]")
    if not 0.0 <= config.min_p <= 1.0:
        raise ValueError("--min_p must be in [0, 1]")
    if config.top_k < 0:
        raise ValueError("--top_k must be nonnegative")
    if config.repetition_penalty <= 0.0:
        raise ValueError("--repetition_penalty must be positive")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 1,000 original extraction turns by word-length decile and rerun uncapped "
            "explicit/implicit extraction."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--download_dir", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args(argv)


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig(
        model_name=str(args.model_name),
        download_dir=str(args.download_dir) if args.download_dir is not None else None,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        batch_size=int(args.batch_size),
        max_tokens=int(args.max_tokens),
        max_model_len=int(args.max_model_len),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed),
    )
    validate_model_config(config)
    return config


def run_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    config = model_config_from_args(args)
    input_dir = cast(Path, args.input_dir)
    output_dir = cast(Path, args.output_dir)
    show_progress = not bool(args.no_tqdm)
    turns, input_audit = load_source_turns(input_dir, show_progress)
    assignments = assign_length_deciles(turns, DECILE_COUNT, SAMPLE_SEED)
    sample = sample_from_deciles(
        assignments,
        DECILE_COUNT,
        SAMPLE_PER_DECILE,
        SAMPLE_SEED,
    )
    if len(sample) != SAMPLE_SIZE:
        raise DiagnosticError(f"Selected {len(sample)} turns; expected exactly {SAMPLE_SIZE}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_checkpoint = output_dir / "iceberg_saturation_sample.jsonl"
    generation_checkpoint = output_dir / "iceberg_saturation_raw_outputs.jsonl"
    sample_hash = ensure_sample_checkpoint(sample_checkpoint, sample)
    signature = run_signature(config, sample_hash)
    sample_ids = {turn.source.turn_id for turn in sample}
    generations = load_generation_checkpoint(generation_checkpoint, sample_ids, signature)
    generations = generate_missing_records(
        sample,
        generations,
        generation_checkpoint,
        config,
        signature,
        show_progress,
    )
    if set(generations) != sample_ids:
        missing = sorted(sample_ids - set(generations))
        extra = sorted(set(generations) - sample_ids)
        raise DiagnosticError(
            f"Generation completeness failure: missing={missing[:5]}, extra={extra[:5]}"
        )

    result_turns = build_turn_results(sample, generations, ITEM_CAP)
    if len(result_turns) != SAMPLE_SIZE:
        raise DiagnosticError(
            f"Parsed {len(result_turns)} complete responses; exactly {SAMPLE_SIZE} are required"
        )
    decile_summary = build_decile_summary_frame(result_turns, DECILE_COUNT, ITEM_CAP)
    write_frame(output_dir / "iceberg_saturation_turns.csv", result_turns)
    write_frame(output_dir / "iceberg_saturation_by_decile.csv", decile_summary)
    save_plot(decile_summary, output_dir)
    summary = build_summary(
        result_turns,
        decile_summary,
        input_audit,
        config,
        sample_hash,
        signature,
    )
    write_json(output_dir / "iceberg_saturation_summary.json", summary)
    logger.info("Iceberg saturation diagnostic complete", extra={"output_dir": str(output_dir)})
    return summary


def main() -> None:
    run_diagnostic(parse_args(None))


if __name__ == "__main__":
    main()
