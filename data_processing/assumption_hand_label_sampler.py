import argparse
import bisect
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence, Set, Tuple, TypedDict


Category = Literal["political", "business", "religion", "commentary", "sports", "news"]
StatementKind = Literal["explicit", "implicit"]
SampleRelation = Literal[
    "same_turn",
    "same_episode_different_turn",
    "same_category_different_episode",
    "random_exclusive",
]

CATEGORIES: Tuple[Category, ...] = ("political", "business", "religion", "commentary", "sports", "news")
RELATION_WEIGHTS: Tuple[Tuple[SampleRelation, int], ...] = (
    ("same_turn", 3),
    ("same_episode_different_turn", 1),
    ("same_category_different_episode", 1),
    ("random_exclusive", 1),
)
STATEMENT_KINDS: Tuple[StatementKind, ...] = ("explicit", "implicit")
DEFAULT_INPUT_ROOT = Path("data")
DEFAULT_OUTPUT_CSV = Path("data_processing/assumption_hand_label_sample.csv")
DEFAULT_SEED = 42


class TurnRecord(TypedDict):
    turn_id: int
    category: Category
    episode_id: str
    episode_file: str
    episode_path: str
    episode_key: str
    turn_idx: str
    speaker_id: str
    turn_text: str
    explicit_count: int
    implicit_count: int


class WeightedPool(TypedDict):
    turn_ids: List[int]
    cumulative_weights: List[int]
    total_weight: int


class DataIndex(TypedDict):
    turns: List[TurnRecord]
    anchors_by_category: Dict[Category, List[int]]
    turns_by_episode: Dict[Tuple[Category, str], List[int]]
    source_by_episode_kind: Dict[Tuple[Category, str, StatementKind], WeightedPool]
    source_by_category_kind: Dict[Tuple[Category, StatementKind], WeightedPool]
    source_by_kind: Dict[StatementKind, WeightedPool]


class SampleSlot(TypedDict):
    category: Category
    relation: SampleRelation
    statement_kind: StatementKind


class StatementChoice(TypedDict):
    source_turn_id: int
    statement_index: int


class CsvRow(TypedDict):
    row_id: str
    category: str
    episode_id: str
    episode_file: str
    turn_idx: str
    speaker_id: str
    turn_text: str
    sample_relation: str
    statement_kind: str
    statement_text: str
    statement_confidence: str
    source_category: str
    source_episode_id: str
    source_episode_file: str
    source_turn_idx: str
    source_speaker_id: str
    source_statement_index: str
    source_record_id: str
    human_label: str
    notes: str


CSV_FIELDNAMES: Tuple[str, ...] = (
    "row_id",
    "category",
    "episode_id",
    "episode_file",
    "turn_idx",
    "speaker_id",
    "turn_text",
    "sample_relation",
    "statement_kind",
    "statement_text",
    "statement_confidence",
    "source_category",
    "source_episode_id",
    "source_episode_file",
    "source_turn_idx",
    "source_speaker_id",
    "source_statement_index",
    "source_record_id",
    "human_label",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample assumption extraction statements for hand labeling."
    )
    parser.add_argument("--total_entries", type=int, required=True)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_turn_rows(path: Path) -> List[Dict[str, Any]]:
    payload: Any = read_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("turns"), list):
        return [row for row in payload["turns"] if isinstance(row, dict)]
    raise ValueError(f"Expected a list of turns or a dict with turns in {path}")


def valid_statement_count(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    count: int = 0
    for item in items:
        if isinstance(item, dict) and normalize_text(item.get("text")):
            count += 1
    return count


def statement_field_name(statement_kind: StatementKind) -> str:
    if statement_kind == "explicit":
        return "explicit_propositions"
    if statement_kind == "implicit":
        return "assumptions"
    raise ValueError(f"Unsupported statement kind: {statement_kind}")


def statement_count(turn: TurnRecord, statement_kind: StatementKind) -> int:
    if statement_kind == "explicit":
        return turn["explicit_count"]
    if statement_kind == "implicit":
        return turn["implicit_count"]
    raise ValueError(f"Unsupported statement kind: {statement_kind}")


def episode_key(category: Category, episode_file: str) -> str:
    return f"{category}:{episode_file}"


def make_turn_record(
    turn_id: int,
    category: Category,
    path: Path,
    row_index: int,
    row: Dict[str, Any],
) -> TurnRecord:
    episode_file: str = path.name
    episode_identifier: str = normalize_text(row.get("episode_id")) or path.stem
    raw_turn_idx: Any = row.get("turn_idx")
    turn_idx: str = normalize_text(raw_turn_idx) or str(row_index)
    speaker_id: str = normalize_text(row.get("speaker_id"))
    turn_text: str = normalize_text(row.get("turn_text"))
    return {
        "turn_id": turn_id,
        "category": category,
        "episode_id": episode_identifier,
        "episode_file": episode_file,
        "episode_path": str(path),
        "episode_key": episode_key(category, episode_file),
        "turn_idx": turn_idx,
        "speaker_id": speaker_id,
        "turn_text": turn_text,
        "explicit_count": valid_statement_count(row.get("explicit_propositions")),
        "implicit_count": valid_statement_count(row.get("assumptions")),
    }


def build_weighted_pool(turns: Sequence[TurnRecord], turn_ids: Iterable[int], statement_kind: StatementKind) -> WeightedPool:
    pool_turn_ids: List[int] = []
    cumulative_weights: List[int] = []
    total_weight: int = 0
    for turn_id in turn_ids:
        weight: int = statement_count(turns[turn_id], statement_kind)
        if weight <= 0:
            continue
        total_weight += weight
        pool_turn_ids.append(turn_id)
        cumulative_weights.append(total_weight)
    return {
        "turn_ids": pool_turn_ids,
        "cumulative_weights": cumulative_weights,
        "total_weight": total_weight,
    }


def add_turn_to_indexes(
    anchors_by_category: Dict[Category, List[int]],
    turns_by_episode: Dict[Tuple[Category, str], List[int]],
    turn: TurnRecord,
) -> Tuple[Dict[Category, List[int]], Dict[Tuple[Category, str], List[int]]]:
    anchors_by_category[turn["category"]].append(turn["turn_id"])
    episode_tuple: Tuple[Category, str] = (turn["category"], turn["episode_key"])
    if episode_tuple not in turns_by_episode:
        turns_by_episode[episode_tuple] = []
    turns_by_episode[episode_tuple].append(turn["turn_id"])
    return anchors_by_category, turns_by_episode


def build_data_index(input_root: Path) -> DataIndex:
    turns: List[TurnRecord] = []
    anchors_by_category: Dict[Category, List[int]] = {category: [] for category in CATEGORIES}
    turns_by_episode: Dict[Tuple[Category, str], List[int]] = {}

    for category in CATEGORIES:
        parsed_dir: Path = input_root / category / "parsed"
        if not parsed_dir.is_dir():
            raise FileNotFoundError(f"Missing parsed assumption directory: {parsed_dir}")

        for path in sorted(parsed_dir.glob("*.json")):
            rows: List[Dict[str, Any]] = load_turn_rows(path)
            for row_index, row in enumerate(rows):
                turn_id: int = len(turns)
                turn: TurnRecord = make_turn_record(turn_id, category, path, row_index, row)
                if not turn["turn_text"]:
                    continue
                turns.append(turn)
                anchors_by_category, turns_by_episode = add_turn_to_indexes(
                    anchors_by_category,
                    turns_by_episode,
                    turn,
                )

    source_by_episode_kind: Dict[Tuple[Category, str, StatementKind], WeightedPool] = {}
    source_by_category_kind: Dict[Tuple[Category, StatementKind], WeightedPool] = {}
    source_by_kind: Dict[StatementKind, WeightedPool] = {}

    for category in CATEGORIES:
        for statement_kind in STATEMENT_KINDS:
            source_by_category_kind[(category, statement_kind)] = build_weighted_pool(
                turns,
                anchors_by_category[category],
                statement_kind,
            )

    for episode_tuple, turn_ids in turns_by_episode.items():
        category, key = episode_tuple
        for statement_kind in STATEMENT_KINDS:
            source_by_episode_kind[(category, key, statement_kind)] = build_weighted_pool(
                turns,
                turn_ids,
                statement_kind,
            )

    all_turn_ids: List[int] = list(range(len(turns)))
    for statement_kind in STATEMENT_KINDS:
        source_by_kind[statement_kind] = build_weighted_pool(turns, all_turn_ids, statement_kind)

    return {
        "turns": turns,
        "anchors_by_category": anchors_by_category,
        "turns_by_episode": turns_by_episode,
        "source_by_episode_kind": source_by_episode_kind,
        "source_by_category_kind": source_by_category_kind,
        "source_by_kind": source_by_kind,
    }


def largest_remainder_counts(total: int, weighted_names: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    if total < 0:
        raise ValueError(f"Total must be non-negative: {total}")
    weight_total: int = sum(weight for _, weight in weighted_names)
    if weight_total <= 0:
        raise ValueError(f"Weights must sum to a positive value: {weighted_names}")

    counts: Dict[str, int] = {}
    remainders: List[Tuple[int, int, str]] = []
    assigned: int = 0
    for order, weighted_name in enumerate(weighted_names):
        name, weight = weighted_name
        numerator: int = total * weight
        base: int = numerator // weight_total
        remainder: int = numerator % weight_total
        counts[name] = base
        assigned += base
        remainders.append((remainder, -order, name))

    remaining: int = total - assigned
    for _, _, name in sorted(remainders, reverse=True)[:remaining]:
        counts[name] += 1
    return counts


def allocate_category_counts(total_entries: int) -> Dict[Category, int]:
    weighted_categories: List[Tuple[str, int]] = [(category, 1) for category in CATEGORIES]
    raw_counts: Dict[str, int] = largest_remainder_counts(total_entries, weighted_categories)
    return {category: raw_counts[category] for category in CATEGORIES}


def relation_counts_for_category(category_count: int) -> Dict[SampleRelation, int]:
    raw_counts: Dict[str, int] = largest_remainder_counts(category_count, list(RELATION_WEIGHTS))
    return {relation: raw_counts[relation] for relation, _ in RELATION_WEIGHTS}


def build_sample_slots(total_entries: int, rng: random.Random) -> List[SampleSlot]:
    if total_entries <= 0:
        raise ValueError(f"total_entries must be positive: {total_entries}")

    category_counts: Dict[Category, int] = allocate_category_counts(total_entries)
    slots: List[SampleSlot] = []
    for category in CATEGORIES:
        relation_counts: Dict[SampleRelation, int] = relation_counts_for_category(category_counts[category])
        for relation, _ in RELATION_WEIGHTS:
            for _ in range(relation_counts[relation]):
                slots.append(
                    {
                        "category": category,
                        "relation": relation,
                        "statement_kind": "explicit",
                    }
                )

    rng.shuffle(slots)
    for index, slot in enumerate(slots):
        slot["statement_kind"] = STATEMENT_KINDS[index % len(STATEMENT_KINDS)]
    rng.shuffle(slots)
    return slots


def choose_from_weighted_pool(pool: WeightedPool, rng: random.Random) -> StatementChoice:
    if pool["total_weight"] <= 0:
        raise ValueError("Cannot choose from an empty statement pool")
    offset: int = rng.randrange(pool["total_weight"])
    pool_index: int = bisect.bisect_right(pool["cumulative_weights"], offset)
    previous_total: int = pool["cumulative_weights"][pool_index - 1] if pool_index > 0 else 0
    return {
        "source_turn_id": pool["turn_ids"][pool_index],
        "statement_index": offset - previous_total,
    }


def turn_allowed_for_choice(
    turn: TurnRecord,
    excluded_turn_ids: Set[int],
    excluded_episode_keys: Set[str],
    excluded_categories: Set[Category],
) -> bool:
    if turn["turn_id"] in excluded_turn_ids:
        return False
    if turn["episode_key"] in excluded_episode_keys:
        return False
    if turn["category"] in excluded_categories:
        return False
    return True


def build_filtered_pool(
    turns: Sequence[TurnRecord],
    pool: WeightedPool,
    statement_kind: StatementKind,
    excluded_turn_ids: Set[int],
    excluded_episode_keys: Set[str],
    excluded_categories: Set[Category],
) -> WeightedPool:
    allowed_turn_ids: List[int] = [
        turn_id
        for turn_id in pool["turn_ids"]
        if turn_allowed_for_choice(
            turns[turn_id],
            excluded_turn_ids,
            excluded_episode_keys,
            excluded_categories,
        )
    ]
    return build_weighted_pool(turns, allowed_turn_ids, statement_kind)


def choose_from_pool_with_exclusions(
    turns: Sequence[TurnRecord],
    pool: WeightedPool,
    statement_kind: StatementKind,
    excluded_turn_ids: Set[int],
    excluded_episode_keys: Set[str],
    excluded_categories: Set[Category],
    rng: random.Random,
) -> StatementChoice:
    if pool["total_weight"] <= 0:
        raise ValueError("Cannot choose from an empty statement pool")

    for _ in range(100):
        choice: StatementChoice = choose_from_weighted_pool(pool, rng)
        turn: TurnRecord = turns[choice["source_turn_id"]]
        if turn_allowed_for_choice(turn, excluded_turn_ids, excluded_episode_keys, excluded_categories):
            return choice

    filtered_pool: WeightedPool = build_filtered_pool(
        turns,
        pool,
        statement_kind,
        excluded_turn_ids,
        excluded_episode_keys,
        excluded_categories,
    )
    if filtered_pool["total_weight"] <= 0:
        raise ValueError(
            "No eligible source statements after exclusions: "
            f"kind={statement_kind}, excluded_turn_ids={sorted(excluded_turn_ids)}, "
            f"excluded_episode_keys={sorted(excluded_episode_keys)}, "
            f"excluded_categories={sorted(excluded_categories)}"
        )
    return choose_from_weighted_pool(filtered_pool, rng)


def anchor_supports_relation(
    data_index: DataIndex,
    anchor_turn: TurnRecord,
    relation: SampleRelation,
    statement_kind: StatementKind,
) -> bool:
    if relation == "same_turn":
        return statement_count(anchor_turn, statement_kind) > 0
    if relation == "same_episode_different_turn":
        episode_pool: WeightedPool = data_index["source_by_episode_kind"][
            (anchor_turn["category"], anchor_turn["episode_key"], statement_kind)
        ]
        return episode_pool["total_weight"] - statement_count(anchor_turn, statement_kind) > 0
    if relation == "same_category_different_episode":
        category_pool: WeightedPool = data_index["source_by_category_kind"][
            (anchor_turn["category"], statement_kind)
        ]
        episode_pool = data_index["source_by_episode_kind"][
            (anchor_turn["category"], anchor_turn["episode_key"], statement_kind)
        ]
        return category_pool["total_weight"] - episode_pool["total_weight"] > 0
    if relation == "random_exclusive":
        kind_pool: WeightedPool = data_index["source_by_kind"][statement_kind]
        category_pool = data_index["source_by_category_kind"][(anchor_turn["category"], statement_kind)]
        return kind_pool["total_weight"] - category_pool["total_weight"] > 0
    raise ValueError(f"Unsupported relation: {relation}")


def choose_anchor_turn_id(
    data_index: DataIndex,
    category: Category,
    relation: SampleRelation,
    statement_kind: StatementKind,
    used_anchor_ids: Set[int],
    rng: random.Random,
) -> int:
    anchor_ids: List[int] = data_index["anchors_by_category"][category]
    candidates: List[int] = [
        turn_id
        for turn_id in anchor_ids
        if turn_id not in used_anchor_ids
        and anchor_supports_relation(
            data_index,
            data_index["turns"][turn_id],
            relation,
            statement_kind,
        )
    ]
    if not candidates:
        raise ValueError(
            "No eligible unused anchors for sample slot: "
            f"category={category}, relation={relation}, statement_kind={statement_kind}, "
            f"used_anchor_count={len(used_anchor_ids)}"
        )
    return rng.choice(candidates)


def choose_statement_for_relation(
    data_index: DataIndex,
    anchor_turn: TurnRecord,
    relation: SampleRelation,
    statement_kind: StatementKind,
    rng: random.Random,
) -> StatementChoice:
    turns: List[TurnRecord] = data_index["turns"]
    if relation == "same_turn":
        count: int = statement_count(anchor_turn, statement_kind)
        if count <= 0:
            raise ValueError(
                "Anchor turn has no source statements for same_turn slot: "
                f"turn_id={anchor_turn['turn_id']}, statement_kind={statement_kind}"
            )
        return {"source_turn_id": anchor_turn["turn_id"], "statement_index": rng.randrange(count)}

    if relation == "same_episode_different_turn":
        episode_pool: WeightedPool = data_index["source_by_episode_kind"][
            (anchor_turn["category"], anchor_turn["episode_key"], statement_kind)
        ]
        return choose_from_pool_with_exclusions(
            turns,
            episode_pool,
            statement_kind,
            {anchor_turn["turn_id"]},
            set(),
            set(),
            rng,
        )

    if relation == "same_category_different_episode":
        category_pool: WeightedPool = data_index["source_by_category_kind"][
            (anchor_turn["category"], statement_kind)
        ]
        return choose_from_pool_with_exclusions(
            turns,
            category_pool,
            statement_kind,
            set(),
            {anchor_turn["episode_key"]},
            set(),
            rng,
        )

    if relation == "random_exclusive":
        kind_pool: WeightedPool = data_index["source_by_kind"][statement_kind]
        return choose_from_pool_with_exclusions(
            turns,
            kind_pool,
            statement_kind,
            set(),
            {anchor_turn["episode_key"]},
            {anchor_turn["category"]},
            rng,
        )

    raise ValueError(f"Unsupported relation: {relation}")


def selected_statement_from_row(
    row: Dict[str, Any],
    statement_kind: StatementKind,
    statement_index: int,
    path: Path,
) -> Tuple[str, str]:
    field_name: str = statement_field_name(statement_kind)
    raw_items: Any = row.get(field_name)
    if not isinstance(raw_items, list):
        raise ValueError(f"Expected list field {field_name} in {path}")

    valid_items: List[Dict[str, Any]] = [
        item for item in raw_items if isinstance(item, dict) and normalize_text(item.get("text"))
    ]
    if statement_index < 0 or statement_index >= len(valid_items):
        raise ValueError(
            "Source statement index out of range: "
            f"path={path}, field={field_name}, index={statement_index}, count={len(valid_items)}"
        )

    item: Dict[str, Any] = valid_items[statement_index]
    return normalize_text(item.get("text")), normalize_text(item.get("confidence"))


def get_turn_row_by_idx(path: Path, source_turn: TurnRecord) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = load_turn_rows(path)
    matching_rows: List[Dict[str, Any]] = [
        row
        for row_index, row in enumerate(rows)
        if (normalize_text(row.get("turn_idx")) or str(row_index)) == source_turn["turn_idx"]
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            "Could not resolve unique source turn row: "
            f"path={path}, turn_idx={source_turn['turn_idx']}, matches={len(matching_rows)}"
        )
    return matching_rows[0]


def make_source_record_id(source_turn: TurnRecord, statement_kind: StatementKind, statement_index: int) -> str:
    return "|".join(
        [
            source_turn["category"],
            source_turn["episode_file"],
            source_turn["turn_idx"],
            statement_kind,
            str(statement_index),
        ]
    )


def make_csv_row(
    row_number: int,
    anchor_turn: TurnRecord,
    source_turn: TurnRecord,
    slot: SampleSlot,
    statement_choice: StatementChoice,
    statement_text: str,
    statement_confidence: str,
) -> CsvRow:
    return {
        "row_id": f"sample_{row_number:06d}",
        "category": anchor_turn["category"],
        "episode_id": anchor_turn["episode_id"],
        "episode_file": anchor_turn["episode_file"],
        "turn_idx": anchor_turn["turn_idx"],
        "speaker_id": anchor_turn["speaker_id"],
        "turn_text": anchor_turn["turn_text"],
        "sample_relation": slot["relation"],
        "statement_kind": slot["statement_kind"],
        "statement_text": statement_text,
        "statement_confidence": statement_confidence,
        "source_category": source_turn["category"],
        "source_episode_id": source_turn["episode_id"],
        "source_episode_file": source_turn["episode_file"],
        "source_turn_idx": source_turn["turn_idx"],
        "source_speaker_id": source_turn["speaker_id"],
        "source_statement_index": str(statement_choice["statement_index"]),
        "source_record_id": make_source_record_id(
            source_turn,
            slot["statement_kind"],
            statement_choice["statement_index"],
        ),
        "human_label": "",
        "notes": "",
    }


def build_sample_rows(data_index: DataIndex, slots: Sequence[SampleSlot], rng: random.Random) -> List[CsvRow]:
    rows: List[CsvRow] = []
    used_anchor_ids: Set[int] = set()
    turns: List[TurnRecord] = data_index["turns"]

    if len(slots) > len(turns):
        raise ValueError(
            f"Requested {len(slots)} entries but only {len(turns)} eligible anchor turns are available"
        )

    for row_index, slot in enumerate(slots):
        anchor_turn_id: int = choose_anchor_turn_id(
            data_index,
            slot["category"],
            slot["relation"],
            slot["statement_kind"],
            used_anchor_ids,
            rng,
        )
        used_anchor_ids.add(anchor_turn_id)
        anchor_turn: TurnRecord = turns[anchor_turn_id]
        statement_choice: StatementChoice = choose_statement_for_relation(
            data_index,
            anchor_turn,
            slot["relation"],
            slot["statement_kind"],
            rng,
        )
        source_turn: TurnRecord = turns[statement_choice["source_turn_id"]]
        source_path: Path = Path(source_turn["episode_path"])
        source_row: Dict[str, Any] = get_turn_row_by_idx(source_path, source_turn)
        statement_text, statement_confidence = selected_statement_from_row(
            source_row,
            slot["statement_kind"],
            statement_choice["statement_index"],
            source_path,
        )
        rows.append(
            make_csv_row(
                row_index,
                anchor_turn,
                source_turn,
                slot,
                statement_choice,
                statement_text,
                statement_confidence,
            )
        )

    return rows


def write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer: csv.DictWriter[str] = csv.DictWriter(csv_file, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[CsvRow], output_csv: Path) -> None:
    relation_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    kind_counts: Dict[str, int] = {}
    for row in rows:
        relation_counts[row["sample_relation"]] = relation_counts.get(row["sample_relation"], 0) + 1
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
        kind_counts[row["statement_kind"]] = kind_counts.get(row["statement_kind"], 0) + 1

    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Category counts: {json.dumps(category_counts, sort_keys=True)}")
    print(f"Relation counts: {json.dumps(relation_counts, sort_keys=True)}")
    print(f"Statement kind counts: {json.dumps(kind_counts, sort_keys=True)}")


def main() -> None:
    args: argparse.Namespace = parse_args()
    rng: random.Random = random.Random(args.seed)
    data_index: DataIndex = build_data_index(args.input_root)
    slots: List[SampleSlot] = build_sample_slots(args.total_entries, rng)
    rows: List[CsvRow] = build_sample_rows(data_index, slots, rng)
    write_csv(args.output_csv, rows)
    print_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
