import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("data/maxim_violations_labeled")
DEFAULT_OUTPUT = Path("data_processing/maxim_violation_audit_sample.csv")
LABELS = ("Relation", "Quantity", "Manner")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--per_label", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_turns(path: Path):
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("turns", [])


def compact(text: str, limit: int = 260) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = []
    for fp in sorted(args.input_root.glob("*/*.json")):
        category = fp.parent.name
        turns = load_turns(fp)
        for i, turn in enumerate(turns):
            label = turn.get("maxim_violation_label")
            if label not in LABELS:
                continue
            prev = turns[i - 1] if i > 0 else {}
            rows.append(
                {
                    "category": category,
                    "episode_file": fp.name,
                    "turn_list_index": i,
                    "turn_idx": turn.get("turn_idx"),
                    "label": label,
                    "turn_type_label": turn.get("turn_type_label"),
                    "conversation_move_label": turn.get("conversation_move_label"),
                    "prev_turn_text": compact(prev.get("turn_text", "")),
                    "current_turn_text": compact(turn.get("turn_text", "")),
                    "auditor_label": "",
                    "is_correct": "",
                    "notes": "",
                }
            )

    sampled = []
    for label in LABELS:
        bucket = [r for r in rows if r["label"] == label]
        rng.shuffle(bucket)
        sampled.extend(bucket[: args.per_label])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "episode_file",
                "turn_list_index",
                "turn_idx",
                "label",
                "turn_type_label",
                "conversation_move_label",
                "prev_turn_text",
                "current_turn_text",
                "auditor_label",
                "is_correct",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(sampled)

    print(
        f"Wrote {len(sampled)} audit rows to {args.output_csv} "
        f"({args.per_label} per label for {', '.join(LABELS)} where available)."
    )


if __name__ == "__main__":
    main()
