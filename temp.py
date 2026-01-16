import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=Path("data/conversation_moves_labeled"))
    ap.add_argument("--output", type=Path, default=Path("experiments/exp2_iceberg/turn_type_hist.png"))
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No .json files found under {args.input_dir}")

    counts = {"Substantive": 0, "Backchannel": 0, "Procedural": 0, "Disrupted": 0, "MISSING": 0}

    for fp in tqdm(files, desc="Episodes"):
        with fp.open("r", encoding="utf-8") as f:
            turns = json.load(f)
        if not isinstance(turns, list):
            continue

        for t in turns:
            tt = t.get("turn_type_label")
            if tt in counts:
                counts[tt] += 1
            elif tt is None:
                counts["MISSING"] += 1
            else:
                # unknown label -> treat as missing bucket
                counts["MISSING"] += 1

    labels = ["Substantive", "Backchannel", "Procedural", "Disrupted", "MISSING"]
    values = [counts[k] for k in labels]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.xlabel("Turn Type")
    plt.ylabel("Count")
    plt.title("Histogram of Turn Types (All Episodes)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    # Also print counts to terminal
    print("Counts:")
    for k in labels:
        print(f"{k}: {counts[k]}")
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
