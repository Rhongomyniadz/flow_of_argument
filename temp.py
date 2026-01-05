import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions_csv",
        type=str,
        default="results/analysis_charts/clarification_prediction/questions.csv",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/analysis_charts/clarification_prediction/ground_truth/questions_gt_first100.csv",
    )
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    in_path = Path(args.questions_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    n = min(args.n, len(df))
    sample = df.head(n).copy()

    # Add ground-truth columns for hand labeling
    sample["is_clarification"] = ""  # fill with 0/1

    sample.to_csv(out_path, index=False)
    print(f"✅ Wrote {n} rows for hand labeling to: {out_path}")
    print("Fill is_clarification with 1 (clarification) or 0 (not).")


if __name__ == "__main__":
    main()
