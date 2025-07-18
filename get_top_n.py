import os
import argparse
import json
from collections import Counter, defaultdict
from sporc import SPORCDataset

def main():
    parser = argparse.ArgumentParser(
        description="Count and save the top-N podcast hosts by episode count."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/",
        help="Path to SPoRC processed data"
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="How many top hosts to include"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="top_hosts.json",
        help="Path to write the JSON results"
    )
    args = parser.parse_args()

    # Load entire dataset (streaming mode)
    sporc = SPORCDataset(local_data_dir=args.data_dir, streaming=True)
    sporc.load_podcast_subset()  # load everything
    episodes = sporc.get_all_episodes()

    # Count episodes per predicted host
    counts = Counter()
    for ep in episodes:
        host_list = getattr(ep, "host_names", None) or getattr(ep, "hostPredictedNames", None)
        host = host_list[0] if isinstance(host_list, list) and host_list else "Unknown"
        counts[host] += 1

    # Prepare top-N list
    top_list = [
        {"host": host, "episode_count": cnt}
        for host, cnt in counts.most_common(args.top_n)
    ]

    # Write to JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(top_list, f, indent=2)

    print(f"Saved top {args.top_n} hosts → {args.output}")

if __name__ == "__main__":
    main()
