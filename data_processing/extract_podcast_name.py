import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract-episode-names")


# -------------------- Helper --------------------

def episode_to_raw(ep):
    """Return the raw dictionary representation of an episode object."""
    for name in ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]:
        if hasattr(ep, name):
            val = getattr(ep, name)
            if isinstance(val, dict):
                return val
    for meth in ["to_dict", "as_dict", "dict"]:
        fn = getattr(ep, meth, None)
        if callable(fn):
            try:
                v = fn()
            except Exception:
                continue
            if isinstance(v, dict):
                return v
    return None


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Extract episode names from SPoRC dataset")
    parser.add_argument(
        "--sporc_dir",
        type=str,
        default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1",
        help="Path to SPoRC local data directory (default: /shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="episode_names.json",
        help="Output file path (.json or .csv)"
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=2,
        help="Minimum speakers per episode (default=1)"
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=2,
        help="Maximum speakers per episode (default=None)"
    )
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load SPoRC dataset
    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()

    episodes = sporc.search_episodes(min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    log.info(
        "Found %d episodes (min_speakers=%s, max_speakers=%s)",
        len(episodes), args.min_speakers, args.max_speakers
    )

    results = []
    for ep in tqdm(episodes, desc="Extracting episode titles"):
        ep_raw = episode_to_raw(ep)
        if not ep_raw:
            continue

        title = ep_raw.get("title") or ep_raw.get("episode_title") or ""
        podcast = ep_raw.get("podcast_name") or ep_raw.get("show_name") or ep_raw.get("collection_name") or ""
        mp3_url = ep_raw.get("mp3_url") or ep_raw.get("url") or ""

        if not title.strip():
            continue

        results.append({
            "title": title.strip(),
            "podcast": podcast.strip(),
            "mp3_url": mp3_url.strip()
        })

    # Save output
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif output_path.suffix.lower() == ".csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "podcast", "mp3_url"])
            writer.writeheader()
            writer.writerows(results)
    else:
        raise ValueError("Unsupported output format (use .json or .csv)")

    log.info("Saved %d episode names to %s", len(results), str(output_path))


if __name__ == "__main__":
    main()