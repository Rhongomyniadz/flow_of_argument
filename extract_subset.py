import argparse
import gzip
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("save-covid-episodes-jsonl")


def canonical_to_raw(canonical_url: str) -> str:
    p = urlparse(canonical_url)
    domain = p.netloc
    scheme = p.scheme
    path = p.path.lstrip("/")
    collapsed = path.replace("/", "").replace("-", "")
    host_noslash = f"{scheme}{domain}"
    return f"/{domain}/o3/{host_noslash}{collapsed}MERGED"


def episode_original_json(ep) -> Optional[Dict]:
    for name in ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]:
        if hasattr(ep, name):
            val = getattr(ep, name)
            if isinstance(val, dict):
                return val
    for meth in ["to_dict", "as_dict", "dict"]:
        fn = getattr(ep, meth, None)
        if callable(fn):
            try:
                val = fn()
            except Exception:
                continue
            if isinstance(val, dict):
                return val
    return None


def main():
    ap = argparse.ArgumentParser(description="Write SPoRC COVID-topic episodes to jsonl.gz (structure preserved).")
    ap.add_argument("--topic_threshold", type=float, default=0.02, help="Min weight to consider 'COVID-topic' in doc_topics.")
    ap.add_argument("--min_speakers", type=int, default=2, help="Minimum speakers per episode.")
    ap.add_argument("--max_speakers", type=int, default=2, help="Maximum speakers per episode.")
    ap.add_argument("--out_jsonl", type=str, default="results/covid_episodes.jsonl.gz", help="Output gzipped JSONL file.")
    ap.add_argument("--topic_keys", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/topic_keys.txt")
    ap.add_argument("--doc_topics", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/doc_topics.txt")
    ap.add_argument("--sporc_dir", type=str,
                    default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/")
    args = ap.parse_args()

    # 1) Find **all** COVID-related topic ids
    topic_keys = pd.read_csv(
        args.topic_keys,
        sep="\t", header=None,
        names=["topic_id", "overall_prop", "keywords"],
    )
    covid_mask = topic_keys["keywords"].str.contains(r"\bcovid\b", case=False, regex=True, na=False)
    covid_ids = topic_keys.loc[covid_mask, "topic_id"].astype(int).tolist()
    if not covid_ids:
        log.error("No COVID-related topics found in topic_keys.")
        return
    log.warning("COVID-related topics found: %s", covid_ids)
    topic_cols = [f"topic_{i}" for i in covid_ids]

    # 2) Read doc_topics **only** for URL + those COVID columns; mark URL if any col > threshold
    log.info("Reading doc_topics: %s", args.doc_topics)
    doc_topics = pd.read_csv(
        args.doc_topics,
        sep="\t",
        header=None,
        names=["row_id", "url"] + [f"topic_{i}" for i in range(100)],
        usecols=["url"] + topic_cols,
        dtype={c: "float32" for c in topic_cols},
    )
    mask = doc_topics[topic_cols].max(axis=1) > args.topic_threshold
    matched_urls = set(doc_topics.loc[mask, "url"])
    log.info("Found %d URLs above threshold %.4f across %d COVID topics",
             len(matched_urls), args.topic_threshold, len(topic_cols))

    del doc_topics, topic_keys
    gc.collect()

    if not matched_urls:
        log.warning("No URLs matched the threshold; nothing to write.")
        return

    # 3) Load SPORC and fetch exactly-2-speaker episodes
    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()
    episodes = sporc.search_episodes(min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    log.info("Scanning %d episodes (min_speakers=%d, max_speakers=%d)",
             len(episodes), args.min_speakers, args.max_speakers)

    # 4) Write matched episodes to gzipped JSONL with **original** structure
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_unmatched = 0
    n_no_raw = 0

    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        for ep in tqdm(episodes, desc="matching episodes"):
            mp3_url = getattr(ep, "mp3_url", None)
            if not mp3_url:
                n_unmatched += 1
                continue

            raw_url = canonical_to_raw(mp3_url)
            if raw_url not in matched_urls:
                n_unmatched += 1
                continue

            raw_obj = episode_original_json(ep)
            if raw_obj is None:
                n_no_raw += 1
                continue

            gz.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")
            n_written += 1

    log.info("Wrote %d episodes to %s", n_written, str(out_path))
    if n_no_raw:
        log.warning("Matched but skipped %d episodes (original JSON not accessible).", n_no_raw)
    log.info("Skipped %d episodes that did not match COVID filter.", n_unmatched)


if __name__ == "__main__":
    main()
