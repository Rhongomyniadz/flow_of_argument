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


# ---------------- Utilities ----------------

def canonical_to_raw(canonical_url: str) -> str:
    """Match SPoRC's 'raw' key used in doc_topics URL column."""
    p = urlparse(canonical_url)
    domain = p.netloc
    scheme = p.scheme
    path = p.path.lstrip("/")
    collapsed = path.replace("/", "").replace("-", "")
    host_noslash = f"{scheme}{domain}"
    return f"/{domain}/o3/{host_noslash}{collapsed}MERGED"


def episode_original_json(ep) -> Optional[Dict]:
    """
    Try to fetch the original dict from the episode object **without altering its structure**.
    We intentionally avoid synthesizing a new dict; if no original dict is discoverable,
    we return None and skip that episode.

    Order of attempts (first hit wins):
      - ep.raw / ep._raw / ep.json / ep._json / ep.record / ep._record / ep.data / ep._data / ep.source / ep._source
      - ep.to_dict() / ep.as_dict() / ep.dict()  (only if they return a dict already shaped by SPoRC)
    """
    preferred_attrs = ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]
    for name in preferred_attrs:
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

    # If none of the above worked, do NOT fabricate a structure.
    return None


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Write SPoRC COVID-topic episodes to jsonl.gz (structure preserved).")
    ap.add_argument("--topic_threshold", type=float, default=0.02, help="Min weight to consider 'COVID-topic' in doc_topics.")
    ap.add_argument("--min_speakers", type=int, default=2, help="Minimum speakers per episode.")
    ap.add_argument("--max_speakers", type=int, default=2, help="Maximum speakers per episode.")
    ap.add_argument("--out_jsonl", type=str, default="data/covid_episodes.jsonl.gz", help="Output gzipped JSONL file.")
    ap.add_argument("--topic_keys", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/topic_keys.txt",
                    help="Path to topic_keys.txt")
    ap.add_argument("--doc_topics", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/doc_topics.txt",
                    help="Path to doc_topics.txt")
    ap.add_argument("--sporc_dir", type=str,
                    default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/",
                    help="Local SPoRC data directory")
    args = ap.parse_args()

    # 1) Identify the COVID topic id
    topic_keys = pd.read_csv(
        args.topic_keys,
        sep="\t",
        header=None,
        names=["topic_id", "overall_prop", "keywords"],
    )
    covid_rows = topic_keys[topic_keys.keywords.str.contains(r"\bcovid\b", case=False, regex=True)]
    if covid_rows.empty:
        log.error("No COVID-related topic found in topic_keys.")
        return
    if len(covid_rows) > 1:
        log.warning("Multiple COVID-related topics found; using the first: %s",
                    covid_rows.topic_id.tolist())
    covid_id = int(covid_rows.iloc[0]["topic_id"])
    topic_col = f"topic_{covid_id}"
    log.info("Using COVID topic_id=%d -> column '%s'", covid_id, topic_col)

    # 2) Read doc_topics (only URL + COVID column) and filter
    log.info("Reading doc_topics: %s", args.doc_topics)
    doc_topics = pd.read_csv(
        args.doc_topics,
        sep="\t",
        header=None,
        names=["row_id", "url"] + [f"topic_{i}" for i in range(100)],
        usecols=["url", topic_col],
        dtype={topic_col: "float32"},
    )
    mask = doc_topics[topic_col] > args.topic_threshold
    matched_urls = set(doc_topics.loc[mask, "url"])
    log.info("Found %d URLs above threshold %.4f", len(matched_urls), args.topic_threshold)

    del doc_topics, topic_keys
    gc.collect()

    if not matched_urls:
        log.warning("No URLs matched the threshold; nothing to write.")
        return

    # 3) Load SPORC and scan episodes
    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()
    episodes = sporc.search_episodes(min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    log.info("Scanning %d episodes (min_speakers=%d, max_speakers=%d)",
             len(episodes), args.min_speakers, args.max_speakers)

    # 4) Write gzipped JSONL with the **original** SPoRC episode blobs
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped_no_raw = 0
    n_unmatched = 0

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
                # We strictly avoid creating a new structure
                n_skipped_no_raw += 1
                continue

            gz.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")
            n_written += 1

    log.info("Wrote %d episodes to %s", n_written, str(out_path))
    if n_skipped_no_raw:
        log.warning("Skipped %d matched episodes because original JSON blob wasn't accessible on the object.", n_skipped_no_raw)
    log.info("Skipped %d non-matching episodes.", n_unmatched)


if __name__ == "__main__":
    main()
