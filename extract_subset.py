import argparse
import gzip
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Set, List
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("save-covid-episodes-jsonl")


# ---------------- Utilities ----------------

def canonical_to_raw(canonical_url: str) -> str:
    """Map an mp3 URL to SPoRC's 'raw' URL format used in doc_topics."""
    p = urlparse(canonical_url or "")
    domain = p.netloc
    scheme = p.scheme
    path = (p.path or "").lstrip("/")
    collapsed = path.replace("/", "").replace("-", "")
    host_noslash = f"{scheme}{domain}"
    return f"/{domain}/o3/{host_noslash}{collapsed}MERGED" if domain and scheme else ""


def original_json_like(obj) -> Optional[Dict]:
    """
    Return the original JSON dict attached to a SPORC Episode/Turn object, if present.
    We do NOT fabricate a new structure if not found.
    """
    # Try common raw containers first
    for name in ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if isinstance(val, dict):
                return val
    # Then try dict-ish methods (only if they return a dict already shaped by SPORC)
    for meth in ["to_dict", "as_dict", "dict"]:
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                val = fn()
            except Exception:
                continue
            if isinstance(val, dict):
                return val
    return None


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Write SPORC COVID-topic episodes and their turns (structure preserved).")
    ap.add_argument("--topic_threshold", type=float, default=0.02, help="Min weight for 'COVID-topic' in doc_topics.")
    ap.add_argument("--min_speakers", type=int, default=2, help="Minimum speakers per episode.")
    ap.add_argument("--max_speakers", type=int, default=2, help="Maximum speakers per episode.")
    ap.add_argument("--episodes_out", type=str, default="data/covid_episodes.jsonl.gz",
                    help="Output gzipped JSONL file for episodes (original JSON).")
    ap.add_argument("--turns_out", type=str, default="data/covid_episodes_turn.jsonl.gz",
                    help="Output gzipped JSONL file for turns (original JSON).")
    ap.add_argument("--topic_keys", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/topic_keys.txt",
                    help="Path to topic_keys.txt")
    ap.add_argument("--doc_topics", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/doc_topics.txt",
                    help="Path to doc_topics.txt")
    ap.add_argument("--sporc_dir", type=str,
                    default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/",
                    help="Local SPORC data directory")
    args = ap.parse_args()

    episodes_out = Path(args.episodes_out)
    turns_out = Path(args.turns_out)
    episodes_out.parent.mkdir(parents=True, exist_ok=True)
    turns_out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Find **all** COVID-related topic ids
    topic_keys = pd.read_csv(
        args.topic_keys,
        sep="\t", header=None,
        names=["topic_id", "overall_prop", "keywords"],
    )
    mask = topic_keys["keywords"].str.contains(r"\bcovid\b", case=False, regex=True, na=False)
    ids: List[int] = topic_keys.loc[mask, "topic_id"].astype(int).tolist()
    if not ids:
        log.error("No COVID-related topics found in topic_keys.")
        return
    log.warning("COVID-related topics found: %s", ids)
    topic_cols = [f"topic_{i}" for i in ids]

    # 2) Read doc_topics (only URL + COVID columns); match URL if any col > threshold
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
    matched_urls: Set[str] = set(doc_topics.loc[mask, "url"])
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

    # NOTE: This follows your pattern (search_episodes + ep.get_all_turns()) to maintain the episode→turn linkage.
    episodes = sporc.search_episodes(min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    log.info("Scanning %d episodes (min_speakers=%d, max_speakers=%d)",
             len(episodes), args.min_speakers, args.max_speakers)

    n_ep_written = 0
    n_ep_skipped_no_raw = 0
    n_ep_not_matched = 0
    n_turns_written = 0
    n_turns_skipped_no_raw = 0

    with gzip.open(episodes_out, "wt", encoding="utf-8") as ep_out, \
         gzip.open(turns_out, "wt", encoding="utf-8") as turn_out:

        for ep in tqdm(episodes, desc="matching & writing episodes"):
            mp3_url = getattr(ep, "mp3_url", None)
            if not mp3_url:
                n_ep_not_matched += 1
                continue

            raw_url = canonical_to_raw(mp3_url)
            if raw_url not in matched_urls:
                n_ep_not_matched += 1
                continue

            # Write ORIGINAL EPISODE JSON
            ep_raw = original_json_like(ep)
            if ep_raw is None:
                n_ep_skipped_no_raw += 1
                continue

            ep_out.write(json.dumps(ep_raw, ensure_ascii=False) + "\n")
            n_ep_written += 1

            # Write ORIGINAL TURNS JSON for this episode via ep.get_all_turns()
            try:
                turns = ep.get_all_turns()
            except Exception as e:
                log.warning("Failed to load turns for episode (mp3_url=%s): %s", mp3_url, e)
                continue

            for t in turns:
                t_raw = original_json_like(t)
                if t_raw is None:
                    n_turns_skipped_no_raw += 1
                    continue
                turn_out.write(json.dumps(t_raw, ensure_ascii=False) + "\n")
                n_turns_written += 1

    log.info("Episodes written: %d  | Skipped(no original JSON): %d  | Not matched: %d",
             n_ep_written, n_ep_skipped_no_raw, n_ep_not_matched)
    log.info("Turns written: %d    | Skipped(no original JSON): %d",
             n_turns_written, n_turns_skipped_no_raw)
    log.info("Done. Episodes -> %s; Turns -> %s", str(episodes_out), str(turns_out))


if __name__ == "__main__":
    main()
