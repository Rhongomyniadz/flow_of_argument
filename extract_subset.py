
import argparse
import gzip
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Set
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


def episode_original_json(ep) -> Optional[Dict]:
    """Return the original SPoRC episode JSON dict from the object, unchanged."""
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


def detect_turns_file(sporc_dir: Path) -> Path:
    """Best-effort detection of the SPoRC turns file if not explicitly provided."""
    candidates = [
        sporc_dir / "speakerTurnData.jsonl.gz",
        sporc_dir / "speakerTurnData.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate turns file under {sporc_dir} "
                            f"(looked for speakerTurnData.jsonl[.gz]). Use --turns_path.")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Write SPoRC COVID-topic episodes and their turns (structure preserved).")
    ap.add_argument("--topic_threshold", type=float, default=0.02, help="Min weight for 'COVID-topic' in doc_topics.")
    ap.add_argument("--min_speakers", type=int, default=2, help="Minimum speakers per episode.")
    ap.add_argument("--max_speakers", type=int, default=2, help="Maximum speakers per episode.")
    # Default outputs now in /data
    ap.add_argument("--out_jsonl", type=str, default="/data/covid_episodes.jsonl.gz",
                    help="Output gzipped JSONL file (episodes).")
    ap.add_argument("--out_turns", type=str, default="/data/covid_episodes_turn.jsonl.gz",
                    help="Output gzipped JSONL file (turns for those episodes).")
    ap.add_argument("--topic_keys", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/topic_keys.txt",
                    help="Path to topic_keys.txt")
    ap.add_argument("--doc_topics", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/doc_topics.txt",
                    help="Path to doc_topics.txt")
    ap.add_argument("--sporc_dir", type=str,
                    default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/",
                    help="Local SPoRC data directory")
    ap.add_argument("--turns_path", type=str, default="",
                    help="Path to speakerTurnData.jsonl[.gz]. If empty, auto-detect under --sporc_dir.")
    args = ap.parse_args()

    sporc_dir = Path(args.sporc_dir)
    out_ep_path = Path(args.out_jsonl)
    out_turns_path = Path(args.out_turns)
    out_ep_path.parent.mkdir(parents=True, exist_ok=True)
    out_turns_path.parent.mkdir(parents=True, exist_ok=True)

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
    sporc = SPORCDataset(local_data_dir=str(sporc_dir), streaming=True)
    sporc.load_podcast_subset()
    episodes = sporc.search_episodes(min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    log.info("Scanning %d episodes (min_speakers=%d, max_speakers=%d)",
             len(episodes), args.min_speakers, args.max_speakers)

    # 4) Write matched episodes to /data (original structure), and collect mp3 URLs for turns
    mp3_urls_written: Set[str] = set()
    raw_urls_written: Set[str] = set()
    n_written = 0
    n_unmatched = 0
    n_no_raw = 0

    with gzip.open(out_ep_path, "wt", encoding="utf-8") as gz_out:
        for ep in tqdm(episodes, desc="writing matched episodes"):
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

            gz_out.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")
            n_written += 1
            mp3_urls_written.add(mp3_url)
            raw_urls_written.add(raw_url)

    log.info("Wrote %d episodes to %s", n_written, str(out_ep_path))
    if n_no_raw:
        log.warning("Matched but skipped %d episodes (original JSON not accessible).", n_no_raw)
    log.info("Skipped %d episodes that did not match COVID filter.", n_unmatched)

    if not mp3_urls_written:
        log.warning("No episodes were written; skipping turns export.")
        return

    # 5) Stream the turns file ONCE, and write only turns from those episodes to /data
    turns_path = Path(args.turns_path).expanduser() if args.turns_path else detect_turns_file(sporc_dir)
    if not turns_path.exists():
        log.error("Turns file not found at %s", str(turns_path))
        return

    acceptable_ids: Set[str] = set()
    for mp3 in mp3_urls_written:
        acceptable_ids.add(mp3)
        acceptable_ids.add(canonical_to_raw(mp3))
    acceptable_ids.update(raw_urls_written)

    n_turns_written = 0
    total_lines = 0

    log.info("Streaming turns from %s …", str(turns_path))
    opener = gzip.open if turns_path.suffix == ".gz" else open
    candidate_keys = ("mp3_url", "audio_url", "episode_mp3", "episode_url", "url")

    with opener(turns_path, "rt", encoding="utf-8") as fin, gzip.open(out_turns_path, "wt", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="filtering turns", unit="lines"):
            total_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            cand = ""
            for k in candidate_keys:
                v = rec.get(k)
                if isinstance(v, str) and v:
                    cand = v
                    break
            if not cand:
                continue

            if cand in acceptable_ids or canonical_to_raw(cand) in acceptable_ids:
                # write the original line (structure preserved)
                fout.write(line if line.endswith("\n") else (line + "\n"))
                n_turns_written += 1

    log.info("Wrote %d matching turns (scanned %d lines) to %s",
             n_turns_written, total_lines, str(out_turns_path))


if __name__ == "__main__":
    main()
