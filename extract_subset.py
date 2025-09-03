import argparse
import gzip
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Set, Iterable
from urllib.parse import urlparse, urlunparse

import re
import pandas as pd
from tqdm import tqdm
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("save-covid-episodes-jsonl")


# ---------------- Utilities ----------------

def canonical_to_raw(canonical_url: str) -> str:
    p = urlparse(canonical_url or "")
    if not p.scheme or not p.netloc:
        return ""
    domain = p.netloc
    scheme = p.scheme
    path = (p.path or "").lstrip("/")
    collapsed = path.replace("/", "").replace("-", "")
    host_noslash = f"{scheme}{domain}"
    return f"/{domain}/o3/{host_noslash}{collapsed}MERGED"


def strip_query(u: str) -> str:
    try:
        p = urlparse(u)
        if not p.scheme or not p.netloc:
            return u
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return u


def toggle_scheme(u: str) -> str:
    try:
        p = urlparse(u)
        if not p.scheme or not p.netloc:
            return u
        new_scheme = "http" if p.scheme == "https" else "https"
        return urlunparse((new_scheme, p.netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return u


def normalize_variants(u: str) -> Set[str]:
    """Produce many comparable variants for robust matching."""
    if not u:
        return set()
    out = set()
    out.add(u)

    # strip query/fragment
    uq = strip_query(u)
    out.add(uq)

    # scheme variants
    out.add(toggle_scheme(u))
    out.add(toggle_scheme(uq))

    # canonical raw version (if looks like http(s) url)
    if uq.startswith("http"):
        out.add(canonical_to_raw(uq))

    # if looks like a raw path already, keep it
    if u.startswith("/"):
        out.add(u)

    # remove trailing slash variants
    if uq.endswith("/"):
        out.add(uq[:-1])

    return {x for x in out if x}


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


def detect_turns_file(sporc_dir: Path) -> Path:
    for c in [sporc_dir / "speakerTurnData.jsonl.gz", sporc_dir / "speakerTurnData.jsonl"]:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate turns file under {sporc_dir} "
                            f"(speakerTurnData.jsonl[.gz]). Use --turns_path.")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Write SPoRC COVID-topic episodes and their turns (structure preserved).")
    ap.add_argument("--topic_threshold", type=float, default=0.02)
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=2)
    ap.add_argument("--out_jsonl", type=str, default="data/covid_episodes.jsonl.gz",
                    help="Episode output (gzipped JSONL).")
    ap.add_argument("--out_turns", type=str, default="data/covid_episodes_turn.jsonl.gz",
                    help="Turn output (gzipped JSONL).")
    ap.add_argument("--topic_keys", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/topic_keys.txt")
    ap.add_argument("--doc_topics", type=str,
                    default="/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/doc_topics.txt")
    ap.add_argument("--sporc_dir", type=str,
                    default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/")
    ap.add_argument("--turns_path", type=str, default="",
                    help="Path to speakerTurnData.jsonl[.gz]. If empty, auto-detect under --sporc_dir.")
    args = ap.parse_args()

    out_ep_path = Path(args.out_jsonl); out_ep_path.parent.mkdir(parents=True, exist_ok=True)
    out_turns_path = Path(args.out_turns); out_turns_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) ALL covid topic ids
    topic_keys = pd.read_csv(args.topic_keys, sep="\t", header=None,
                             names=["topic_id", "overall_prop", "keywords"])
    covid_mask = topic_keys["keywords"].str.contains(r"\bcovid\b", case=False, regex=True, na=False)
    covid_ids = topic_keys.loc[covid_mask, "topic_id"].astype(int).tolist()
    if not covid_ids:
        log.error("No COVID-related topics found in topic_keys.")
        return
    log.warning("COVID-related topics found: %s", covid_ids)
    topic_cols = [f"topic_{i}" for i in covid_ids]

    # 2) doc_topics filter
    log.info("Reading doc_topics: %s", args.doc_topics)
    doc_topics = pd.read_csv(
        args.doc_topics, sep="\t",
        header=None, names=["row_id", "url"] + [f"topic_{i}" for i in range(100)],
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

    # 3) Load SPORC + select episodes (exactly 2 speakers)
    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()
    episodes = sporc.search_episodes(min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    log.info("Scanning %d episodes (min_speakers=%d, max_speakers=%d)",
             len(episodes), args.min_speakers, args.max_speakers)

    # 4) Write matching episodes and collect acceptable ids for turn matching
    mp3_urls_written: Set[str] = set()
    raw_urls_written: Set[str] = set()
    acceptable: Set[str] = set()
    n_written = n_unmatched = n_no_raw = 0

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

            # build variants for matching later
            for v in normalize_variants(mp3_url):
                acceptable.add(v)
            acceptable.add(raw_url)  # already raw
            for v in normalize_variants(strip_query(mp3_url)):
                acceptable.add(v)

    log.info("Wrote %d episodes to %s", n_written, str(out_ep_path))
    if n_no_raw:
        log.warning("Matched but skipped %d episodes (original JSON not accessible).", n_no_raw)
    log.info("Skipped %d episodes that did not match COVID filter.", n_unmatched)

    if not mp3_urls_written:
        log.warning("No episodes were written; skipping turns export.")
        return

    # 5) Stream turns once; match robustly
    turns_path = Path(args.turns_path).expanduser() if args.turns_path else detect_turns_file(Path(args.sporc_dir))
    if not turns_path.exists():
        log.error("Turns file not found at %s", str(turns_path))
        return

    candidate_keys = (
        "mp3_url", "audio_url", "audio", "episode_mp3", "episode_url",
        "episode_mp3_url", "episode_audio_url", "url",
        "raw_url", "raw", "canonical_url", "source_url",
    )

    opener = gzip.open if turns_path.suffix == ".gz" else open
    n_turns_written = 0
    total_lines = 0
    key_hits = {k: 0 for k in candidate_keys}

    log.info("Streaming turns from %s …", str(turns_path))
    with opener(turns_path, "rt", encoding="utf-8") as fin, gzip.open(out_turns_path, "wt", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="filtering turns", unit="lines"):
            total_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # find a candidate value
            cand_val = ""
            for k in candidate_keys:
                v = rec.get(k)
                if isinstance(v, str) and v:
                    cand_val = v
                    key_hits[k] += 1
                    break
            if not cand_val:
                continue

            cand_variants = normalize_variants(cand_val)
            # also consider raw form of cand (if http)
            if cand_val.startswith("http"):
                cand_variants.add(canonical_to_raw(strip_query(cand_val)))

            if acceptable.intersection(cand_variants):
                # write original line unchanged
                fout.write(line if line.endswith("\n") else (line + "\n"))
                n_turns_written += 1

    log.info("Turn scan complete. Scanned: %d lines, wrote: %d", total_lines, n_turns_written)
    # Show which keys were useful
    useful = {k: v for k, v in key_hits.items() if v}
    if useful:
        log.info("Observed candidate URL keys in turns (counts): %s", useful)
    else:
        log.warning("Did not observe any known URL keys in the first pass. "
                    "You may need to inspect the turns file to see which key links to episodes.")

if __name__ == "__main__":
    main()
