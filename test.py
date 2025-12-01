# import gzip
# import json

# cluster_speaker_turn_path = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/speakerTurnData.jsonl.gz'
# local_speaker_turn_path = 'data/covid_episodes_turn.jsonl.gz'
# cluster_episode_level_path = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/episodeLevelData.jsonl.gz'
# local_episode_level_path = 'data/covid_episodes.jsonl.gz'

# def print_sample(file_path):
#     """
#     Opens a gzipped JSONL file, reads the first two lines,
#     parses each JSON line, and prints them in a pretty format.
#     """
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:
#         line = f.readline().strip()
#         if not line:
#             print("No more lines to read.")
        
#         try:
#             sample = json.loads(line)
#             print(f"Sample:")
#             print(json.dumps(sample, indent=4, ensure_ascii=False))
#             print(sample.get("speaker")[0])
#         except json.JSONDecodeError as e:
#             print(f"Failed to parse JSON line: {e}")
            
            
# print_sample(local_speaker_turn_path)





# political appearance episodes interviews sampling script
# import json
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# # -----------------------------
# # Config: update if needed
# # -----------------------------
# JSONL_PATHS = [
#     "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpInterviews2.jsonl",
#     "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpsDataCleaned_Interviews_withDBIds.jsonl",
# ]

# TURNS_DIR = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes_interviews")

# N_SAMPLES_PER_FILE = 3          # how many records to print per jsonl
# MAX_KEYS_TO_SHOW = 60           # truncate big key lists
# PRETTY_JSON_CHARS = 3500        # truncate pretty json output


# # -----------------------------
# # JSONL reading
# # -----------------------------
# def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except Exception as e:
#                 print(f"[WARN] {path.name}:{line_no} JSON decode error: {e}")
#                 continue
#             if isinstance(obj, dict):
#                 yield obj
#             else:
#                 # some pipelines accidentally write arrays; still show them
#                 yield {"_non_dict_record": obj}


# def shorten(s: str, max_len: int) -> str:
#     s = s or ""
#     return s if len(s) <= max_len else s[: max_len - 3] + "..."


# def pretty(obj: Any) -> str:
#     return shorten(json.dumps(obj, ensure_ascii=False, indent=2), PRETTY_JSON_CHARS)


# # -----------------------------
# # Heuristics to find episode identifiers
# # -----------------------------
# ID_CANDIDATE_KEYS = [
#     "db_id", "dbId", "dbID", "episode_db_id", "episodeDbId",
#     "episode_id", "episodeId", "ep_id", "epId",
#     "guid", "rss_guid", "rssGuid",
#     "spotifyEpisodeId", "appleEpisodeId",
#     "podcastIndexEpisodeId", "podcastindexEpisodeId",
#     "file_id", "fileId",
#     "turns_path", "turnsPath", "turns_file", "turnsFile",
# ]


# def extract_id_fields(rec: Dict[str, Any]) -> List[Tuple[str, Any]]:
#     out = []
#     for k in ID_CANDIDATE_KEYS:
#         if k in rec and rec[k] not in (None, "", [], {}):
#             out.append((k, rec[k]))
#     return out


# def find_turns_file(turns_dir: Path, rec: Dict[str, Any]) -> Optional[Path]:
#     """
#     Best-effort matching:
#     - If record already contains a turns_path-like field, try it.
#     - Else try common id-like fields and look for filenames containing that id.
#     """
#     # Direct path hints first
#     for k in ["turns_path", "turnsPath", "turns_file", "turnsFile"]:
#         p = rec.get(k)
#         if isinstance(p, str) and p:
#             candidate = Path(p)
#             if candidate.exists():
#                 return candidate
#             candidate2 = turns_dir / candidate.name
#             if candidate2.exists():
#                 return candidate2

#     # Try id fields against filenames
#     ids = [v for (k, v) in extract_id_fields(rec)]
#     ids_str = []
#     for v in ids:
#         if isinstance(v, (int, float)):
#             ids_str.append(str(int(v)))
#         elif isinstance(v, str):
#             ids_str.append(v.strip())
#     ids_str = [s for s in ids_str if s]

#     if not turns_dir.exists():
#         return None

#     # Cheap search: scan a limited prefix of directory listing
#     files = list(turns_dir.glob("*.json*"))  # .json, .jsonl, .json.gz etc.
#     for ident in ids_str:
#         for fp in files:
#             if ident in fp.name:
#                 return fp
#     return None


# # -----------------------------
# # Main: print examples
# # -----------------------------
# def inspect_one_jsonl(jsonl_path: str, turns_dir: Path, n_samples: int) -> None:
#     p = Path(jsonl_path)
#     print("\n" + "=" * 90)
#     print(f"JSONL: {p}  (exists={p.exists()})")
#     if not p.exists():
#         return

#     samples = []
#     for rec in iter_jsonl(p):
#         samples.append(rec)
#         if len(samples) >= n_samples:
#             break

#     if not samples:
#         print("No readable records found.")
#         return

#     # Show union of keys (from first ~50 records would be nicer; keep it light)
#     keys = sorted({k for r in samples for k in r.keys()})
#     print(f"Sampled {len(samples)} records. Keys (up to {MAX_KEYS_TO_SHOW}):")
#     print(shorten(", ".join(keys), 5000))

#     for i, rec in enumerate(samples, 1):
#         print(f"\n--- Sample record {i} ---")
#         id_fields = extract_id_fields(rec)
#         if id_fields:
#             print("ID-like fields:", ", ".join([f"{k}={repr(v)}" for k, v in id_fields]))
#         else:
#             print("ID-like fields: (none of the common keys found)")

#         tf = find_turns_file(turns_dir, rec)
#         print(f"Matched turns file: {tf if tf else '(no match found via heuristics)'}")

#         print("\nRecord preview:")
#         print(pretty(rec))


# def main():
#     print(f"TURNS_DIR: {TURNS_DIR} (exists={TURNS_DIR.exists()})")
#     for jp in JSONL_PATHS:
#         inspect_one_jsonl(jp, TURNS_DIR, N_SAMPLES_PER_FILE)


# if __name__ == "__main__":
#     main()


import gzip
import json
from pathlib import Path
from typing import Iterable, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("find-prev-count-one")


def iter_json_objects(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from .json, .jsonl, and .jsonl.gz files."""
    if not path.exists():
        return

    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as fh:
        # Try streaming JSONL one object per line
        fh.seek(0)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Fall back to reading whole file
                break

        # Reset and attempt to parse whole file if not JSONL
        fh.seek(0)
        try:
            obj = json.load(fh)
        except Exception:
            return

        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        elif isinstance(obj, dict):
            yield obj


def assumptions_count_from_obj(obj: Dict[str, Any]) -> int:
    """Return len(obj['assumptions']) if present, else 0."""
    a = obj.get("assumptions")
    if isinstance(a, list):
        return len(a)
    return 0


def prev_assumptions_count(obj: Dict[str, Any]) -> Optional[int]:
    """
    Return a count of assumptions for the 'prev' side if available.
    Checks multiple common keys/representations.
    """
    # direct numeric fields (binned or raw)
    for key in ("num_assumptions_prev_capped", "num_assumptions_prev", "num_prev_assumptions"):
        if key in obj and isinstance(obj[key], (int, float)):
            return int(obj[key])

    # check nested 'prev' object
    prev = obj.get("prev") or obj.get("previous") or obj.get("prev_turn") or obj.get("previous_turn")
    if isinstance(prev, dict):
        return assumptions_count_from_obj(prev)

    # check adjacent fields named like prev_assumptions
    for key in ("prev_assumptions", "previous_assumptions", "assumptions_prev"):
        if key in obj and isinstance(obj[key], list):
            return len(obj[key])

    return None


def find_matches(root: Path, target_prev_count: int = 1):
    matches = []
    files = list(root.glob("**/*"))
    for f in files:
        if f.is_dir():
            continue
        if f.suffix.lower() not in {".json", ".jsonl", ".gz"}:
            continue
        for obj in iter_json_objects(f):
            prev_count = prev_assumptions_count(obj)
            if prev_count is None:
                # If prev info isn't present, check top-level 'assumptions' as fallback
                if assumptions_count_from_obj(obj) == target_prev_count:
                    matches.append({"source": str(f), "object": obj})
            else:
                if prev_count == target_prev_count:
                    matches.append({"source": str(f), "object": obj})
            # Also check nested turns arrays
            if "turns" in obj and isinstance(obj["turns"], list):
                for turn in obj["turns"]:
                    prev_count_turn = prev_assumptions_count(turn)
                    if prev_count_turn is None:
                        if assumptions_count_from_obj(turn) == target_prev_count:
                            matches.append({"source": str(f), "object": turn})
                    elif prev_count_turn == target_prev_count:
                        matches.append({"source": str(f), "object": turn})

    return matches


def main():
    folder = Path("results/political/parsed")
    if not folder.exists():
        log.error("Path does not exist: %s", folder)
        return

    matches = find_matches(folder, target_prev_count=1)
    log.info("Found %d matches where prev-assumptions == 1", len(matches))

    out_file = Path("results/political/matches_prev_assumptions_1.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for item in matches:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    if matches:
        log.info("First match saved to output and printed below:")
        print(json.dumps(matches[0], ensure_ascii=False, indent=2))
    else:
        log.info("No matches found.")

if __name__ == "__main__":
    main()
