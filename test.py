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



import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Config: update if needed
# -----------------------------
JSONL_PATHS = [
    "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpInterviews2.jsonl",
    "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpsDataCleaned_Interviews_withDBIds.jsonl",
]

TURNS_DIR = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes_interviews")

N_SAMPLES_PER_FILE = 3          # how many records to print per jsonl
MAX_KEYS_TO_SHOW = 60           # truncate big key lists
PRETTY_JSON_CHARS = 3500        # truncate pretty json output


# -----------------------------
# JSONL reading
# -----------------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] {path.name}:{line_no} JSON decode error: {e}")
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                # some pipelines accidentally write arrays; still show them
                yield {"_non_dict_record": obj}


def shorten(s: str, max_len: int) -> str:
    s = s or ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def pretty(obj: Any) -> str:
    return shorten(json.dumps(obj, ensure_ascii=False, indent=2), PRETTY_JSON_CHARS)


# -----------------------------
# Heuristics to find episode identifiers
# -----------------------------
ID_CANDIDATE_KEYS = [
    "db_id", "dbId", "dbID", "episode_db_id", "episodeDbId",
    "episode_id", "episodeId", "ep_id", "epId",
    "guid", "rss_guid", "rssGuid",
    "spotifyEpisodeId", "appleEpisodeId",
    "podcastIndexEpisodeId", "podcastindexEpisodeId",
    "file_id", "fileId",
    "turns_path", "turnsPath", "turns_file", "turnsFile",
]


def extract_id_fields(rec: Dict[str, Any]) -> List[Tuple[str, Any]]:
    out = []
    for k in ID_CANDIDATE_KEYS:
        if k in rec and rec[k] not in (None, "", [], {}):
            out.append((k, rec[k]))
    return out


def find_turns_file(turns_dir: Path, rec: Dict[str, Any]) -> Optional[Path]:
    """
    Best-effort matching:
    - If record already contains a turns_path-like field, try it.
    - Else try common id-like fields and look for filenames containing that id.
    """
    # Direct path hints first
    for k in ["turns_path", "turnsPath", "turns_file", "turnsFile"]:
        p = rec.get(k)
        if isinstance(p, str) and p:
            candidate = Path(p)
            if candidate.exists():
                return candidate
            candidate2 = turns_dir / candidate.name
            if candidate2.exists():
                return candidate2

    # Try id fields against filenames
    ids = [v for (k, v) in extract_id_fields(rec)]
    ids_str = []
    for v in ids:
        if isinstance(v, (int, float)):
            ids_str.append(str(int(v)))
        elif isinstance(v, str):
            ids_str.append(v.strip())
    ids_str = [s for s in ids_str if s]

    if not turns_dir.exists():
        return None

    # Cheap search: scan a limited prefix of directory listing
    files = list(turns_dir.glob("*.json*"))  # .json, .jsonl, .json.gz etc.
    for ident in ids_str:
        for fp in files:
            if ident in fp.name:
                return fp
    return None


# -----------------------------
# Main: print examples
# -----------------------------
def inspect_one_jsonl(jsonl_path: str, turns_dir: Path, n_samples: int) -> None:
    p = Path(jsonl_path)
    print("\n" + "=" * 90)
    print(f"JSONL: {p}  (exists={p.exists()})")
    if not p.exists():
        return

    samples = []
    for rec in iter_jsonl(p):
        samples.append(rec)
        if len(samples) >= n_samples:
            break

    if not samples:
        print("No readable records found.")
        return

    # Show union of keys (from first ~50 records would be nicer; keep it light)
    keys = sorted({k for r in samples for k in r.keys()})
    print(f"Sampled {len(samples)} records. Keys (up to {MAX_KEYS_TO_SHOW}):")
    print(shorten(", ".join(keys), 5000))

    for i, rec in enumerate(samples, 1):
        print(f"\n--- Sample record {i} ---")
        id_fields = extract_id_fields(rec)
        if id_fields:
            print("ID-like fields:", ", ".join([f"{k}={repr(v)}" for k, v in id_fields]))
        else:
            print("ID-like fields: (none of the common keys found)")

        tf = find_turns_file(turns_dir, rec)
        print(f"Matched turns file: {tf if tf else '(no match found via heuristics)'}")

        print("\nRecord preview:")
        print(pretty(rec))


def main():
    print(f"TURNS_DIR: {TURNS_DIR} (exists={TURNS_DIR.exists()})")
    for jp in JSONL_PATHS:
        inspect_one_jsonl(jp, TURNS_DIR, N_SAMPLES_PER_FILE)


if __name__ == "__main__":
    main()
