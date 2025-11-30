import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Only used to index numeric tokens in filenames for id/rssKey matching
DIGITS_RE = re.compile(r"\d{4,}")

# -----------------------------
# Defaults (your paths)
# -----------------------------
DEFAULT_TURNS_DIR = Path(
    "/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes_interviews"
)

DEFAULT_JSONLS = [
    "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpInterviews2.jsonl",
    "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpsDataCleaned_Interviews_withDBIds.jsonl",
]

DEFAULT_N_SAMPLES = 5


# -----------------------------
# IO helpers
# -----------------------------
def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open_text(path) as f:
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
                yield {"_non_dict_record": obj}


def read_json_or_jsonl(path: Path, max_jsonl_lines: Optional[int] = None) -> Any:
    # Treat .jsonl strictly as JSONL (even though it begins with "{")
    if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
        out = []
        with open_text(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] {path.name}:{i} JSONL decode error: {e}")
                    continue
                if max_jsonl_lines is not None and len(out) >= max_jsonl_lines:
                    break
        return out

    # Otherwise fall back to normal JSON
    with open_text(path) as f:
        return json.load(f)


# -----------------------------
# Turns indexing
# -----------------------------
def build_turns_index(turns_dir: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]], List[Path]]:
    files: List[Path] = []
    for ext in ("*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz"):
        files.extend(turns_dir.rglob(ext))
    files = sorted(set(files))

    stem_index: Dict[str, Path] = {}
    digits_index: Dict[str, List[Path]] = {}

    for fp in files:
        # regular stem
        stem_index[fp.stem] = fp

        # handle double suffix stems for .jsonl.gz etc.
        name = fp.name
        if name.endswith(".jsonl.gz"):
            stem2 = name[:-len(".jsonl.gz")]
        elif name.endswith(".json.gz"):
            stem2 = name[:-len(".json.gz")]
        elif name.endswith(".jsonl"):
            stem2 = name[:-len(".jsonl")]
        elif name.endswith(".json"):
            stem2 = name[:-len(".json")]
        else:
            stem2 = fp.stem
        stem_index.setdefault(stem2, fp)

        # numeric token indexing
        for m in DIGITS_RE.findall(fp.name):
            digits_index.setdefault(m, []).append(fp)

    return stem_index, digits_index, files


# -----------------------------
# Exactly-the-shown-fields extraction
# -----------------------------
def guid_ep_text(rec: Dict[str, Any]) -> Optional[str]:
    g = rec.get("guid_ep")
    if isinstance(g, dict):
        t = g.get("text")
        return t.strip() if isinstance(t, str) and t.strip() else None
    return None


def shown_fields(rec: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    ONLY the fields you showed from your JSONL examples.
    """
    out: Dict[str, Optional[str]] = {
        "id": None,
        "rssKey": None,
        "guid_ep.text": None,
        "key": None,
    }

    if rec.get("id") is not None:
        out["id"] = str(rec.get("id")).strip()

    if rec.get("rssKey") is not None:
        out["rssKey"] = str(rec.get("rssKey")).strip()

    out["guid_ep.text"] = guid_ep_text(rec)

    if isinstance(rec.get("key"), str) and rec["key"].strip():
        out["key"] = rec["key"].strip()

    return out


# -----------------------------
# Matching using ONLY shown fields
# -----------------------------
def match_turns_file(
    fields: Dict[str, Optional[str]],
    stem_index: Dict[str, Path],
    digits_index: Dict[str, List[Path]],
    all_files: List[Path],
) -> Tuple[Optional[Path], str]:
    """
    Matching order (ONLY):
      1) id
      2) rssKey
      3) guid_ep.text
      4) key
    Modes:
      - exact stem match
      - digit-token match (id/rssKey only)
      - substring match (guid/key)
    """
    order = ["id", "rssKey", "guid_ep.text", "key"]

    # 1) exact stem match
    for k in order:
        v = fields.get(k)
        if v and v in stem_index:
            return stem_index[v], f"stem=={k}:{v}"

    # 2) digit-token match (only for id/rssKey)
    for k in ["id", "rssKey"]:
        v = fields.get(k)
        if v and v.isdigit() and v in digits_index:
            paths = digits_index[v]
            if len(paths) == 1:
                return paths[0], f"digits=={k}:{v}"
            return paths[0], f"digits=={k}:{v} (COLLISION {len(paths)} files)"

    # 3) substring match (only for guid/key)
    for k in ["guid_ep.text", "key"]:
        v = fields.get(k)
        if v and len(v) >= 8:
            hits = [fp for fp in all_files if v in fp.name]
            if hits:
                return hits[0], f"substr=={k}:{v} (hits={len(hits)})"

    return None, "no_match"


# -----------------------------
# Turns schema preview (same as before)
# -----------------------------
SPEAKER_KEYS = ["speaker_id", "speakerId", "speaker", "speaker_name", "speakerName", "spk", "spk_id"]
TEXT_KEYS = ["turn_text", "text", "utterance", "content", "transcript", "sentence", "turnText"]
ASSUMPTION_KEYS = ["assumptions", "assumption_list", "implicit_assumptions", "assumptions_extracted"]


def extract_turns(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ["turns", "data", "utterances", "segments"]:
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [obj]
    return []


def pick_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def preview_turns_file(path: Path, n_turns: int = 3) -> None:
    try:
        raw = read_json_or_jsonl(path, max_jsonl_lines=max(50, n_turns))
    except Exception as e:
        print(f"[WARN] Failed to read turns file {path}: {e}")
        return

    turns = extract_turns(raw)
    print(f"Turns parsed: {len(turns)} (showing first {min(n_turns, len(turns))})")
    if not turns:
        print("No turns found / unrecognized format.")
        return

    t0 = turns[0]
    spk_k = pick_first_key(t0, SPEAKER_KEYS)
    txt_k = pick_first_key(t0, TEXT_KEYS)
    asm_k = pick_first_key(t0, ASSUMPTION_KEYS)

    print("Turn keys (first turn):", ", ".join(sorted(list(t0.keys()))[:80]))
    print(f"Inferred speaker key: {spk_k}")
    print(f"Inferred text key:    {txt_k}")
    print(f"Inferred assumptions: {asm_k}")

    for i, t in enumerate(turns[:n_turns], 1):
        spk = t.get(spk_k) if spk_k else None
        txt = t.get(txt_k) if txt_k else None
        asm = t.get(asm_k) if asm_k else None
        asm_len = len(asm) if isinstance(asm, list) else (0 if asm is None else 1)

        print(f"\n--- Turn {i} ---")
        print("speaker =", repr(spk))
        if isinstance(txt, str):
            print("text    =", repr(txt[:240] + ("..." if len(txt) > 240 else "")))
        else:
            print("text    =", repr(txt))
        print("assumps =", f"type={type(asm).__name__}, len={asm_len}")


# -----------------------------
# JSONL inspection
# -----------------------------
def inspect_jsonl(
    jsonl_path: Path,
    stem_index: Dict[str, Path],
    digits_index: Dict[str, List[Path]],
    all_files: List[Path],
    n_samples: int,
) -> None:
    print("\n" + "=" * 110)
    print(f"JSONL: {jsonl_path} (exists={jsonl_path.exists()})")
    if not jsonl_path.exists():
        return

    samples = []
    for rec in iter_jsonl(jsonl_path):
        samples.append(rec)
        if len(samples) >= n_samples:
            break

    if not samples:
        print("No readable records.")
        return

    for i, rec in enumerate(samples, 1):
        f = shown_fields(rec)

        print(f"\n--- Sample record {i} ---")
        print("title_ep =", repr(rec.get("title_ep")))
        print("pubDate  =", repr(rec.get("pubDate_ep")))
        print("id       =", f["id"])
        print("rssKey   =", f["rssKey"])
        print("guid_ep  =", repr(f["guid_ep.text"]))
        print("key      =", (repr(f["key"])[:120] + "...") if isinstance(f["key"], str) and len(f["key"]) > 120 else repr(f["key"]))

        fp, why = match_turns_file(f, stem_index, digits_index, all_files)
        print("MATCH    =", fp if fp else "(none)", "|", why)

        if fp:
            print("\nTurns file preview:")
            preview_turns_file(fp, n_turns=3)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns_dir", type=str, default=str(DEFAULT_TURNS_DIR))
    ap.add_argument("--jsonl", type=str, action="append", default=None,
                    help="pass multiple --jsonl; if omitted uses the built-in defaults")
    ap.add_argument("--n", type=int, default=DEFAULT_N_SAMPLES)
    args = ap.parse_args()

    turns_dir = Path(args.turns_dir)
    jsonls = args.jsonl if args.jsonl else DEFAULT_JSONLS

    print(f"TURNS_DIR: {turns_dir} (exists={turns_dir.exists()})")
    if not turns_dir.exists():
        raise FileNotFoundError(turns_dir)

    stem_index, digits_index, all_files = build_turns_index(turns_dir)
    print(f"Indexed turns files: {len(all_files)}")
    print("Example filenames:", [p.name for p in all_files[:10]])

    for jp in jsonls:
        inspect_jsonl(Path(jp), stem_index, digits_index, all_files, n_samples=args.n)


if __name__ == "__main__":
    main()
