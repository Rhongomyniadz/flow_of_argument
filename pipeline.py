import argparse
import gzip
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

# =========================================================
# Defaults (YOUR political interview data)
# =========================================================
DEFAULT_EPISODES_JSONL = Path(
    "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpsDataCleaned_Interviews_withDBIds.jsonl"
)
DEFAULT_TURNS_DIR = Path(
    "/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes_interviews"
)
DEFAULT_OUT_ROOT = Path("results")

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("political-prompt3-by-episode")

_WORD_RE = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


# =========================================================
# Load episode metadata (use key shown in your example: "id")
# =========================================================
def load_episode_id_and_meta(episodes_jsonl: Path) -> Dict[int, Dict[str, Any]]:
    """
    Returns: {episode_id: {title_ep, pubDate_ep, rssKey, guid_ep, key}}
    (Only fields we might want for debugging / manifest.)
    """
    out: Dict[int, Dict[str, Any]] = {}
    with open_text(episodes_jsonl) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("id") is None:
                continue
            try:
                eid = int(rec["id"])
            except Exception:
                continue

            out[eid] = {
                "title_ep": rec.get("title_ep"),
                "pubDate_ep": rec.get("pubDate_ep"),
                "rssKey": rec.get("rssKey"),
                "guid_ep": rec.get("guid_ep"),
                "key": rec.get("key"),
            }
    return out


# =========================================================
# Turns index: episode_id -> turns file by filename stem
# =========================================================
def turns_file_id(path: Path) -> Optional[int]:
    name = path.name
    if name.endswith(".jsonl.gz"):
        base = name[: -len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        base = name[: -len(".jsonl")]
    else:
        return None
    try:
        return int(base)
    except Exception:
        return None


def build_turns_index(turns_dir: Path) -> Dict[int, Path]:
    idx: Dict[int, Path] = {}
    candidates = list(turns_dir.rglob("*.jsonl")) + list(turns_dir.rglob("*.jsonl.gz"))
    for p in candidates:
        eid = turns_file_id(p)
        if eid is None:
            continue
        if eid not in idx:
            idx[eid] = p
        else:
            # prefer uncompressed .jsonl if both exist
            if idx[eid].name.endswith(".jsonl.gz") and p.name.endswith(".jsonl"):
                idx[eid] = p
    return idx


# =========================================================
# Robust JSON reading for turns files
#   - Try JSONL (1 JSON per line) first
#   - If that yields nothing, fall back to JSONDecoder.raw_decode scan
# =========================================================
def iter_json_objects_jsonl(path: Path) -> Iterator[Any]:
    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def iter_json_objects_rawdecode(path: Path) -> Iterator[Any]:
    with open_text(path) as f:
        text = f.read()

    s = text.lstrip()
    if not s:
        return

    # JSON array case
    if s[0] == "[":
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                for x in arr:
                    yield x
            else:
                yield arr
        except Exception:
            return
        return

    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except Exception:
            break
        yield obj
        i = j


def _explode_obj(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for k in ("turns", "speakerTurns", "utterances", "data"):
            v = obj.get(k)
            if isinstance(v, list) and v:
                return v
        return [obj]
    if isinstance(obj, list):
        return obj
    return []


def iter_turn_records(path: Path) -> Iterator[Dict[str, Any]]:
    yielded_any = False
    for obj in iter_json_objects_jsonl(path):
        yielded_any = True
        for t in _explode_obj(obj):
            if isinstance(t, dict):
                yield t
    if yielded_any:
        return
    for obj in iter_json_objects_rawdecode(path):
        for t in _explode_obj(obj):
            if isinstance(t, dict):
                yield t


# =========================================================
# Turn extraction: known keys first, else auto-pick best string field
# =========================================================
@dataclass
class ExtractedTurn:
    turn_idx: int
    text: str
    speaker_id: Any
    speaker_name: Optional[str]
    speaker_role: Optional[str]
    chosen_text_key: str


def _best_text_field(turn: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    best = None  # (score, key, text)
    for k, v in turn.items():
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        lk = k.lower()
        score = len(s)
        if "turn" in lk and "text" in lk:
            score += 8000
        if "text" in lk:
            score += 5000
        if "transcript" in lk or "utter" in lk or "content" in lk:
            score += 4000
        if best is None or score > best[0]:
            best = (score, k, s)
    if best is None:
        return None, None
    return best[2], best[1]


def extract_turn(turn: Dict[str, Any], turn_idx: int) -> Optional[ExtractedTurn]:
    chosen_key = ""
    txt = None

    if isinstance(turn.get("turn_text"), str):
        txt = turn["turn_text"].strip()
        chosen_key = "turn_text"
    elif isinstance(turn.get("turnText"), str):
        txt = turn["turnText"].strip()
        chosen_key = "turnText"

    if not txt:
        txt2, k2 = _best_text_field(turn)
        if txt2:
            txt = txt2
            chosen_key = k2 or ""

    if not txt:
        return None

    spk = turn.get("speaker_id", None)
    if spk is None:
        spk = turn.get("speaker", None)
    if isinstance(spk, list) and spk:
        spk = spk[0]

    name = turn.get("inferred_speaker_name")
    if name is None:
        name = turn.get("inferredSpeakerName")
    name = name.strip() if isinstance(name, str) else None

    role = turn.get("inferred_speaker_role")
    if role is None:
        role = turn.get("inferredSpeakerRole")
    role = role.strip() if isinstance(role, str) else None

    return ExtractedTurn(
        turn_idx=turn_idx,
        text=txt,
        speaker_id=spk,
        speaker_name=name,
        speaker_role=role,
        chosen_text_key=chosen_key,
    )


def load_episode_turns(turns_path: Path, min_words: int) -> List[ExtractedTurn]:
    out: List[ExtractedTurn] = []
    for i, t in enumerate(iter_turn_records(turns_path)):
        et = extract_turn(t, turn_idx=i)
        if et is None:
            continue
        if count_words(et.text) < min_words:
            continue
        out.append(et)
    return out


# =========================================================
# Prompt 3 + parsing normalization
# =========================================================
_CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_json_candidates(txt: str) -> Iterable[str]:
    if not txt:
        return
    for m in _CODE_FENCE_RE.finditer(txt):
        block = m.group(1).strip()
        if block:
            yield block
    start_idx = None
    depth = 0
    for i, ch in enumerate(txt):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    yield txt[start_idx : i + 1]


def _to_float_conf(v: Any, default: float = 0.5) -> float:
    try:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            x = float(v.strip().replace("%", ""))
        else:
            return default
        if x != x:
            return default
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def _norm_item(x: Any, default_conf: float = 0.7) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        text = x.get("text")
        conf = x.get("confidence", 0.5)
    elif isinstance(x, str):
        text = x
        conf = default_conf
    else:
        return None
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return {"text": text, "confidence": _to_float_conf(conf, default_conf)}


def _normalize_list(items: Any, cap: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    if items is None:
        return out
    if isinstance(items, dict) and "list" in items and isinstance(items["list"], list):
        items = items["list"]
    if not isinstance(items, (list, tuple)):
        cand = _norm_item(items)
        if cand and cand["text"].casefold() not in seen:
            seen.add(cand["text"].casefold())
            out.append(cand)
        return out[:cap]
    for x in items:
        cand = _norm_item(x)
        if not cand:
            continue
        key = cand["text"].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    out.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return out[:cap]


def parse_and_normalize_llm(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    for cand in _iter_json_candidates(text):
        try:
            obj = json.loads(cand)
            if not isinstance(obj, dict):
                continue
            exp = _normalize_list(obj.get("explicit_propositions", []), cap=10)
            asm = _normalize_list(obj.get("assumptions", []), cap=10)
            exp_keys = {e["text"].casefold() for e in exp}
            asm = [a for a in asm if a["text"].casefold() not in exp_keys]
            return {"explicit_propositions": exp, "assumptions": asm}
        except Exception:
            continue
    return None


PROMPT_3 = """SYSTEM:
You are a pragmatics and social cognition expert.
Interpret the social, emotional, and interpersonal dimensions of the turn.

DEFINITIONS:
- Explicit propositions: Direct statements/claims/descriptions.
- Assumptions: Deeper social/affective beliefs (trust, respect, authority, identity, morality) that underlie the stance.

RULES:
- Return UP TO 10 items per list, ordered by MOST → LEAST salient for social meaning.
- Explicit propositions must be atomic, content-bearing claims with confidence (0.0-1.0).
- Assumptions should be generalized beliefs (pass a “generality test” if entities become roles), each with confidence (0.0-1.0).
- No duplication between Explicit propositions and Assumptions; avoid trivial paraphrases.
- Strict JSON ONLY; no extra keys.

TASK:
Given the following text:
"{turn_text}"

OUTPUT FORMAT:
{{
  "explicit_propositions": [
    {{"text": "...", "confidence": 0.93}},
    {{"text": "...", "confidence": 0.89}}
  ],
  "assumptions": [
    {{"text": "Status and expertise warrant deference in public discussions.", "confidence": 0.90}},
    {{"text": "Personal narratives build trust with an audience.", "confidence": 0.86}}
  ]
}}"""


# =========================================================
# vLLM wrapper
# =========================================================
class LLMInterface:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.6,
        top_p: float = 0.95,
        min_p: float = 0.1,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
        max_tokens: int = 1200,
        max_model_len: int = 100000,
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        outs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in outs]


# =========================================================
# Main: sample N episodes; write one json per episode id
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Sample N episodes; run Prompt 3; save one JSON per episode id.")
    ap.add_argument("--episodes_jsonl", type=str, default=str(DEFAULT_EPISODES_JSONL))
    ap.add_argument("--turns_dir", type=str, default=str(DEFAULT_TURNS_DIR))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUT_ROOT))

    ap.add_argument("--num_episodes", type=int, default=50)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # optional safety cap; default = no cap
    ap.add_argument("--max_turns_per_episode", type=int, default=0, help="0 means no cap; else max turns to process per episode.")

    ap.add_argument("--episodes_to_try", type=int, default=20000, help="Max eligible episodes to consider while filling num_episodes.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)

    args = ap.parse_args()

    episodes_jsonl = Path(args.episodes_jsonl)
    turns_dir = Path(args.turns_dir)
    out_root = Path(args.output_root)

    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"Episodes JSONL not found: {episodes_jsonl}")
    if not turns_dir.exists():
        raise FileNotFoundError(f"Turns dir not found: {turns_dir}")

    log.info("Loading episode ids from %s", episodes_jsonl)
    ep_meta = load_episode_id_and_meta(episodes_jsonl)
    ep_ids = list(ep_meta.keys())
    log.info("Episode ids loaded: %d", len(ep_ids))

    log.info("Building turns index from %s", turns_dir)
    turns_index = build_turns_index(turns_dir)
    log.info("Indexed turns files: %d", len(turns_index))

    eligible = [eid for eid in ep_ids if eid in turns_index]
    log.info("Episode ids with turns files present: %d", len(eligible))
    if not eligible:
        log.error("No eligible episode ids (id not found as a turns filename).")
        return

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    eligible = eligible[: min(len(eligible), args.episodes_to_try)]

    parsed_dir = out_root / "political"
    raw_dir = out_root / "raw"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMInterface(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size)

    manifest = {
        "episodes_jsonl": str(episodes_jsonl),
        "turns_dir": str(turns_dir),
        "num_episodes_target": args.num_episodes,
        "min_words": args.min_words,
        "max_turns_per_episode": args.max_turns_per_episode,
        "seed": args.seed,
        "model_name": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "episodes_written": [],
        "episodes_skipped_no_turns": 0,
    }

    written = 0
    for eid in tqdm(eligible, desc="Episodes processed"):
        if written >= args.num_episodes:
            break

        turns_path = turns_index[eid]
        turns = load_episode_turns(turns_path, min_words=args.min_words)
        if not turns:
            manifest["episodes_skipped_no_turns"] += 1
            continue

        if args.max_turns_per_episode and args.max_turns_per_episode > 0 and len(turns) > args.max_turns_per_episode:
            # Keep chronological: take first K (you can change to random sample if you prefer)
            turns = turns[: args.max_turns_per_episode]

        prompts = [PROMPT_3.format(turn_text=t.text) for t in turns]

        outputs: List[str] = []
        for start in range(0, len(prompts), args.batch_size):
            outputs.extend(llm.generate_batch(prompts[start : start + args.batch_size]))

        parsed_rows = []
        raw_rows = []
        parsed_ok = 0

        for t, raw_out in zip(turns, outputs):
            norm = parse_and_normalize_llm(raw_out)
            base = {
                "episode_id": eid,
                "turn_file": str(turns_path),
                "turn_idx": t.turn_idx,
                "turn_text": t.text,
                "speaker_id": t.speaker_id,
                "inferred_speaker_name": t.speaker_name,
                "inferred_speaker_role": t.speaker_role,
                "chosen_text_key": t.chosen_text_key,
            }
            if norm is not None:
                parsed_ok += 1
                parsed_rows.append({**base, **norm})
            else:
                parsed_rows.append(base)

            raw_rows.append({"turn_idx": t.turn_idx, "turn_text": t.text, "raw_output": raw_out})

        out_parsed = parsed_dir / f"{eid}.json"          # <- filename is id
        out_raw = raw_dir / f"{eid}_raw.json"
        with open(out_parsed, "w", encoding="utf-8") as f:
            json.dump(parsed_rows, f, indent=2, ensure_ascii=False)
        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(raw_rows, f, indent=2, ensure_ascii=False)

        meta = ep_meta.get(eid, {})
        manifest["episodes_written"].append(
            {
                "episode_id": eid,
                "turn_file": str(turns_path),
                "n_turns_written": len(parsed_rows),
                "parsed_ok": parsed_ok,
                "parsed_path": str(out_parsed),
                "raw_path": str(out_raw),
                # these are just for convenience/debug
                "title_ep": meta.get("title_ep"),
                "pubDate_ep": meta.get("pubDate_ep"),
                "rssKey": meta.get("rssKey"),
                "guid_ep": meta.get("guid_ep"),
                "key": meta.get("key"),
            }
        )

        written += 1

    manifest_path = out_root / "manifest_prompt3_by_episode.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info("Done. Episode files written: %d", written)
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
