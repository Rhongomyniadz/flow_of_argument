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
DEFAULT_OUT_ROOT = Path("results/political_prompt3_samples")

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("political-prompt3-sampler-v2")

_WORD_RE = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


# =========================================================
# Episode id loading (use what you showed: "id")
# =========================================================
def load_episode_ids(episodes_jsonl: Path) -> List[int]:
    ids: List[int] = []
    with open_text(episodes_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("id") is not None:
                try:
                    ids.append(int(rec["id"]))
                except Exception:
                    pass
    return ids


# =========================================================
# Turns index: id -> actual file path by filename stem
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
#   - First try JSONL (1 JSON per line)
#   - If that yields nothing, fall back to JSONDecoder.raw_decode scanning
# =========================================================
def iter_json_objects_jsonl(path: Path) -> Iterator[Any]:
    any_parsed = False
    bad = 0
    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                any_parsed = True
                yield obj
            except Exception:
                bad += 1
                continue
    # caller can detect "yielded nothing" by checking separately
    if not any_parsed and bad > 0:
        # silent; debug happens in probe
        return


def iter_json_objects_rawdecode(path: Path) -> Iterator[Any]:
    # Works for: multi-line JSON objects, concatenated JSON objects, JSON arrays, etc.
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
            # give up
            break
        yield obj
        i = j


def iter_turn_records(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Normalize possible wrappers:
      - dict turn
      - dict with 'turns' or 'speakerTurns' etc.
      - list of turns
    """
    # Try JSONL first
    yielded = 0
    for obj in iter_json_objects_jsonl(path):
        yielded += 1
        for t in _explode_obj(obj):
            if isinstance(t, dict):
                yield t

    if yielded > 0:
        return

    # Fallback: raw_decode scanning
    for obj in iter_json_objects_rawdecode(path):
        for t in _explode_obj(obj):
            if isinstance(t, dict):
                yield t


def _explode_obj(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        # common wrappers
        for k in ("turns", "speakerTurns", "utterances", "data"):
            v = obj.get(k)
            if isinstance(v, list) and v:
                return v
        return [obj]
    if isinstance(obj, list):
        return obj
    return []


# =========================================================
# Turn extraction: use known keys, else auto-pick best string field
# =========================================================
@dataclass
class ExtractedTurn:
    text: str
    speaker_id: Any
    speaker_name: Optional[str]
    speaker_role: Optional[str]
    chosen_text_key: str


def _best_text_field(turn: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    If standard keys missing, pick the best string field.
    Prefers keys containing text-ish substrings and longer content.
    Returns (text, key).
    """
    best = None  # (score, key, text)
    for k, v in turn.items():
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        lk = k.lower()
        # heuristic scoring
        score = len(s)
        if "text" in lk:
            score += 5000
        if "turn" in lk and "text" in lk:
            score += 8000
        if "transcript" in lk or "utter" in lk or "content" in lk:
            score += 4000
        if best is None or score > best[0]:
            best = (score, k, s)
    if best is None:
        return None, None
    return best[2], best[1]


def extract_turn(turn: Dict[str, Any]) -> Optional[ExtractedTurn]:
    # Text: your expected keys first
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

    # Speaker id
    spk = turn.get("speaker_id", None)
    if spk is None:
        spk = turn.get("speaker", None)
    if isinstance(spk, list) and spk:
        spk = spk[0]

    # Speaker name / role
    name = turn.get("inferred_speaker_name")
    if name is None:
        name = turn.get("inferredSpeakerName")
    if isinstance(name, str):
        name = name.strip()
    else:
        name = None

    role = turn.get("inferred_speaker_role")
    if role is None:
        role = turn.get("inferredSpeakerRole")
    if isinstance(role, str):
        role = role.strip()
    else:
        role = None

    return ExtractedTurn(
        text=txt,
        speaker_id=spk,
        speaker_name=name,
        speaker_role=role,
        chosen_text_key=chosen_key,
    )


# =========================================================
# Prompt 3 ONLY + parsing normalization (same as your style)
# =========================================================
_CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_json_candidates(txt: str) -> Iterable[str]:
    if not txt:
        return
    for m in _CODE_FENCE_RE.finditer(txt):
        block = m.group(1).strip()
        if block:
            yield block
    s = txt
    start_idx = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    yield s[start_idx : i + 1]


def _to_float_conf(v: Any, default: float = 0.5) -> float:
    try:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            xs = v.strip().replace("%", "")
            x = float(xs)
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
# Sampling: stop once we have N turns (no need to scan all 9663)
# =========================================================
def sample_turns(
    episode_ids: List[int],
    turns_index: Dict[int, Path],
    n_samples: int,
    min_words: int,
    seed: int,
    episodes_to_try: int,
    probe_k: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    existing = [eid for eid in episode_ids if eid in turns_index]
    log.info("Episode ids loaded: %d", len(episode_ids))
    log.info("Episode ids with turns files present: %d", len(existing))
    if not existing:
        return []

    rng.shuffle(existing)
    to_try = existing if episodes_to_try <= 0 else existing[: min(episodes_to_try, len(existing))]
    log.info("Trying up to %d episodes to collect %d turns.", len(to_try), n_samples)

    samples: List[Dict[str, Any]] = []

    # probe: show what one turns file actually looks like and what text field we pick
    if probe_k > 0:
        print("\n==== PROBE (turns file reality check) ====")
        shown = 0
        for eid in to_try:
            p = turns_index[eid]
            first_obj = None
            for obj in iter_turn_records(p):
                first_obj = obj
                break
            if first_obj is None:
                print(f"id={eid} file={p}  -> NO parsed objects (parser failed)")
            else:
                et = extract_turn(first_obj)
                keys = list(first_obj.keys())[:30]
                print(f"id={eid} file={p}")
                print(f"  first_record_keys(first 30): {keys}")
                if et is None:
                    print("  extract_turn: FAILED (no usable text field found)")
                else:
                    wc = count_words(et.text)
                    snip = et.text.replace("\n", " ")[:200] + ("..." if len(et.text) > 200 else "")
                    print(f"  extract_turn: OK  chosen_text_key={et.chosen_text_key!r}  words={wc}")
                    print(f"  snippet: {snip}")
            shown += 1
            if shown >= probe_k:
                break

    for eid in tqdm(to_try, desc="Sampling episodes"):
        p = turns_index[eid]

        # reservoir sample ONE good turn from this episode file
        chosen: Optional[Dict[str, Any]] = None
        seen_good = 0

        for turn in iter_turn_records(p):
            et = extract_turn(turn)
            if et is None:
                continue
            if count_words(et.text) < min_words:
                continue

            seen_good += 1
            item = {
                "episode_id": eid,
                "turn_file": str(p),
                "turn_text": et.text,
                "speaker_id": et.speaker_id,
                "inferred_speaker_name": et.speaker_name,
                "inferred_speaker_role": et.speaker_role,
                "chosen_text_key": et.chosen_text_key,
            }
            if chosen is None:
                chosen = item
            else:
                # classic reservoir sampling
                if rng.randrange(seen_good) == 0:
                    chosen = item

        if chosen is not None:
            samples.append(chosen)
            if len(samples) >= n_samples:
                break

    log.info("Sampled turns: %d", len(samples))
    return samples


def main():
    ap = argparse.ArgumentParser(description="Sample N political turns and run Prompt 3 via vLLM (robust parsing).")
    ap.add_argument("--episodes_jsonl", type=str, default=str(DEFAULT_EPISODES_JSONL))
    ap.add_argument("--turns_dir", type=str, default=str(DEFAULT_TURNS_DIR))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUT_ROOT))

    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes_to_try", type=int, default=2000, help="0 = try all")
    ap.add_argument("--probe_k", type=int, default=3)

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
    ep_ids = load_episode_ids(episodes_jsonl)
    log.info("Episode ids loaded: %d", len(ep_ids))

    log.info("Building turns index from %s", turns_dir)
    turns_index = build_turns_index(turns_dir)
    log.info("Indexed turns files: %d", len(turns_index))

    sampled = sample_turns(
        episode_ids=ep_ids,
        turns_index=turns_index,
        n_samples=args.n_samples,
        min_words=args.min_words,
        seed=args.seed,
        episodes_to_try=args.episodes_to_try,
        probe_k=args.probe_k,
    )

    if not sampled:
        log.error("Still sampled 0 turns. The PROBE above will tell whether parsing fails or the text key differs.")
        return

    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=1200,
    )

    prompts = [PROMPT_3.format(turn_text=t["turn_text"]) for t in sampled]

    outputs: List[str] = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating (prompt3)"):
        outputs.extend(llm.generate_batch(prompts[start : start + args.batch_size]))

    parsed_ok = 0
    results = []
    raw_results = []

    for t, raw_out in zip(sampled, outputs):
        norm = parse_and_normalize_llm(raw_out)
        base = {k: t[k] for k in t.keys()}
        if norm is not None:
            parsed_ok += 1
            results.append({**base, **norm})
        else:
            results.append(base)
        raw_results.append({"turn_text": t["turn_text"], "raw_output": raw_out})

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "raw").mkdir(parents=True, exist_ok=True)

    out_json = out_root / f"prompt3_political_samples{args.n_samples}.json"
    out_raw = out_root / "raw" / f"prompt3_political_samples{args.n_samples}_raw.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    log.info("Parsed OK: %d/%d", parsed_ok, len(results))
    log.info("Wrote: %s", out_json)
    log.info("Wrote: %s", out_raw)

    print("\n==== Preview (first 3) ====")
    for i, r in enumerate(results[:3], 1):
        txt = (r.get("turn_text") or "").replace("\n", " ")
        print(f"\n[{i}] episode_id={r.get('episode_id')} file={r.get('turn_file')}")
        print(f"  chosen_text_key={r.get('chosen_text_key')!r}")
        print(f"  speaker={r.get('inferred_speaker_name')!r} role={r.get('inferred_speaker_role')!r}")
        print("  turn_text:", txt[:240] + ("..." if len(txt) > 240 else ""))


if __name__ == "__main__":
    main()
