#!/usr/bin/env python3
import argparse
import gc
import gzip
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

# =========================================================
# Defaults (your political interview data)
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
log = logging.getLogger("political-prompt3-sampler")

_WORD_RE = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Robust JSONL iterator: expects 1 JSON object per line.
    (This avoids the 'Extra data' issue from json.load on a .jsonl file.)
    """
    with open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Keep it quiet at INFO level; noisy otherwise.
                continue
            if isinstance(obj, dict):
                yield obj


def load_episode_ids(episodes_jsonl: Path) -> List[int]:
    ids: List[int] = []
    with open_text(episodes_jsonl) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            v = rec.get("id")  # use what you showed: "id"
            if v is None:
                continue
            try:
                ids.append(int(v))
            except Exception:
                continue
    return ids


def turns_file_id(path: Path) -> Optional[int]:
    """
    Extract episode id from filenames like:
      86945.jsonl
      86945.jsonl.gz
    """
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
    """
    Index existing turns files by id -> path, using the actual directory content
    (matching your inspect.py behavior), rather than assuming a sharding scheme.
    """
    idx: Dict[int, Path] = {}

    # Prefer .jsonl over .jsonl.gz if both exist for same id
    # (keep whichever is uncompressed).
    candidates = list(turns_dir.rglob("*.jsonl")) + list(turns_dir.rglob("*.jsonl.gz"))
    for p in candidates:
        eid = turns_file_id(p)
        if eid is None:
            continue
        if eid not in idx:
            idx[eid] = p
        else:
            # prefer .jsonl
            if idx[eid].name.endswith(".jsonl.gz") and p.name.endswith(".jsonl"):
                idx[eid] = p
    return idx


# =========================================================
# Turn extraction (no key-dumping; just known schema)
# =========================================================
def extract_turn(turn: Dict[str, Any]) -> Tuple[Optional[str], Any, Optional[str], Optional[str]]:
    # Text
    txt = turn.get("turn_text")
    if txt is None:
        txt = turn.get("turnText")
    if isinstance(txt, str):
        txt = txt.strip()
    else:
        txt = None

    # Speaker id
    spk = turn.get("speaker_id")
    if spk is None:
        spk = turn.get("speaker")
    if isinstance(spk, list) and spk:
        spk = spk[0]

    # Speaker name
    name = turn.get("inferred_speaker_name")
    if name is None:
        name = turn.get("inferredSpeakerName")
    if isinstance(name, str):
        name = name.strip()
    else:
        name = None

    # Speaker role
    role = turn.get("inferred_speaker_role")
    if role is None:
        role = turn.get("inferredSpeakerRole")
    if isinstance(role, str):
        role = role.strip()
    else:
        role = None

    return txt, spk, name, role


# ---------------- Balanced JSON extraction ----------------
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
        if cand:
            key = cand["text"].casefold()
            if key not in seen:
                seen.add(key)
                out.append(cand)
        return out[:cap]

    for x in items:
        cand = _norm_item(x)
        if not cand:
            continue
        key = cand["text"].casefold()
        if key in seen:
            for i, y in enumerate(out):
                if y["text"].casefold() == key and cand["confidence"] > y["confidence"]:
                    out[i] = cand
            continue
        seen.add(key)
        out.append(cand)

    out.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return out[:cap]


def _dedup_cross_lists(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prim_keys = {p["text"].casefold() for p in primary}
    sec_clean = [s for s in secondary if s["text"].casefold() not in prim_keys]
    return primary, sec_clean


def parse_and_normalize_llm(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    for cand in _iter_json_candidates(text):
        try:
            obj = json.loads(cand)
            if not isinstance(obj, dict):
                continue
            exp = obj.get("explicit_propositions", [])
            asm = obj.get("assumptions", [])
            exp_n = _normalize_list(exp, cap=10)
            asm_n = _normalize_list(asm, cap=10)
            exp_n, asm_n = _dedup_cross_lists(exp_n, asm_n)
            return {"explicit_propositions": exp_n, "assumptions": asm_n}
        except Exception:
            continue
    return None


# =========================================================
# vLLM Wrapper
# =========================================================
class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
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
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
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
# Prompt 3 ONLY
# =========================================================
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
# Sampling turns with correct id->file matching
# =========================================================
def sample_turns(
    episode_ids: List[int],
    turns_index: Dict[int, Path],
    n_samples: int,
    min_words: int,
    seed: int,
    episodes_to_scan: int,
    per_episode_cap: int,
    probe_k: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    existing = [eid for eid in episode_ids if eid in turns_index]
    log.info("Episode ids loaded: %d", len(episode_ids))
    log.info("Episode ids with turns files present: %d", len(existing))
    if not existing:
        return []

    rng.shuffle(existing)
    scan_list = existing if episodes_to_scan <= 0 else existing[: min(episodes_to_scan, len(existing))]
    log.info("Scanning %d episodes for sampling (episodes_to_scan=%d).", len(scan_list), episodes_to_scan)

    # small join/probing preview
    if probe_k > 0:
        print("\n==== PROBE (id -> turns file -> one extracted turn) ====")
        shown = 0
        for eid in scan_list:
            p = turns_index[eid]
            # find first parseable turn
            first_ok = None
            for turn in iter_jsonl(p):
                txt, spk, name, role = extract_turn(turn)
                if txt:
                    first_ok = (txt, spk, name, role)
                    break
            if first_ok is None:
                # could be empty/unparseable file; still show mapping
                print(f"id={eid}  turns_file={p}  (no parseable turns)")
            else:
                txt, spk, name, role = first_ok
                wc = count_words(txt)
                snip = txt.replace("\n", " ")[:220] + ("..." if len(txt) > 220 else "")
                print(f"id={eid}  turns_file={p}")
                print(f"  speaker_id={spk!r} name={name!r} role={role!r} words={wc}")
                print(f"  turn_text: {snip}")
            shown += 1
            if shown >= probe_k:
                break

    reservoir: List[Dict[str, Any]] = []
    seen_global = 0

    for eid in tqdm(scan_list, desc="Scanning episodes"):
        p = turns_index[eid]

        ep_res: List[Dict[str, Any]] = []
        ep_seen = 0

        for turn in iter_jsonl(p):
            txt, spk, name, role = extract_turn(turn)
            if not txt:
                continue
            if count_words(txt) < min_words:
                continue

            ep_seen += 1
            item = {
                "episode_id": eid,
                "turn_file": str(p),
                "turn_text": txt,
                "speaker_id": spk,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role,
            }

            # per-episode cap reservoir (avoid one episode dominating)
            if len(ep_res) < per_episode_cap:
                ep_res.append(item)
            else:
                j = rng.randrange(ep_seen)
                if j < per_episode_cap:
                    ep_res[j] = item

        # global reservoir from the per-episode sampled items
        for item in ep_res:
            seen_global += 1
            if len(reservoir) < n_samples:
                reservoir.append(item)
            else:
                j = rng.randrange(seen_global)
                if j < n_samples:
                    reservoir[j] = item

    log.info("Sampled turns: %d (global seen candidates=%d).", len(reservoir), seen_global)
    return reservoir


def main():
    ap = argparse.ArgumentParser(description="Sample N political interview turns and run ONLY Prompt 3 via vLLM.")
    ap.add_argument("--episodes_jsonl", type=str, default=str(DEFAULT_EPISODES_JSONL))
    ap.add_argument("--turns_dir", type=str, default=str(DEFAULT_TURNS_DIR))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes_to_scan", type=int, default=10000, help="0 = scan all episodes with turns files")
    ap.add_argument("--per_episode_cap", type=int, default=20)
    ap.add_argument("--probe_k", type=int, default=5, help="Print K id->turn-file mappings and one turn snippet each")
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

    log.info("Building turns index from %s", turns_dir)
    turns_index = build_turns_index(turns_dir)
    log.info("Indexed turns files: %d", len(turns_index))

    episode_ids = load_episode_ids(episodes_jsonl)
    sampled_turns = sample_turns(
        episode_ids=episode_ids,
        turns_index=turns_index,
        n_samples=args.n_samples,
        min_words=args.min_words,
        seed=args.seed,
        episodes_to_scan=args.episodes_to_scan,
        per_episode_cap=args.per_episode_cap,
        probe_k=args.probe_k,
    )

    if not sampled_turns:
        log.warning(
            "No turns sampled. If PROBE shows 'no parseable turns', the turns files may not be JSONL (1 JSON per line) "
            "or use different text keys than expected."
        )
        return

    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=1200,
    )

    prompts = [PROMPT_3.format(turn_text=t["turn_text"]) for t in sampled_turns]

    outputs: List[str] = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating (prompt3)"):
        outputs.extend(llm.generate_batch(prompts[start : start + args.batch_size]))

    results: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    parsed_ok = 0

    for t, raw_out in zip(sampled_turns, outputs):
        norm = parse_and_normalize_llm(raw_out)
        base = {
            "episode_id": t["episode_id"],
            "turn_file": t["turn_file"],
            "speaker_id": t["speaker_id"],
            "inferred_speaker_name": t["inferred_speaker_name"],
            "inferred_speaker_role": t["inferred_speaker_role"],
            "turn_text": t["turn_text"],
        }
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

    gc.collect()

    log.info("Parsed OK: %d/%d", parsed_ok, len(results))
    log.info("Wrote: %s", out_json)
    log.info("Wrote: %s", out_raw)

    print("\n==== Preview (first 3) ====")
    for i, r in enumerate(results[:3], 1):
        txt = (r.get("turn_text") or "").replace("\n", " ")
        print(f"\n[{i}] episode_id={r.get('episode_id')} file={r.get('turn_file')}")
        print(f"  speaker={r.get('inferred_speaker_name')!r} role={r.get('inferred_speaker_role')!r}")
        print("  turn_text:", txt[:240] + ("..." if len(txt) > 240 else ""))


if __name__ == "__main__":
    main()
