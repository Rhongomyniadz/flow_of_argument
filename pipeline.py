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
DEFAULT_OUT_ROOT = Path("results/political_samples")

# =========================================================
# STRICT turn keys (as shown in your earlier pipeline style)
# =========================================================
TURN_TEXT_KEY = "turn_text"
SPEAKER_ID_KEY = "speaker_id"
SPEAKER_NAME_KEY = "inferred_speaker_name"
SPEAKER_ROLE_KEY = "inferred_speaker_role"

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
    with open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                log.warning("JSONL decode error %s:%d: %s", path, line_no, e)
                continue
            if isinstance(obj, dict):
                yield obj


def id_to_turns_path(turns_dir: Path, id_value: int) -> Tuple[Path, Path]:
    """
    Observed structure: .../<firstdigit>/<seconddigit>/<id>.jsonl
    Returns: (jsonl_path, jsonl_gz_path)
    """
    s = str(int(id_value))
    d1 = s[0] if len(s) >= 1 else "0"
    d2 = s[1] if len(s) >= 2 else "0"
    p = turns_dir / d1 / d2 / f"{s}.jsonl"
    pgz = turns_dir / d1 / d2 / f"{s}.jsonl.gz"
    return p, pgz


def load_episode_ids(episodes_jsonl: Path) -> List[int]:
    ids: List[int] = []
    for rec in iter_jsonl(episodes_jsonl):
        v = rec.get("id")
        if v is None:
            continue
        try:
            ids.append(int(v))
        except Exception:
            continue
    return ids


def extract_turn_strict(turn: Dict[str, Any]) -> Tuple[Optional[str], Any, Optional[str], Optional[str]]:
    """
    STRICT: uses only keys:
      - turn_text
      - speaker_id
      - inferred_speaker_name
      - inferred_speaker_role
    """
    txt = turn.get(TURN_TEXT_KEY)
    if isinstance(txt, str):
        txt = txt.strip()
    else:
        txt = None

    spk = turn.get(SPEAKER_ID_KEY)
    if isinstance(spk, list) and spk:
        spk = spk[0]

    name = turn.get(SPEAKER_NAME_KEY)
    if isinstance(name, str):
        name = name.strip()
    else:
        name = None

    role = turn.get(SPEAKER_ROLE_KEY)
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


def _dedup_cross_lists(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
# Sampling turns from political data (by episode id -> turns file)
# =========================================================
def sample_turns_from_political_data(
    episodes_jsonl: Path,
    turns_dir: Path,
    n_samples: int,
    min_words: int,
    seed: int,
    episodes_to_scan: int,
    per_episode_cap: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    log.info("Loading episode ids from %s", episodes_jsonl)
    ids = load_episode_ids(episodes_jsonl)
    log.info("Episode ids loaded: %d", len(ids))

    # keep only ids with existing turns file
    existing: List[int] = []
    for eid in ids:
        p, pgz = id_to_turns_path(turns_dir, eid)
        if p.exists() or pgz.exists():
            existing.append(eid)
    log.info("Episode ids with turns files present: %d", len(existing))
    if not existing:
        return []

    rng.shuffle(existing)
    scan_list = existing if episodes_to_scan <= 0 else existing[: min(episodes_to_scan, len(existing))]
    log.info("Scanning %d episodes for sampling (episodes_to_scan=%d).", len(scan_list), episodes_to_scan)

    # global reservoir (approx uniform over scanned subset)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for eid in tqdm(scan_list, desc="Scanning episodes"):
        p, pgz = id_to_turns_path(turns_dir, eid)
        turn_file = p if p.exists() else pgz if pgz.exists() else None
        if turn_file is None:
            continue

        # per-episode reservoir to avoid huge episodes dominating
        ep_res: List[Dict[str, Any]] = []
        ep_seen = 0

        for turn in iter_jsonl(turn_file):
            txt, spk, name, role = extract_turn_strict(turn)
            if not txt or count_words(txt) < min_words:
                continue

            ep_seen += 1
            item = {
                "episode_id": eid,
                "turn_file": str(turn_file),
                "turn_text": txt,
                "speaker_id": spk,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role,
            }

            if len(ep_res) < per_episode_cap:
                ep_res.append(item)
            else:
                j = rng.randrange(ep_seen)
                if j < per_episode_cap:
                    ep_res[j] = item

        for item in ep_res:
            seen += 1
            if len(reservoir) < n_samples:
                reservoir.append(item)
            else:
                j = rng.randrange(seen)
                if j < n_samples:
                    reservoir[j] = item

    log.info("Sampled turns: %d (global seen candidates=%d).", len(reservoir), seen)
    return reservoir


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Sample N political interview turns and run ONLY Prompt 3 via vLLM.")
    ap.add_argument("--episodes_jsonl", type=str, default=str(DEFAULT_EPISODES_JSONL))
    ap.add_argument("--turns_dir", type=str, default=str(DEFAULT_TURNS_DIR))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes_to_scan", type=int, default=800, help="0 = scan all episodes with turns files")
    ap.add_argument("--per_episode_cap", type=int, default=20)
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

    sampled_turns = sample_turns_from_political_data(
        episodes_jsonl=episodes_jsonl,
        turns_dir=turns_dir,
        n_samples=args.n_samples,
        min_words=args.min_words,
        seed=args.seed,
        episodes_to_scan=args.episodes_to_scan,
        per_episode_cap=args.per_episode_cap,
    )

    if not sampled_turns:
        log.warning("No turns sampled. Try lowering --min_words or increasing --episodes_to_scan.")
        return

    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=1200,
    )

    prompts = [PROMPT_3.format(turn_text=t["turn_text"]) for t in sampled_turns]

    outputs: List[str] = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating (prompt3)"):
        chunk = prompts[start : start + args.batch_size]
        outputs.extend(llm.generate_batch(chunk))

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

    print("\n==== Preview (first 5) ====")
    for i, r in enumerate(results[:5], 1):
        print(f"\n[{i}] episode_id={r.get('episode_id')} speaker={r.get('inferred_speaker_name')!r} role={r.get('inferred_speaker_role')!r}")
        txt = (r.get("turn_text") or "").replace("\n", " ")
        print("turn_text:", txt[:240] + ("..." if len(txt) > 240 else ""))
        eps = r.get("explicit_propositions") or []
        asm = r.get("assumptions") or []
        print("EP:", [e["text"] for e in eps[:2]] if eps else [])
        print("A :", [a["text"] for a in asm[:2]] if asm else [])


if __name__ == "__main__":
    main()
