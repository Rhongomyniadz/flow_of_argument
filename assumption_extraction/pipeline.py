import argparse
import gzip
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

# =========================================================
# Defaults (category-based raw episode JSON files)
# =========================================================
DEFAULT_RAW_ROOT = Path("raw")
DEFAULT_OUT_ROOT = Path("data")
DEFAULT_CATEGORIES = ["business", "commentary", "news", "religion", "sports"]

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-by-category")

_WORD_RE = re.compile(r"\w+")
_SPEAKER_TOKEN_RE = re.compile(r"SPEAKER_\d+")


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
# Turn extraction (NO dataclass; return dict)
# =========================================================
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


def normalize_speaker_id(spk: Any) -> Any:
    """
    Normalize speaker id into a single speaker token when possible.

    Examples:
    - "SPEAKER_00,SPEAKER_01" -> "SPEAKER_00"
    - ["SPEAKER_02", "SPEAKER_01"] -> "SPEAKER_02"
    """
    if spk is None:
        return None

    if isinstance(spk, list) and spk:
        return normalize_speaker_id(spk[0])

    if isinstance(spk, str):
        s = spk.strip()
        if not s:
            return None
        tokens = _SPEAKER_TOKEN_RE.findall(s)
        if tokens:
            return tokens[0]
        if "," in s:
            first = s.split(",", 1)[0].strip()
            return first or None
        return s

    return spk


def speaker_token_count(spk: Any) -> int:
    """
    Count how many speaker tokens are present in the raw speaker field.
    Used to detect noisy rows like "SPEAKER_01,SPEAKER_00".
    """
    if spk is None:
        return 0
    if isinstance(spk, list):
        cnt = 0
        for x in spk:
            cnt += speaker_token_count(x)
        return cnt
    if isinstance(spk, str):
        s = spk.strip()
        if not s:
            return 0
        tokens = _SPEAKER_TOKEN_RE.findall(s)
        if tokens:
            return len(tokens)
        if "," in s:
            return len([p for p in s.split(",") if p.strip()])
        return 1
    return 1


def extract_turn(turn: Dict[str, Any], turn_idx: int) -> Optional[Dict[str, Any]]:
    txt = None

    if isinstance(turn.get("turn_text"), str):
        txt = turn["turn_text"].strip()
    elif isinstance(turn.get("turnText"), str):
        txt = turn["turnText"].strip()

    if not txt:
        txt2, _ = _best_text_field(turn)
        if txt2:
            txt = txt2

    if not txt:
        return None

    raw_spk = turn.get("speaker_id", None)
    if raw_spk is None:
        raw_spk = turn.get("speaker", None)
    spk = normalize_speaker_id(raw_spk)

    return {
        "turn_idx": turn_idx,
        "text": txt,
        "speaker_id": spk,
        "has_multi_speaker": speaker_token_count(raw_spk) > 1,
        "meta": dict(turn),  # keep all original metadata (mfcc/F0/F1/times/etc.)
    }


def load_episode_turns_raw(turns_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(iter_turn_records(turns_path)):
        et = extract_turn(t, turn_idx=i)
        if et is None:
            continue
        out.append(et)
    return out


def discover_category_dirs(raw_root: Path, categories: Optional[List[str]], auto_categories: bool) -> List[str]:
    if auto_categories:
        return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    if categories:
        return categories
    return DEFAULT_CATEGORIES


def discover_episode_files(raw_root: Path, category: str) -> List[Path]:
    cat_dir = raw_root / category
    if not cat_dir.exists() or not cat_dir.is_dir():
        return []
    return sorted([p for p in cat_dir.rglob("*.json") if p.is_file()])


def infer_episode_id(turns_path: Path, raw_turns: List[Dict[str, Any]]) -> str:
    for t in raw_turns:
        meta = t.get("meta")
        if not isinstance(meta, dict):
            continue
        for key in ("episode_id", "episodeId", "id"):
            val = meta.get(key)
            if val is None:
                continue
            s = str(val).strip()
            if s:
                return s
    return turns_path.stem


def sanitize_filename_stem(value: Any) -> str:
    s = str(value).strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = s.strip("._")
    return s or "unknown_episode"


# =========================================================
# Merge + filter helpers (keep/aggregate metadata)
# =========================================================
def merge_multi_speaker_turns_into_previous(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For rows flagged as multi-speaker, merge them into the previous turn.
    This handles diarization artifacts such as "SPEAKER_01,SPEAKER_00".
    """
    if not turns:
        return []

    out: List[Dict[str, Any]] = []
    for t in turns:
        if t.get("has_multi_speaker") and out:
            prev = out[-1]
            prev_text = prev.get("text", "")
            cur_text = t.get("text", "")
            if prev_text and cur_text:
                prev["text"] = f"{prev_text} {cur_text}".strip()
            elif cur_text:
                prev["text"] = cur_text

            prev_meta = prev.get("meta")
            cur_meta = t.get("meta")
            if isinstance(prev_meta, dict) and isinstance(cur_meta, dict):
                for k, v in cur_meta.items():
                    if k in prev_meta:
                        continue
                    prev_meta[k] = v
            continue
        out.append(t)

    for i, t in enumerate(out):
        t["turn_idx"] = i
    return out


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _weighted_mean(pairs: List[Tuple[float, float]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for v, w in pairs:
        if w <= 0:
            continue
        num += v * w
        den += w
    if den == 0.0:
        return None
    return num / den


def merge_consecutive_by_speaker_renumber(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent turns with identical speaker_id.
    - concatenate using a single space
    - aggregate metadata:
        * startTime = first
        * endTime   = last
        * duration  = end-start if both present
        * numeric features (mfcc/F0/F1 etc.) = weighted mean (weights by per-turn duration, fallback wordCount, fallback transcript word count)
        * transcript = merged text
        * wordCount  = recomputed from merged text
    - renumber turn_idx to 0..N-1 after merging
    Returns list of dicts:
      {
        "turn_idx": int,
        "turn_text": str,
        "speaker_id": Any,
        "meta": Dict[str, Any]
      }
    """
    if not turns:
        return []

    merged: List[Dict[str, Any]] = []

    curr_spk = turns[0].get("speaker_id")
    curr_parts = [turns[0].get("text", "")]
    curr_metas = [turns[0].get("meta", {})]

    def flush_group(spk: Any, parts: List[str], metas: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_text = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip()).strip()

        # start/end (take first/last available)
        start = None
        end = None
        for m in metas:
            if start is None and _is_number(m.get("startTime")):
                start = float(m["startTime"])
                break
        for m in reversed(metas):
            if end is None and _is_number(m.get("endTime")):
                end = float(m["endTime"])
                break

        duration = None
        if start is not None and end is not None and end >= start:
            duration = end - start

        # weights per original turn
        weights: List[float] = []
        for m in metas:
            w = None
            if _is_number(m.get("duration")):
                w = float(m["duration"])
            elif _is_number(m.get("wordCount")):
                w = float(m["wordCount"])
            else:
                w = float(count_words(m.get("transcript") or m.get("turn_text") or m.get("turnText") or ""))
            weights.append(max(w, 0.0))

        # numeric keys across metas
        numeric_keys = set()
        for m in metas:
            if not isinstance(m, dict):
                continue
            for k, v in m.items():
                if _is_number(v):
                    numeric_keys.add(k)

        agg_meta: Dict[str, Any] = {}
        for k in numeric_keys:
            pairs: List[Tuple[float, float]] = []
            for m, w in zip(metas, weights):
                v = m.get(k) if isinstance(m, dict) else None
                if _is_number(v):
                    pairs.append((float(v), w))
            mean = _weighted_mean(pairs)
            if mean is not None:
                agg_meta[k] = mean

        # override/ensure key metadata
        agg_meta["speaker"] = spk
        agg_meta["speaker_id"] = spk
        if start is not None:
            agg_meta["startTime"] = start
        if end is not None:
            agg_meta["endTime"] = end
        if duration is not None:
            agg_meta["duration"] = duration

        # transcript + wordCount from merged
        agg_meta["transcript"] = merged_text
        agg_meta["wordCount"] = count_words(merged_text)

        return {"speaker_id": spk, "turn_text": merged_text, "meta": agg_meta}

    for t in turns[1:]:
        spk = t.get("speaker_id")
        if spk == curr_spk:
            curr_parts.append(t.get("text", ""))
            curr_metas.append(t.get("meta", {}))
        else:
            merged.append(flush_group(curr_spk, curr_parts, curr_metas))
            curr_spk = spk
            curr_parts = [t.get("text", "")]
            curr_metas = [t.get("meta", {})]

    merged.append(flush_group(curr_spk, curr_parts, curr_metas))

    for i, m in enumerate(merged):
        m["turn_idx"] = i

    return merged


# =========================================================
# [NEW] Enforce strict speaker alternation (ABAB pattern)
# =========================================================
def enforce_speaker_alternation(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter turns to enforce strict speaker alternation (ABAB pattern).
    
    Rules:
    - Keep the first turn unconditionally
    - Then only keep turns where speaker_id differs from the last kept turn
    - If all turns have the same speaker_id, keep all of them (monologue case)
    - Renumber turn_idx to be contiguous 0..N-1 after filtering
    """
    if not turns:
        return []
    
    # Edge case: if all turns are from the same speaker, keep all (monologue)
    unique_speakers = {t.get("speaker_id") for t in turns}
    if len(unique_speakers) <= 1:
        for i, m in enumerate(turns):
            m["turn_idx"] = i
        return turns
    
    result: List[Dict[str, Any]] = [turns[0]]
    last_speaker = turns[0].get("speaker_id")
    
    for t in turns[1:]:
        spk = t.get("speaker_id")
        # Only keep if speaker differs from last kept turn
        if spk != last_speaker:
            result.append(t)
            last_speaker = spk
    
    # Renumber turn_idx to be contiguous 0..N-1
    for i, m in enumerate(result):
        m["turn_idx"] = i
    
    return result


# =========================================================
# Prompt + parsing normalization
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


PROMPT = """You are analyzing the content of different turns in a conversation. Your task is to separate and extract the explicit and implicit information in one turn.

CRITICAL DEFINITIONS:
- Explicit propositions: Direct statements or factual claims clearly expressed in the text of the turn.
- Assumptions: The premises that must hold for the speaker's stance to make sense or be coherent. These assumptions can include categories such as
  - causal assumptions
  - normative assumptions
  - epistemic assumptions
  - beliefs about the audience or world
  - goals
  - beliefs about what counts as knowledge/evidence/trustworthy sources
  - social/affective beliefs (trust, respect, authority, identity, morality) that justify the speaker's stance

TASK:
- Extract a list of propositions and a list of assumptions for the given turn.

RULES:
- Return up to 10 propositions. Prefer fewer, higher-quality items
- Return up to 10 assumptions. Prefer fewer, higher-quality items
- Order each list from most to least salient for the turn's communicative intent
- Each explicit proposition must be an atomic statement with a numeric confidence score
- Each assumption must be an implicit belief, not a paraphrase of explicit propositions
- Do not duplicate content across explicit propositions and implicit assumptions.
- Generate results in JSON only. Do not include commentary, markdown, or extra keys. Double quotes only. No trailing commas.
- If there are no propositions or beliefs for a category, output an empty list.

OUTPUT FORMAT (strict JSON with exactly these keys):
{{
  "explicit_propositions": [
    {{"text": "...", "confidence": 0.95}},
    {{"text": "...", "confidence": 0.90}}
  ],
  "assumptions": [
    {{"text": "...", "confidence": 0.93}},
    {{"text": "...", "confidence": 0.88}}
  ]
}}

TASK:
Extract the list of propositions and list of assumptions for this speaker turn:
"{turn_text}"

"""


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
# Main: process raw/{category} episodes; write to data/{category}
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Process raw/{category}/*.json episodes; run Prompt; save outputs to data/{category}."
    )
    ap.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument(
        "--auto_categories",
        action="store_true",
        help="Discover all category folders under --raw_root. Overrides --categories.",
    )

    ap.add_argument(
        "--num_episodes",
        type=int,
        default=0,
        help="Per category: number of episodes to write. 0 means all.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min_words_per_turn",
        type=int,
        default=0,
        help="Drop merged turns shorter than this many words. Default 0 keeps all turns.",
    )
    ap.add_argument(
        "--max_turns_per_episode",
        type=int,
        default=0,
        help="0 means no cap; else max merged turns to process per episode.",
    )
    ap.add_argument(
        "--required_num_speakers",
        type=int,
        default=0,
        help="Keep only episodes with exactly this many speakers after cleaning. Set 0 to disable.",
    )
    ap.add_argument(
        "--episodes_to_try",
        type=int,
        default=0,
        help="Per category candidate cap before filtering. 0 means all files in that category.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)

    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_base = Path(args.output_root)

    if not raw_root.exists() or not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found or not a directory: {raw_root}")
    out_base.mkdir(parents=True, exist_ok=True)

    categories = discover_category_dirs(raw_root, args.categories, args.auto_categories)
    if not categories:
        log.error("No categories to process under raw_root=%s", raw_root)
        return
    log.info("Categories to process: %s", ", ".join(categories))

    llm = LLMInterface(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size)
    rng = random.Random(args.seed)

    global_manifest: Dict[str, Any] = {
        "raw_root": str(raw_root),
        "output_root": str(out_base),
        "categories": categories,
        "num_episodes_per_category": args.num_episodes,
        "episodes_to_try_per_category": args.episodes_to_try,
        "min_words_per_turn": args.min_words_per_turn,
        "max_turns_per_episode": args.max_turns_per_episode,
        "required_num_speakers": args.required_num_speakers,
        "seed": args.seed,
        "model_name": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "category_manifests": {},
    }

    for category in categories:
        episode_files = discover_episode_files(raw_root, category)
        if not episode_files:
            log.warning("No episode JSON files found for category=%s", category)
            continue

        rng.shuffle(episode_files)
        if args.episodes_to_try and args.episodes_to_try > 0:
            episode_files = episode_files[: min(len(episode_files), args.episodes_to_try)]

        out_root = out_base / category
        parsed_dir = out_root / "parsed"
        raw_dir = out_root / "raw"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "category": category,
            "raw_category_dir": str(raw_root / category),
            "num_episodes_target": args.num_episodes,
            "episodes_to_try": args.episodes_to_try,
            "min_words_per_turn": args.min_words_per_turn,
            "max_turns_per_episode": args.max_turns_per_episode,
            "required_num_speakers": args.required_num_speakers,
            "seed": args.seed,
            "model_name": args.model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "episodes_written": [],
            "episodes_skipped_no_turns": 0,
            "episodes_skipped_speaker_count_mismatch": 0,
        }

        written = 0
        for turns_path in tqdm(episode_files, desc=f"{category}: Episodes processed"):
            if args.num_episodes and args.num_episodes > 0 and written >= args.num_episodes:
                break

            raw_turns = load_episode_turns_raw(turns_path)
            if not raw_turns:
                manifest["episodes_skipped_no_turns"] += 1
                continue

            cleaned_turns = merge_multi_speaker_turns_into_previous(raw_turns)

            if args.required_num_speakers and args.required_num_speakers > 0:
                unique_speakers = {
                    t.get("speaker_id")
                    for t in cleaned_turns
                    if t.get("speaker_id") is not None
                }
                if len(unique_speakers) != args.required_num_speakers:
                    manifest["episodes_skipped_speaker_count_mismatch"] += 1
                    continue

            merged = merge_consecutive_by_speaker_renumber(cleaned_turns)

            if args.min_words_per_turn and args.min_words_per_turn > 0:
                merged = [m for m in merged if count_words(m.get("turn_text", "")) >= args.min_words_per_turn]

            merged = enforce_speaker_alternation(merged)
            if not merged:
                manifest["episodes_skipped_no_turns"] += 1
                continue

            if args.max_turns_per_episode and args.max_turns_per_episode > 0 and len(merged) > args.max_turns_per_episode:
                merged = merged[: args.max_turns_per_episode]

            prompts = [PROMPT.format(turn_text=m["turn_text"]) for m in merged]
            outputs: List[str] = []
            for start in range(0, len(prompts), args.batch_size):
                outputs.extend(llm.generate_batch(prompts[start : start + args.batch_size]))

            parsed_rows = []
            raw_rows = []
            parsed_ok = 0

            episode_id = infer_episode_id(turns_path, raw_turns)
            episode_file_stem = sanitize_filename_stem(turns_path.stem)

            for m, raw_out in zip(merged, outputs):
                norm = parse_and_normalize_llm(raw_out)
                meta = m.get("meta", {}) if isinstance(m.get("meta"), dict) else {}

                row: Dict[str, Any] = {}
                row["category"] = category
                row["episode_id"] = episode_id
                row["turn_file"] = str(turns_path)
                row["turn_idx"] = m["turn_idx"]
                row["speaker_id"] = m["speaker_id"]

                for k, v in meta.items():
                    if k in row:
                        continue
                    row[k] = v

                row["turn_text"] = m["turn_text"]

                if norm is not None:
                    parsed_ok += 1
                    row.update(norm)

                parsed_rows.append(row)
                raw_rows.append({"turn_idx": m["turn_idx"], "turn_text": m["turn_text"], "raw_output": raw_out})

            out_parsed = parsed_dir / f"{episode_file_stem}.json"
            out_raw = raw_dir / f"{episode_file_stem}_raw.json"
            with open(out_parsed, "w", encoding="utf-8") as f:
                json.dump(parsed_rows, f, indent=2, ensure_ascii=False)
            with open(out_raw, "w", encoding="utf-8") as f:
                json.dump(raw_rows, f, indent=2, ensure_ascii=False)

            manifest["episodes_written"].append(
                {
                    "category": category,
                    "episode_id": episode_id,
                    "source_episode_path": str(turns_path),
                    "n_turns_written": len(parsed_rows),
                    "parsed_ok": parsed_ok,
                    "parsed_path": str(out_parsed),
                    "raw_path": str(out_raw),
                }
            )
            written += 1

        manifest_path = out_root / "manifest_by_episode.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        global_manifest["category_manifests"][category] = {
            "episodes_input_considered": len(episode_files),
            "episodes_written": written,
            "manifest_path": str(manifest_path),
        }

        log.info("Category=%s done. Episode files written: %d. Manifest: %s", category, written, manifest_path)

    global_manifest_path = out_base / "manifest_by_category.json"
    with open(global_manifest_path, "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, indent=2, ensure_ascii=False)

    log.info("Done. Category manifest: %s", global_manifest_path)


if __name__ == "__main__":
    main()
