import argparse
import json
import gzip
import re
import hashlib
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable, Tuple

import pandas as pd
from tqdm import tqdm

# If you use vLLM:
from vllm import LLM, SamplingParams


# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-extract-and-normalize")


# =========================================================
# Utilities
# =========================================================
_WORD_RE = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def safe_slug(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.-]+", "_", s)
    return s[:max_len] if s else "untitled"


def episode_key(ep: Dict) -> str:
    """Stable filename key; prefers mp3, falls back to title, then hash."""
    title = (ep.get("epTitle") or ep.get("title") or "").strip()
    mp3 = (ep.get("mp3url") or "").strip()
    if mp3:
        h = hashlib.sha1(mp3.encode("utf-8")).hexdigest()[:10]
        return f"{safe_slug(title, 48)}_{h}" if title else f"ep_{h}"
    if title:
        return safe_slug(title, 64)
    h = hashlib.sha1(json.dumps(ep, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"ep_{h}"


def load_jsonl_gz(path: str) -> List[Dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# ---------------- Balanced JSON extraction ----------------
_CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_json_candidates(txt: str) -> Iterable[str]:
    """Yield possible JSON substrings: fenced blocks first, then brace-balanced spans."""
    if not txt:
        return
    # Prefer ```json fenced``` blocks if present
    for m in _CODE_FENCE_RE.finditer(txt):
        block = m.group(1).strip()
        if block:
            yield block

    # Fall back to scanning for top-level brace-balanced JSON objects
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
            # Strip percent signs or stray chars
            xs = v.strip().replace("%", "")
            x = float(xs)
        else:
            return default
        if x != x:  # NaN
            return default
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def _norm_item(x: Any, default_conf: float = 0.7) -> Optional[Dict[str, Any]]:
    """Normalize an item into {'text': str, 'confidence': float} or None."""
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
    conf_f = _to_float_conf(conf, default_conf)
    return {"text": text, "confidence": conf_f}


def _normalize_list(items: Any, cap: int = 10) -> List[Dict[str, Any]]:
    """Coerce arbitrary list-like to a clean, deduped list of dicts with confidence."""
    out: List[Dict[str, Any]] = []
    seen = set()
    if items is None:
        return out
    if isinstance(items, dict) and "list" in items and isinstance(items["list"], list):
        items = items["list"]
    if not isinstance(items, (list, tuple)):
        cand = _norm_item(items)
        if cand:
            key = cand["text"].strip().casefold()
            if key not in seen:
                seen.add(key)
                out.append(cand)
        return out[:cap]

    for x in items:
        cand = _norm_item(x)
        if not cand:
            continue
        key = cand["text"].strip().casefold()
        if key in seen:
            # Keep the highest confidence on duplicates
            for i, y in enumerate(out):
                if y["text"].strip().casefold() == key and cand["confidence"] > y["confidence"]:
                    out[i] = cand
            continue
        seen.add(key)
        out.append(cand)

    # Sort by confidence desc to approximate salience, truncate to cap
    out.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return out[:cap]


def _dedup_cross_lists(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove entries in secondary that duplicate texts in primary (casefold match)."""
    prim_keys = {p["text"].strip().casefold() for p in primary}
    sec_clean = [s for s in secondary if s["text"].strip().casefold() not in prim_keys]
    return primary, sec_clean


def parse_and_normalize_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the first valid JSON with keys:
      - explicit_propositions: list of {text, confidence}
      - assumptions:           list of {text, confidence}
    If missing, default to [] and normalize everything.
    """
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
            # Remove overlaps (assumptions shouldn't repeat explicit texts)
            exp_n, asm_n = _dedup_cross_lists(exp_n, asm_n)
            return {"explicit_propositions": exp_n, "assumptions": asm_n}
        except Exception:
            continue

    # Nothing parsable
    return None


# =========================================================
# vLLM Wrapper
# =========================================================
class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2,
        temperature: float = 0.6,
        top_p: float = 0.95,
        min_p: float = 0.1,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
        max_tokens: int = 6000,
    ):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
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

# ---------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------
PROMPTS = [

    # ─────────────────────────────────────────────────────────────
    # Prompt 1 — Cognitive Linguistics Baseline
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a cognitive linguistics analyst who specializes in interpreting conversation turns.
Your task is to extract explicit propositions and underlying assumptions with high precision.

CRITICAL DEFINITIONS:
- Explicit propositions: Direct statements or factual claims clearly expressed in the text.
- Assumptions: Deeper belief-level premises (causal/normative/epistemic/audience/goal) that must hold for the speaker's stance to make sense; more general than EP.

RULES:
- Return UP TO 10 items for each list (0-10). Prefer fewer, higher-quality items.
- Order each list from MOST to LEAST salient for the turn's communicative intent.
- Each Explicit propositions must be an atomic statement with a numeric confidence score (0.0-1.0).
- Each Assumptions must be a generalized belief (not a paraphrase of Explicit propositions) with a numeric confidence score (0.0-1.0).
- No duplication across Explicit propositions and Assumptions.
- Strict JSON ONLY. No commentary, markdown, or extra keys. Double quotes only. No trailing commas.
- If none for a category, output an empty list.

TASK:
Given this speaker turn:
"{turn_text}"

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
}}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 2 — Logical / Reasoning Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a reasoning and logic analyst trained to extract propositional structures and infer unstated premises from arguments.
Focus on how ideas follow from each other and what premises support conclusions.

DEFINITIONS:
- Explicit propositions: Direct claims stated in the text.
- Assumptions: Foundational belief-level rules of inference or commitments required for the reasoning to be valid.

RULES:
- Return UP TO 10 items per list, ordered by MOST → LEAST salient for the argument.
- Explicit propositions must be atomic, non-overlapping, each with a confidence score (0.0-1.0).
- Assumptions should be more general than the text and may be conditional/causal, each with a confidence score (0.0-1.0).
- No duplication between Explicit propositions and Assumptions; no paraphrase padding.
- Strict JSON ONLY; no extra keys.

TASK:
Analyze this conversation turn:
"{turn_text}"

OUTPUT FORMAT:
{{
  "explicit_propositions": [
    {{"text": "...", "confidence": 0.94}},
    {{"text": "...", "confidence": 0.90}}
  ],
  "assumptions": [
    {{"text": "If X, then Y.", "confidence": 0.92}},
    {{"text": "Agents act to maximize expected utility under constraints.", "confidence": 0.87}}
  ]
}}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 3 — Social / Pragmatic Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
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
}}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 4 — Causal Reasoning Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a causal inference analyst focusing on how speakers explain why things happen.

DEFINITIONS:
- Explicit propositions: Direct statements of causes/effects.
- Assumptions: Core causal beliefs about mechanisms/dependencies/agency that support the reasoning; more general than EP.

RULES:
- Return UP TO 10 items per list, ordered by MOST → LEAST salient for causal explanation.
- Explicit propositions must be atomic cause/effect claims with confidence (0.0-1.0).
- Assumptions should articulate portable causal beliefs (mechanisms or constraints), each with confidence (0.0-1.0), not surface restatements.
- No duplication between Explicit propositions and Assumptions.
- Strict JSON ONLY; no extra keys.

TASK:
Analyze this turn:
"{turn_text}"

OUTPUT FORMAT:
{{
  "explicit_propositions": [
    {{"text": "...", "confidence": 0.94}},
    {{"text": "...", "confidence": 0.90}}
  ],
  "assumptions": [
    {{"text": "Automation amplifies both efficiencies and upstream human errors.", "confidence": 0.93}},
    {{"text": "Data availability is a primary driver of AI capability gains.", "confidence": 0.88}}
  ]
}}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 5 — Epistemic / Knowledge-State Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are an epistemic reasoning analyst who studies how speakers express certainty, belief, and doubt.

DEFINITIONS:
- Explicit propositions: Direct factual/evaluative claims.
-  Assumptions: General beliefs about what counts as knowledge/evidence/trustworthy sources.

RULES:
- Return UP TO 10 items per list, ordered by MOST → LEAST salient for epistemic stance.
- Explicit propositions must be atomic claims about reality/belief with confidence (0.0-1.0).
- Assumptions must be generalized epistemic commitments (e.g., evidence standards, expertise, testimony), each with confidence (0.0-1.0).
- Avoid duplication between Explicit propositions and  Assumptions; no padding.
- Strict JSON ONLY; no extra keys.

TASK:
Given this text:
"{turn_text}"

OUTPUT FORMAT:
{{
  "explicit_propositions": [
    {{"text": "...", "confidence": 0.95}},
    {{"text": "...", "confidence": 0.90}}
  ],
  "assumptions": [
    {{"text": "Empirical results outweigh anecdotal experience in decision-making.", "confidence": 0.94}},
    {{"text": "Domain expertise should guide interpretation of uncertain data.", "confidence": 0.88}}
  ]
}}"""
]


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run 5 assumption prompts; parse & normalize outputs.")
    ap.add_argument("--data_dir", type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1")
    ap.add_argument("--output_root", type=str, default="results/prompt_camprison")
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    ep_path = Path(args.data_dir) / "episodeLevelData.jsonl.gz"
    turn_path = Path(args.data_dir) / "speakerTurnData.jsonl.gz"

    # --- Load & filter episodes ---
    log.info(f"Loading episode metadata from {ep_path}")
    episodes = load_jsonl_gz(str(ep_path))
    df_ep = pd.DataFrame(episodes)
    log.info("Total episodes loaded: %d", len(df_ep))

    possible_title_cols = ["epTitle", "title"]
    possible_speaker_cols = ["numMainSpeakers", "num_main_speakers"]
    title_col = next((c for c in possible_title_cols if c in df_ep.columns), None)
    speaker_col = next((c for c in possible_speaker_cols if c in df_ep.columns), None)

    if speaker_col:
        df_ep = df_ep[df_ep[speaker_col] == 2]
    log.info(f"Using '{title_col}' for titles and '{speaker_col}' for speaker count filter.")

    # Five target episodes (by title, case-insensitive substring)
    targets = [
        "Mostafa Elbermawy — on Long-Lasting Work, Self-Development, and Why AI Will Not Replace Us",
        "China's Six Front War With America - How To Weaponise COVID-19, 5G & AI",
        "Al and Rishal talk about Rishal's book Grokking AI Algorithms",
        "AI and data-driven adaptation with Colin Shearer",
        "Augmented Intelligence with AI in Manufacturing - Paul Boris",
        "testing episode for prompt engineering"
    ]
    mask = df_ep[title_col].fillna("").apply(lambda x: any(t.lower() in str(x).lower() for t in targets))
    selected_eps = df_ep[mask].to_dict(orient="records")
    log.info("Found %d matching episodes.", len(selected_eps))
    if not selected_eps:
        log.warning("No matching episodes found. Exiting.")
        return

    # Index target mp3s
    target_mp3s = {(ep.get("mp3url") or "").strip() for ep in selected_eps if ep.get("mp3url")}
    log.info(f"Streaming speaker turns for {len(target_mp3s)} target episodes.")

    # --- Stream turns ---
    turn_records: Dict[str, List[Dict]] = {mp3: [] for mp3 in target_mp3s}
    with gzip.open(turn_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            mp3 = (rec.get("mp3url") or "").strip()
            if mp3 not in target_mp3s:
                continue
            text = (rec.get("turnText") or "").strip()
            if not text or count_words(text) < args.min_words:
                continue
            # speaker key is a list in SPoRC turns
            spk = rec.get("speaker")
            speaker_id = spk[0] if isinstance(spk, list) and spk else (spk or None)
            turn_records[mp3].append({
                "turn_text": text,
                "speaker_id": speaker_id,
                "inferred_speaker_name": rec.get("inferredSpeakerName"),
                "inferred_speaker_role": rec.get("inferredSpeakerRole"),
            })
            if i and i % 1_000_000 == 0:
                log.info(f"Scanned {i:,} turn lines...")

    total_kept = sum(len(v) for v in turn_records.values())
    log.info(f"Collected {total_kept:,} valid turns across {len(target_mp3s)} episodes.")

    # --- Run LLM ---
    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    output_root = Path(args.output_root)
    parsed_roots = []
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for prompt_id, tmpl in enumerate(PROMPTS, start=1):
        parsed_dir = output_root / f"prompt{prompt_id}"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        parsed_roots.append(parsed_dir)

        total_turns = 0
        for ep in tqdm(selected_eps, desc=f"Prompt {prompt_id} episodes"):
            ep_mp3 = (ep.get("mp3url") or "").strip()
            turns = turn_records.get(ep_mp3, [])
            if not turns:
                continue

            # Build prompts
            prompts = [tmpl.format(turn_text=t["turn_text"]) for t in turns]

            # Batch generate
            outputs = []
            for start in range(0, len(prompts), args.batch_size):
                chunk = prompts[start:start + args.batch_size]
                outs = llm.generate_batch(chunk)
                outputs.extend(outs)

            # Parse & normalize
            parsed_results: List[Dict] = []
            raw_results: List[Dict] = []
            parsed_ok = 0

            for t, raw_out in zip(turns, outputs):
                norm = parse_and_normalize_llm(raw_out)
                base = {
                    "turn_text": t["turn_text"],
                    "speaker_id": t["speaker_id"],
                    "inferred_speaker_name": t["inferred_speaker_name"],
                    "inferred_speaker_role": t["inferred_speaker_role"],
                }
                if norm is not None:
                    parsed_ok += 1
                    parsed_results.append({**base, **norm})
                else:
                    # If unparsable, still store the base metadata (no EP/A keys)
                    parsed_results.append(base)
                # Raw file: ONLY turn_text + raw_output (per your requirement)
                raw_results.append({
                    "turn_text": t["turn_text"],
                    "raw_output": raw_out
                })

            key = episode_key(ep)

            # Save parsed structured JSON (flattened)
            with open(parsed_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(parsed_results, f, indent=2, ensure_ascii=False)

            # Save raw completions separately (minimal fields)
            with open(raw_dir / f"{key}_prompt{prompt_id}.json", "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)

            gc.collect()
            total_turns += len(outputs)
            log.info(f"{key}: parsed {parsed_ok}/{len(outputs)} successfully.")

        log.info(f"Prompt {prompt_id} complete: {total_turns} turns processed.")
        log.info(f"→ Parsed JSON dir: {parsed_dir}")
    log.info(f"✅ All 5 prompt variants completed. Raw dir: {raw_dir}")


if __name__ == "__main__":
    main()
