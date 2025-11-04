#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import re
import gc
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

from tqdm import tqdm
from vllm import LLM, SamplingParams
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-full-run")

# -------------------- Output parsing helpers --------------------

def normalize_output(raw: str) -> List[str]:
    """Extract ["key_points_assumed", ...] from messy LLM output."""
    import ast, json as _json, re as _re
    if not raw:
        return []
    s = raw.strip()

    # Strip code fences
    m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=_re.S)
    if m:
        s = m.group(1).strip()

    def _from_obj(obj) -> List[str]:
        vals = obj.get("key_points_assumed", [])
        return [v.strip() for v in vals if isinstance(v, str) and v.strip()]

    try:
        obj = _json.loads(s)
        out = _from_obj(obj)
        if out:
            return out
    except Exception:
        pass

    candidates = [s]
    if "####" in s:
        candidates.extend([seg.strip() for seg in s.split("####") if seg.strip()])
    ans_idx = s.lower().find("answer:")
    if ans_idx != -1:
        candidates.append(s[ans_idx + len("answer:"):].strip())

    def _json_blocks(text: str) -> List[str]:
        blocks, stack, start = [], 0, -1
        for i, ch in enumerate(text):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start != -1:
                        blocks.append(text[start:i+1])
                        start = -1
        return blocks

    for cand in candidates:
        for blk in _json_blocks(cand):
            try:
                obj = _json.loads(blk)
                out = _from_obj(obj)
                if out:
                    return out
            except Exception:
                pass
            try:
                obj = ast.literal_eval(blk)
                if isinstance(obj, dict):
                    out = _from_obj(obj)
                    if out:
                        return out
            except Exception:
                pass

    bullets = re.findall(r"^\s*[-*]\s+(.*\S)\s*$", s, flags=re.M)
    if bullets:
        out = []
        for b in bullets[:8]:
            x = b.strip()
            if not x.endswith("."):
                x += "."
            out.append(x)
        return out
    return []

def count_words(text: str) -> int:
    import re as _re
    return len(_re.findall(r"\w+", text or ""))

def safe_slug(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.-]+", "_", s)
    return s[:max_len] if s else "untitled"

def episode_key(ep: Dict) -> str:
    """Build a stable per-episode key for file naming."""
    mp3 = (ep.get("mp3_url") or "").strip()
    title = (ep.get("title") or "").strip()
    if mp3:
        h = hashlib.sha1(mp3.encode("utf-8")).hexdigest()[:10]
        return f"{safe_slug(title, 48)}_{h}" if title else f"ep_{h}"
    if title:
        return safe_slug(title, 64)
    h = hashlib.sha1(json.dumps(ep, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"ep_{h}"

# -------------------- vLLM interface --------------------

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
        download_dir: Optional[str] = None,
        max_tokens: int = 10000,
    ):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=download_dir,
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
        if not prompts:
            return []
        outs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

# -------------------- Prompt builder --------------------

PROMPT_TMPL = """SYSTEM:
You are an expert at analyzing conversations and surfacing implicit assumptions. You strictly avoid repeating explicit statements and instead infer what must be true for the speaker’s words to make sense. You write concisely and avoid speculation beyond reasonable inference.

TASK:
Analyze the following conversation turn and identify the underlying assumptions. Your job is to:
- Infer unstated beliefs the speaker holds
- Surface implicit knowledge the speaker assumes the audience has
- Capture contextual assumptions about the situation or environment
- Each assumption must be a single, well-formed sentence focused on one idea
- Do not restate anything explicitly said in the text
- Generate at least 10 assumptions
- Each assumption must be specific, non-overlapping, and not a paraphrase of explicit content
- Sort the list in STRICTLY descending order of confidence with the highest first.

Speaker turn text:
{turn_text}

OUTPUT FORMAT:
Return a JSON object with one key "key_points_assumed" containing a list of clear, specific assumptions.

Print your output in JSON format.
"""

def build_prompts(turns: List[Dict], min_words: int) -> Tuple[List[str], List[Tuple[Dict, int]]]:
    prompts, meta = [], []
    for idx, t in enumerate(turns):
        text = (t.get("text") or t.get("turn_text") or "").strip()
        if count_words(text) < min_words:
            continue
        prompts.append(PROMPT_TMPL.format(turn_text=text))
        meta.append((t, idx))
    return prompts, meta

# -------------------- SPoRC helpers --------------------

def iter_dialogue_episodes(sporc: SPORCDataset, min_speakers: int, max_speakers: int):
    episodes = sporc.search_episodes(min_speakers=min_speakers, max_speakers=max_speakers)
    for ep in episodes:
        yield ep

def episode_to_raw(ep) -> Optional[Dict]:
    for name in ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]:
        if hasattr(ep, name):
            val = getattr(ep, name)
            if isinstance(val, dict):
                return val
    for meth in ["to_dict", "as_dict", "dict"]:
        fn = getattr(ep, meth, None)
        if callable(fn):
            try:
                v = fn()
            except Exception:
                continue
            if isinstance(v, dict):
                return v
    return None

def episode_turns(ep) -> List[Dict]:
    try:
        turns = ep.get_all_turns()
    except Exception as e:
        log.warning("Failed to load turns for episode: %s", e)
        return []

    out = []
    for t in turns:
        rec = None
        for name in ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]:
            if hasattr(t, name) and isinstance(getattr(t, name), dict):
                rec = getattr(t, name)
                break
        if rec is None:
            for meth in ["to_dict", "as_dict", "dict"]:
                fn = getattr(t, meth, None)
                if callable(fn):
                    try:
                        v = fn()
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        rec = v
                        break
        if rec is None:
            continue

        text = (rec.get("text") or rec.get("turn_text") or "").strip()
        if not text:
            continue

        out.append({
            "text": text,
            "speaker_id": rec.get("speaker_id", rec.get("speaker")),
            "inferred_speaker_name": rec.get("inferred_speaker_name", "NO_INFERRED_SPEAKER"),
            "inferred_speaker_role": rec.get("inferred_speaker_role", "NO_INFERRED_ROLE"),
        })
    return out

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Full-dataset assumption extraction on 2-speaker SPoRC dialogues")
    ap.add_argument("--sporc_dir", type=str, required=True, 
                    default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/",
                    help="Local SPORC data directory")
    ap.add_argument("--output_dir", type=str, default="results/all_dialogues",
                    help="Directory for per-episode JSON outputs")
    ap.add_argument("--min_words", type=int, default=50, help="Min words in a turn to run LLM")
    ap.add_argument("--episodes_per_shard", type=int, default=1000,
                    help="Episodes processed per shard before forcing GC")
    ap.add_argument("--batch_size", type=int, default=16, help="LLM batch size for turns within an episode")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--download_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Init SPoRC
    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()
    episodes = list(iter_dialogue_episodes(sporc, min_speakers=2, max_speakers=2))
    log.info("Found %d episodes with exactly two speakers.", len(episodes))

    # Init LLM
    llm = LLMInterface(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.download_dir,
    )

    # Dataset counters
    total_podcasts: Set[str] = set()
    total_episodes = 0
    total_turns = 0

    # Process episodes
    processed = 0
    for idx, ep in enumerate(tqdm(episodes, desc="episodes")):
        ep_raw = episode_to_raw(ep)
        if not ep_raw:
            continue

        key = episode_key(ep_raw)
        out_json = out_dir / f"{key}.json"
        raw_json = raw_dir / f"{key}.json"

        if out_json.exists() and raw_json.exists():
            continue

        tlist = episode_turns(ep)
        if not tlist:
            continue

        # Update podcast/episode counters
        show_id = ep_raw.get("podcast_id") or ep_raw.get("show_id") or ep_raw.get("collection_id")
        if show_id:
            total_podcasts.add(str(show_id))
        total_episodes += 1

        # Count qualifying turns
        eligible_turns = [t for t in tlist if count_words(t.get("text", "")) >= args.min_words]
        total_turns += len(eligible_turns)

        prompts, meta = build_prompts(tlist, min_words=args.min_words)
        if not prompts:
            continue

        final_records: List[Dict] = []
        raw_records: List[Dict]  = []

        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start:start + args.batch_size]
            meta_chunk = meta[start:start + args.batch_size]
            try:
                outputs = llm.generate_batch(chunk)
            except Exception as e:
                log.warning("LLM batch failed (episode=%s): %s", key, e)
                outputs = [""] * len(chunk)

            for (turn_rec, _), raw_out in zip(meta_chunk, outputs):
                role = turn_rec.get("inferred_speaker_role") or "NO_INFERRED_ROLE"
                name = turn_rec.get("inferred_speaker_name")
                speaker = turn_rec.get("speaker_id")
                text = turn_rec.get("text", "")

                final_records.append({
                    "turn_text": text,
                    "speaker_id": speaker,
                    "inferred_speaker_name": name,
                    "inferred_speaker_role": role,
                    "assumptions": normalize_output(raw_out),
                })
                raw_records.append({
                    "turn_text": text,
                    "speaker_id": speaker,
                    "inferred_speaker_name": name,
                    "inferred_speaker_role": role,
                    "raw_output": raw_out,
                })

        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(final_records, f, indent=2, ensure_ascii=False)
            with open(raw_json, "w", encoding="utf-8") as f:
                json.dump(raw_records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error("Failed to write outputs for episode %s: %s", key, e)

        processed += 1
        if processed % args.episodes_per_shard == 0:
            log.info("Shard boundary reached at %d episodes. Forcing GC.", processed)
            gc.collect()

    # --- Summary logging ---
    log.info(
        "SUMMARY → Podcasts: %d | Episodes: %d | Turns (≥%d words): %d",
        len(total_podcasts),
        total_episodes,
        args.min_words,
        total_turns,
    )

    log.info("Done. Wrote outputs to %s (and raw/). Episodes processed: %d", str(out_dir), processed)

if __name__ == "__main__":
    main()
