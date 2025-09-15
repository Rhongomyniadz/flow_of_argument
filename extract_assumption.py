import os
import ast
import re
import gc
import json
import gzip
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple, Set

from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-extract")


# -------------------- Episode & turn parsing --------------------
def normalize_output(raw: str) -> List[str]:
    """
    Extract ["key_points_assumed", ...] from messy LLM output.
    Handles prose before/after JSON, multiple JSON blocks, code fences,
    and a stray trailing '}'.
    """

    if not raw:
        return []

    s = raw.strip()

    # Strip code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        s = m.group(1).strip()

    def _from_obj(obj) -> List[str]:
        vals = obj.get("key_points_assumed", [])
        return [v.strip() for v in vals if isinstance(v, str) and v.strip()]

    # Try strict JSON on the whole thing first
    try:
        obj = json.loads(s)
        out = _from_obj(obj)
        if out:
            return out
    except Exception:
        pass

    # Heuristic candidates: after '####' and after 'Answer:'
    candidates: List[str] = []
    if "####" in s:
        candidates.extend([seg.strip() for seg in s.split("####") if seg.strip()])

    ans_idx = s.lower().find("answer:")
    if ans_idx != -1:
        candidates.append(s[ans_idx + len("answer:"):].strip())

    # Always include the full string as a candidate
    candidates.append(s)

    # Balanced-brace scanner to collect all {...} blocks
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
            # Prefer blocks that mention the key
            if '"key_points_assumed"' not in blk and "'key_points_assumed'" not in blk:
                # Still try, but lower priority
                pass
            # Strict JSON
            try:
                obj = json.loads(blk)
                out = _from_obj(obj)
                if out:
                    return out
            except Exception:
                pass
            # Tolerant parse (single quotes, trailing commas)
            try:
                obj = ast.literal_eval(blk)
                if isinstance(obj, dict):
                    out = _from_obj(obj)
                    if out:
                        return out
            except Exception:
                pass

    # Last-chance: turn bullets into sentences
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
    return len(re.findall(r"\w+", text or ""))


# -------------------- Local streamers & path discovery --------------------
def stream_jsonl(path: Path) -> Iterable[Dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for o in obj:
                    if isinstance(o, dict):
                        yield o

def find_episodes_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Data file not found: {p}")
    # auto-discover under CWD
    candidates = [
        Path("data/covid_episodes.jsonl.gz"),
        Path("data/covid_episodes.jsonl")
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Could not locate covid_episodes.jsonl[.gz] under ./data or ./results; "
                            "pass --data_path explicitly.")

def find_turns_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Turns file not found: {p}")
    candidates = [
        Path("data/covid_episodes_turn.jsonl.gz"),
        Path("data/covid_episodes_turn.jsonl")
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Could not locate covid_episodes_turn.jsonl[.gz] under ./data or ./results; "
                            "pass --turns_path explicitly.")


def collect_needed_turns_from_local(mp3_urls: Set[str],
                                    turns_path: Path,
                                    max_per_episode: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Stream the local turns JSONL(.gz) and collect turns only for the target episodes.

    IMPORTANT: We assume the subset-generator has *already* canonicalized and matched on URL
    (e.g., by mapping to SPoRC's raw URL format during selection), and that the original JSON
    written to `covid_episodes_turn.jsonl[.gz]` preserves a canonical per-episode URL field
    identical to the episode's mp3_url we read from `covid_episodes.jsonl[.gz]`.

    Therefore, matching is done by *direct equality* against the episode's mp3_url across
    common candidate URL fields in the turn record. We do NOT attempt any further canonicalization.
    """
    opener = gzip.open if turns_path.suffix == ".gz" else open

    # Direct-equality match set per episode
    keys_by_mp3 = {mp3: {mp3} for mp3 in mp3_urls}

    out: Dict[str, List[Dict]] = {mp3: [] for mp3 in mp3_urls}
    hits_left = set(mp3_urls)

    with opener(turns_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="scan local turns", unit="lines"):
            try:
                rec = json.loads(line)
            except Exception:
                continue

            cand = rec.get("mp3_url")
            if not cand:
                continue

            matched_mp3 = None
            for mp3, keys in keys_by_mp3.items():
                if cand in keys:
                    matched_mp3 = mp3
                    break

            if not matched_mp3:
                continue

            # Extract minimal turn fields
            text = (rec.get("text") or rec.get("turn_text") or "").strip()
            if not text:
                continue
            turn_obj = {
                "text": text,
                "speaker_id": rec.get("speaker_id", rec.get("speaker")),
                "inferred_speaker_name": rec.get("inferred_speaker_name", "NO_INFERRED_SPEAKER"),
                "inferred_speaker_role": rec.get("inferred_speaker_role", "NO_INFERRED_ROLE"),
            }
            out[matched_mp3].append(turn_obj)

            if max_per_episode and len(out[matched_mp3]) >= max_per_episode:
                if matched_mp3 in hits_left:
                    hits_left.remove(matched_mp3)

            # Optional early stop if all found and capped
            if max_per_episode and not hits_left:
                break

    return out


# -------------------- LLM wrapper --------------------

class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7,
        top_p: float = 0.8,
        min_p: float = 0.1,
        repetition_penalty: float = 1.1,
        max_tokens: int = 2048,
        download_dir: str = "/shared/4/models"
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir,
        )
        self.params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        out = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in out]


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="", help="Path to local episodes jsonl(.gz). If empty, auto-discover.")
    ap.add_argument("--turns_path", type=str, default="", help="Path to local turns jsonl(.gz). If empty, auto-discover.")
    ap.add_argument("--min_words", type=int, default=50, help="Min words in a turn to run LLM.")
    ap.add_argument("--sample_n", type=int, default=30, help="Max episodes to process (reservoir).")
    ap.add_argument("--gpu_id", type=int, default=0, help="GPU id for vLLM.")
    args = ap.parse_args()

    # 1) Locate the local episodes & turns files
    try:
        episodes_path = find_episodes_path(args.data_path)
        turns_path = find_turns_path(args.turns_path)
    except FileNotFoundError as e:
        log.error(str(e))
        return

    log.info("Streaming episodes from %s …", str(episodes_path))
    log.info("Using local turns from %s …", str(turns_path))

    # 2) Build reservoir of candidate episodes (2-speaker via embedded turns OR metadata)
    reservoir: List[Dict] = []
    total_seen = 0
    all_sampled_mp3: List[str] = []
    episodes_buffer: List[Tuple[Dict, str]] = []  # tuple: (episode_json, mp3_url)

    for ep in tqdm(stream_jsonl(episodes_path), desc="scan episodes jsonl"):
        mp3 = ep.get("mp3_url")
        if not mp3:
            continue

        total_seen += 1
        if len(reservoir) < args.sample_n:
            reservoir.append(ep)
            episodes_buffer.append((ep, mp3))  # Keep only episode and mp3_url
            all_sampled_mp3.append(mp3)
        else:
            j = random.randint(0, total_seen - 1)
            if j < args.sample_n:
                reservoir[j] = ep
                episodes_buffer[j] = (ep, mp3)  # Keep only episode and mp3_url
                all_sampled_mp3[j] = mp3

    log.info("Reservoir sampled %d episodes (out of %d 2-speaker matches)", len(reservoir), total_seen)
    if not reservoir:
        log.warning("No qualifying episodes found; exiting.")
        return

    # 3) Back-fill turns for sampled episodes that lack embedded turns (from local turns file)
    need_turns_for = {mp3 for (_, mp3) in episodes_buffer}
    turns_by_mp3: Dict[str, List[Dict]] = {}
    if need_turns_for:
        log.info("Back-filling turns for %d sampled episodes from local turns file", len(need_turns_for))
        turns_by_mp3 = collect_needed_turns_from_local(need_turns_for, turns_path)

    # 4) Init LLM
    llm = LLMInterface(gpu_id=args.gpu_id)

    # 5) Process episodes and write per-episode outputs
    out_dir = Path("results/covid")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_out_dir = Path("results/raw/covid")
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    for ep, mp3 in tqdm(episodes_buffer, desc="processing eps"):
        title = ep.get("title")
        label = re.sub(r"[^\w\-]", "_", title)

        # Get turns from collected turns
        tlist = turns_by_mp3.get(mp3, [])
        if not tlist:
            log.warning("No turns found for sampled episode: %s", title)
            continue

        prompts, meta = [], []
        for idx, t in enumerate(tqdm(tlist, desc=f"turns {label}", leave=False)):
            text = t.get("text").strip()
            if count_words(text) < args.min_words:
                continue
            role = t.get("inferred_speaker_role").strip().upper()
            prompts.append(f"""
SYSTEM:
You are an expert at analyzing conversations and surfacing implicit assumptions. You strictly avoid repeating explicit statements and instead infer what must be true for the speaker’s words to make sense. You write concisely and avoid speculation beyond reasonable inference.

TASK:
Analyze the following conversation turn and identify the underlying assumptions. Your job is to:
1) Infer unstated beliefs the speaker holds
2) Surface implicit knowledge the speaker assumes the audience has
3) State hidden premises that support their claims or requests
4) Capture contextual assumptions about the situation/environment

SCOPE & QUANTITY:
- Generate MANY assumptions (aim for 12-20 distinct items when the text permits; otherwise, include as many as are well-supported).
- Each assumption must be specific, non-overlapping, and not a paraphrase of explicit content.

CONFIDENCE & ORDERING:
- For each assumption, estimate a confidence in [0.0, 1.0] where:
  - 0.90-1.00 = highly likely, strongly implied
  - 0.70-0.89 = plausible with moderate support
  - 0.50-0.69 = tentative but defensible
- Sort the list in STRICTLY descending order of confidence (highest first).
- Calibrate confidence based on direct cues, strength/number of supporting hints, and typical conversational pragmatics.

OUTPUT FORMAT (JSON only):
Return a JSON object with one key "key_points_assumed" whose value is a list of objects with fields:
- "assumption": a single complete sentence stating the assumption
- "confidence": a float in [0,1] with two decimal places
Include "evidence_spans": a short phrase or two (from the input) that justifies the inference (no more than 12 words each). If used, keep it minimal.

CONSTRAINTS FOR ASSUMPTIONS:
- Each assumption must be a single, well-formed sentence focused on one idea.
- Do not restate anything explicitly said in the text.
- Prefer concrete, testable claims over vague generalities.
- If an assumption depends on a stronger parent assumption, include both and reflect lower confidence for the child.
- Avoid world knowledge that is not reasonably suggested by the text.

EVALUATION PASS:
After drafting, quickly de-duplicate and tighten wording before output.

INPUT:
{text}
""")
            id_raw = t.get("speaker")
            speaker_id = "-".join(str(x) for x in id_raw)
            meta.append((
                text,
                speaker_id,
                t.get("inferred_speaker_name"),
                t.get("inferred_speaker_role"),
            ))

        if not prompts:
            log.warning("No qualifying turns (min_words=%d) for: %s", args.min_words, title)
            continue

        outputs = llm.generate_batch(prompts)
        records = []
        raw_records = []
        for (text, speaker, name, role), raw_out in zip(meta, outputs):
            records.append({
                "turn_text": text,
                "speaker_id": speaker,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role if role else "NO_INFERRED_ROLE",
                "assumptions": normalize_output(raw_out),
            })
            
            raw_records.append({
                "turn_text": text,
                "speaker_id": speaker,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role if role else "NO_INFERRED_ROLE",
                "raw_output": raw_out,
            })

        with open(out_dir / f"{label}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            
        with open(raw_out_dir / f"{label}.json", "w", encoding="utf-8") as f:
            json.dump(raw_records, f, indent=2, ensure_ascii=False)

        del prompts, meta, outputs, records
        gc.collect()

    log.info("Done. Wrote episode JSONs to %s", str(out_dir))


if __name__ == "__main__":
    main()
