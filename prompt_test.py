import argparse
import json
import logging
import re
import gc
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prompt-experiments")

# -------------------- Helpers --------------------

def count_words(text: str) -> int:
    import re as _re
    return len(_re.findall(r"\w+", text or ""))

def safe_slug(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.-]+", "_", s)
    return s[:max_len] if s else "untitled"

def episode_key(ep: Dict) -> str:
    mp3 = (ep.get("mp3_url") or "").strip()
    title = (ep.get("title") or "").strip()
    if mp3:
        h = hashlib.sha1(mp3.encode("utf-8")).hexdigest()[:10]
        return f"{safe_slug(title, 48)}_{h}" if title else f"ep_{h}"
    if title:
        return safe_slug(title, 64)
    h = hashlib.sha1(json.dumps(ep, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"ep_{h}"

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

# -------------------- vLLM Wrapper --------------------

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
        max_tokens: int = 6000,
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
        outs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in outs]

# -------------------- Prompt Variants --------------------

PROMPTS = [
    # Prompt 1 — baseline (confidence + explicit/implicit)
    """SYSTEM:
You are a cognitive linguistics analyst. Identify explicit and implicit propositions and infer five assumptions with confidence scores.

TASK:
Given this speaker turn:
"{turn_text}"

Return JSON:
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
     {"text": "...", "confidence": 0.9}, {"text": "...", "confidence": 0.7}
  ]
}""",

    # Prompt 2 — focus on reasoning chains
    """SYSTEM:
You are a reasoning analyst. Identify logical propositions and implicit premises leading to the speaker’s conclusion.

Given this speaker turn:
"{turn_text}"

Output JSON with:
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.88}, ...]
}""",

    # Prompt 3 — focus on emotional or social assumptions
    """SYSTEM:
You are a pragmatics expert focusing on social and affective meaning.
For each turn, identify explicit and implicit propositions plus five assumptions reflecting beliefs, emotions, or social perspectives.

Text:
"{turn_text}"

Return JSON:
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.91}, ...]
}""",

    # Prompt 4 — focus on causal reasoning
    """SYSTEM:
You are a causal inference analyst. Identify propositions that imply cause–effect relationships and infer underlying assumptions with confidence.

Speaker turn:
"{turn_text}"

Return JSON:
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.85}, ...]
}""",

    # Prompt 5 — focus on epistemic stance (knowledge, belief, certainty)
    """SYSTEM:
You are an epistemic reasoning analyst. Identify explicit and implicit propositions and infer assumptions revealing how certain, doubtful, or confident the speaker is.

Text:
"{turn_text}"

Return JSON:
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.9}, ...]
}"""
]

# -------------------- Experiment Runner --------------------

def run_prompt_variant(prompt_template: str, prompt_id: int, args, episodes, llm):
    out_dir = Path(args.output_root) / f"prompt{prompt_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_turns = 0

    for ep in tqdm(episodes, desc=f"Prompt {prompt_id} episodes"):
        ep_raw = episode_to_raw(ep)
        if not ep_raw:
            continue

        key = episode_key(ep_raw)
        turns = episode_turns(ep)
        if not turns:
            continue

        prompts, meta = [], []
        for t in turns:
            text = t.get("text", "").strip()
            if count_words(text) < args.min_words:
                continue
            prompts.append(prompt_template.format(turn_text=text))
            meta.append(t)

        if not prompts:
            continue

        outputs = []
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start:start + args.batch_size]
            outs = llm.generate_batch(chunk)
            outputs.extend(outs)

        total_turns += len(outputs)
        results = []
        for t, raw in zip(meta, outputs):
            results.append({
                "turn_text": t.get("text"),
                "speaker_id": t.get("speaker_id"),
                "inferred_speaker_name": t.get("inferred_speaker_name"),
                "inferred_speaker_role": t.get("inferred_speaker_role"),
                "llm_output": raw,
            })

        with open(out_dir / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        gc.collect()

    log.info("Prompt %d done. Episodes: %d | Turns: %d | Saved to %s",
             prompt_id, len(episodes), total_turns, str(out_dir))

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Run 5 prompt variants for 5 SPoRC episodes")
    ap.add_argument("--sporc_dir", type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1")
    ap.add_argument("--output_root", type=str, default="results/test_prompt")
    ap.add_argument("--episodes_n", type=int, default=5)
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    args = ap.parse_args()

    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()
    episodes = sporc.search_episodes(min_speakers=2, max_speakers=2)
    episodes = episodes[:args.episodes_n]
    log.info("Loaded %d two-speaker episodes for testing", len(episodes))

    llm = LLMInterface(model_name=args.model_name)

    for i, p in enumerate(PROMPTS, start=1):
        run_prompt_variant(p, i, args, episodes, llm)

    log.info("✅ All 5 prompt variants completed.")


if __name__ == "__main__":
    main()