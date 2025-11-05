import argparse
import json
import logging
import re
import gc
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams
from sporc import SPORCDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai-assumption-prompts")

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

    # ─────────────────────────────────────────────────────────────
    # Prompt 1 — Cognitive Linguistics Baseline
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a cognitive linguistics analyst who specializes in interpreting conversation turns. 
Your task is to extract propositions and underlying assumptions with high precision and clear differentiation between explicit and implicit meaning.

DEFINITIONS:
- Explicit propositions: Direct statements or factual claims clearly expressed in the text.
- Implicit propositions: Unstated but logically implied meanings or presuppositions necessary to understand the text.
- Assumptions: Deeper underlying beliefs, values, or worldviews that must hold true for the speaker's reasoning to make sense.

TASK:
Given the following speaker turn:
"{turn_text}"

Follow these steps carefully:
1. Identify **explicit propositions**. Extract only what is overtly said or clearly asserted.
2. Identify **implicit propositions**. Derive these from presuppositions, entailments, or contextual implications.
3. Infer **five distinct assumptions** that logically underlie the speaker's reasoning or worldview. 
   Each assumption must:
   - Be phrased as a single, clear sentence.
   - Not restate explicit content.
   - Be specific and conceptually rich.
   - Include a numeric confidence score between 0.0 and 1.0 indicating your certainty.

OUTPUT FORMAT (strict JSON):
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
    {"text": "...", "confidence": 0.93},
    {"text": "...", "confidence": 0.88},
    {"text": "...", "confidence": 0.81},
    {"text": "...", "confidence": 0.76},
    {"text": "...", "confidence": 0.71}
  ]
}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 2 — Logical / Reasoning Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a reasoning and logic analyst trained to extract propositional structures and infer unstated premises from arguments. 
Focus on how ideas follow from each other and what premises support the conclusions.

DEFINITIONS:
- Explicit propositions: Statements of fact or opinion directly expressed by the speaker.
- Implicit propositions: Logical premises or presuppositions implied by the explicit statements.
- Assumptions: Foundational beliefs or rules of inference the speaker must accept for their reasoning to hold.

TASK:
Analyze this conversation turn:
"{turn_text}"

Perform:
1. Identify **explicit propositions** forming the visible argument.
2. Identify **implicit propositions** that connect or justify the explicit claims.
3. Infer **five logical assumptions**, phrased as conditional or causal statements, each with a numeric confidence score.

Focus on logical coherence — what must be true for the argument to be internally valid.

OUTPUT FORMAT (strict JSON):
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
    {"text": "If X, then Y", "confidence": 0.92},
    {"text": "People act rationally when given incentives.", "confidence": 0.87},
    ...
  ]
}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 3 — Social / Pragmatic Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a pragmatics and social cognition expert. 
Your goal is to interpret the social, emotional, and interpersonal dimensions of a conversation turn.

DEFINITIONS:
- Explicit propositions: Direct statements, claims, or descriptions.
- Implicit propositions: Presuppositions or conversational implicatures revealing emotional or relational subtext.
- Assumptions: Deeper social or affective beliefs (e.g., about trust, respect, authority, morality, or identity).

TASK:
Given the following text:
"{turn_text}"

Perform the following:
1. Extract **explicit propositions** that describe observable statements.
2. Extract **implicit propositions** that convey emotional tone, interpersonal stance, or social context.
3. Infer **five assumptions** that reveal the speaker's affective or social worldview.
   Each should be a full sentence expressing a psychological or social belief, with a numeric confidence score.

Ensure that each assumption is distinct and reveals the speaker's underlying attitude or emotion.

OUTPUT FORMAT (strict JSON):
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
    {"text": "People who fail to adapt are personally responsible for their struggles.", "confidence": 0.9},
    {"text": "Hard work defines personal worth.", "confidence": 0.88},
    ...
  ]
}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 4 — Causal Reasoning Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are a causal inference analyst focusing on how speakers explain why things happen. 
Your goal is to detect explicit and implicit cause–effect relationships and infer underlying causal assumptions.

DEFINITIONS:
- Explicit propositions: Statements that describe observable causes or effects directly.
- Implicit propositions: Unstated causal connections or enabling conditions implied by the text.
- Assumptions: Core causal beliefs about how the world works — mechanisms, dependencies, or agency — that support the reasoning.

TASK:
Analyze this turn:
"{turn_text}"

Steps:
1. Identify **explicit propositions** that describe or assert cause–effect relationships.
2. Identify **implicit propositions** that link events or conditions causally.
3. Infer **five causal assumptions** about how or why outcomes occur, each including a numeric confidence score (0.0–1.0).

Each assumption should be specific, mechanistic, and avoid repeating surface-level content.

OUTPUT FORMAT (strict JSON):
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
    {"text": "Technological change accelerates when data becomes abundant.", "confidence": 0.93},
    {"text": "Human errors in decision systems propagate through automation.", "confidence": 0.86},
    ...
  ]
}""",

    # ─────────────────────────────────────────────────────────────
    # Prompt 5 — Epistemic / Knowledge-State Analyst
    # ─────────────────────────────────────────────────────────────
    """SYSTEM:
You are an epistemic reasoning analyst who studies how speakers express certainty, belief, and doubt. 
Your task is to uncover both propositional content and the speaker's stance toward knowledge and truth.

DEFINITIONS:
- Explicit propositions: Direct factual or evaluative statements.
- Implicit propositions: Unstated presuppositions or epistemic attitudes implied by tone or framing.
- Assumptions: Foundational epistemic beliefs about what counts as knowledge, evidence, or truth for the speaker.

TASK:
Given this text:
"{turn_text}"

Perform the following:
1. Extract **explicit propositions** that convey claims about reality or belief.
2. Extract **implicit propositions** that reveal epistemic stance (certainty, doubt, authority, etc.).
3. Infer **five epistemic assumptions** reflecting how the speaker understands or trusts knowledge sources.
   Each assumption must include a numeric confidence score.

OUTPUT FORMAT (strict JSON):
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
    {"text": "Empirical observation is more reliable than intuition.", "confidence": 0.94},
    {"text": "Expertise should guide decision-making.", "confidence": 0.88},
    ...
  ]
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
    ap = argparse.ArgumentParser(description="Run 5 prompt variants for 5 AI-related SPoRC episodes")
    ap.add_argument("--sporc_dir", type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1")
    ap.add_argument("--output_root", type=str, default="results/test_prompt")
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    args = ap.parse_args()

    # Load SPoRC
    sporc = SPORCDataset(local_data_dir=args.sporc_dir, streaming=True)
    sporc.load_podcast_subset()
    all_eps = sporc.search_episodes(min_speakers=2, max_speakers=2)

    target_titles = [
        "Mostafa Elbermawy — on Long-Lasting Work, Self-Development, and Why AI Will Not Replace Us",
        "China's Six Front War With America - How To Weaponise COVID-19, 5G & AI",
        "Al and Rishal talk about Rishal's book Grokking AI Algorithms",
        "AI and data-driven adaptation with Colin Shearer",
        "Augmented Intelligence with AI in Manufacturing - Paul Boris"
    ]

    # Find matching episodes
    selected = []
    for ep in all_eps:
        title = getattr(ep, "title", "") or ""
        if any(t.lower() in title.lower() for t in target_titles):
            selected.append(ep)
        if len(selected) == len(target_titles):
            break

    log.info("Found %d target episodes for AI analysis.", len(selected))
    for e in selected:
        log.info("Matched: %s", getattr(e, "title", ""))

    if not selected:
        log.warning("No matching episodes found! Check title spellings.")
        return

    llm = LLMInterface(model_name=args.model_name)
    for i, p in enumerate(PROMPTS, start=1):
        run_prompt_variant(p, i, args, selected, llm)

    log.info("✅ All 5 prompt variants completed on the 5 AI episodes.")


if __name__ == "__main__":
    main()