#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-safe assumption extraction on local SPoRC JSONL files
using only mp3url + turnText.  Processes five AI-related episodes
with five analytical prompt variants.
"""

import argparse
import json
import gzip
import re
import hashlib
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("local-assumption-prompts-streamed")


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def count_words(text: str) -> int:
    import re as _re
    return len(_re.findall(r"\w+", text or ""))


def safe_slug(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.-]+", "_", s)
    return s[:max_len] if s else "untitled"


def episode_key(ep: Dict) -> str:
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


# ---------------------------------------------------------
# vLLM Wrapper
# ---------------------------------------------------------
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
Your task is to extract propositions and underlying assumptions with high precision and clear differentiation between explicit and implicit meaning.

DEFINITIONS:
- Explicit propositions: Direct statements or factual claims clearly expressed in the text.
- Implicit propositions: Unstated but logically implied meanings or presuppositions necessary to understand the text.
- Assumptions: Deeper underlying beliefs, values, or worldviews that must hold true for the speaker’s reasoning to make sense.

TASK:
Given the following speaker turn:
"{turn_text}"

Follow these steps carefully:
1. Identify **explicit propositions**. Extract only what is overtly said or clearly asserted.
2. Identify **implicit propositions**. Derive these from presuppositions, entailments, or contextual implications.
3. Infer **five distinct assumptions** that logically underlie the speaker’s reasoning or worldview. 
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
3. Infer **five assumptions** that reveal the speaker’s affective or social worldview.
   Each should be a full sentence expressing a psychological or social belief, with a numeric confidence score.

Ensure that each assumption is distinct and reveals the speaker’s underlying attitude or emotion.

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
Your task is to uncover both propositional content and the speaker’s stance toward knowledge and truth.

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


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stream turns by mp3url and run 5 prompt variants")
    ap.add_argument("--data_dir", type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1")
    ap.add_argument("--output_root", type=str, default="results/local_prompts_streamed")
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    args = ap.parse_args()

    ep_path = Path(args.data_dir) / "episodeLevelData.jsonl.gz"
    turn_path = Path(args.data_dir) / "speakerTurnData.jsonl.gz"

    # --- Load and filter episodes ---
    log.info(f"Loading episode metadata from {ep_path}")
    episodes = load_jsonl_gz(str(ep_path))
    df_ep = pd.DataFrame(episodes)
    log.info("Total episodes loaded: %d", len(df_ep))

    possible_title_cols = ["epTitle", "title", "episode_title", "name"]
    possible_speaker_cols = ["numMainSpeakers", "num_main_speakers"]
    title_col = next((c for c in possible_title_cols if c in df_ep.columns), None)
    speaker_col = next((c for c in possible_speaker_cols if c in df_ep.columns), None)

    if speaker_col:
        df_ep = df_ep[df_ep[speaker_col] == 2]
    log.info(f"Using '{title_col}' for titles and '{speaker_col}' for speaker count filter.")

    # --- Target episode titles ---
    targets = [
        "Mostafa Elbermawy — on Long-Lasting Work, Self-Development, and Why AI Will Not Replace Us",
        "China's Six Front War With America - How To Weaponise COVID-19, 5G & AI",
        "Al and Rishal talk about Rishal’s book Grokking AI Algorithms",
        "AI and data-driven adaptation with Colin Shearer",
        "Augmented Intelligence with AI in Manufacturing - Paul Boris"
    ]
    mask = df_ep[title_col].fillna("").apply(lambda x: any(t.lower() in str(x).lower() for t in targets))
    selected_eps = df_ep[mask].to_dict(orient="records")
    log.info("Found %d matching episodes.", len(selected_eps))
    if not selected_eps:
        return

    # --- Build mp3url index for streaming turn filtering ---
    target_mp3s = {(ep.get("mp3url") or "").strip() for ep in selected_eps if ep.get("mp3url")}
    log.info(f"Streaming turn file and collecting turns for {len(target_mp3s)} target episodes.")

    turn_records: Dict[str, List[str]] = {mp3: [] for mp3 in target_mp3s}

    # --- Stream turns directly from file ---
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

            turn_records[mp3].append(text)

            if i % 1_000_000 == 0 and i > 0:
                log.info(f"Scanned {i:,} lines...")

    total_kept = sum(len(v) for v in turn_records.values())
    log.info(f"Collected {total_kept:,} turns total across {len(target_mp3s)} episodes.")

    # --- Run prompts ---
    llm = LLMInterface(model_name=args.model_name)

    for prompt_id, tmpl in enumerate(PROMPTS, start=1):
        out_dir = Path(args.output_root) / f"prompt{prompt_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        total_turns = 0

        for ep in tqdm(selected_eps, desc=f"Prompt {prompt_id} episodes"):
            ep_mp3 = (ep.get("mp3url") or "").strip()
            if ep_mp3 not in turn_records:
                continue
            turns = turn_records[ep_mp3]
            if not turns:
                continue

            prompts = [tmpl.format(turn_text=t) for t in turns]
            outputs = []
            for start in range(0, len(prompts), args.batch_size):
                chunk = prompts[start:start + args.batch_size]
                outs = llm.generate_batch(chunk)
                outputs.extend(outs)

            total_turns += len(outputs)
            results = [{"turn_text": t, "llm_output": o} for t, o in zip(turns, outputs)]

            key = episode_key(ep)
            with open(out_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            gc.collect()

        log.info(f"Prompt {prompt_id} complete: {total_turns} turns processed. Saved to {out_dir}")

    log.info("✅ All 5 prompt variants completed memory-safely.")


if __name__ == "__main__":
    main()

