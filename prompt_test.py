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
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("local-assumption-prompts-v2")

# ---------------------------------------------------------
# Utility functions
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
    mp3 = (ep.get("mp3url") or ep.get("mp3_url") or "").strip()
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
# LLM Wrapper (Qwen3-30B)
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
# Prompts (enhanced 30B optimized)
# ---------------------------------------------------------
PROMPTS = [
    # 1. Cognitive linguistics baseline
    """SYSTEM:
You are a cognitive linguistics analyst who interprets conversation turns.
Extract propositions and underlying assumptions with clear separation between explicit and implicit meaning.

DEFINITIONS:
- Explicit propositions: Direct statements clearly expressed in the text.
- Implicit propositions: Meanings implied but not stated directly.
- Assumptions: Deeper beliefs or worldviews that must hold for the reasoning to make sense.

TASK:
Analyze this speaker turn:
"{turn_text}"

OUTPUT (strict JSON):
{
  "explicit_propositions": ["...", "..."],
  "implicit_propositions": ["...", "..."],
  "assumptions": [
    {"text": "...", "confidence": 0.92},
    {"text": "...", "confidence": 0.87},
    {"text": "...", "confidence": 0.82},
    {"text": "...", "confidence": 0.77},
    {"text": "...", "confidence": 0.73}
  ]
}""",

    # 2. Logical reasoning
    """SYSTEM:
You are a reasoning analyst. Identify logical propositions and hidden premises that support the speaker’s argument.

DEFINITIONS:
- Explicit propositions: Directly asserted statements.
- Implicit propositions: Unstated premises connecting explicit claims.
- Assumptions: Foundational logical rules or beliefs enabling the reasoning.

TASK:
Analyze this turn:
"{turn_text}"

OUTPUT (strict JSON):
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.9}, ...]
}""",

    # 3. Pragmatic / social meaning
    """SYSTEM:
You are a pragmatics expert focusing on emotional and social meaning.
Identify how the speaker’s words reveal beliefs, emotions, or social stance.

DEFINITIONS:
- Explicit propositions: Direct verbal content.
- Implicit propositions: Emotional or social implications.
- Assumptions: Underlying social or affective beliefs implied by the turn.

TASK:
Analyze this text:
"{turn_text}"

OUTPUT (strict JSON):
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.91}, ...]
}""",

    # 4. Causal inference
    """SYSTEM:
You are a causal inference analyst. Identify cause–effect statements and deeper causal assumptions.

DEFINITIONS:
- Explicit propositions: Direct cause–effect statements.
- Implicit propositions: Implied causal links.
- Assumptions: Underlying causal rules or mechanisms implied by reasoning.

TASK:
Speaker turn:
"{turn_text}"

OUTPUT (strict JSON):
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.85}, ...]
}""",

    # 5. Epistemic reasoning
    """SYSTEM:
You are an epistemic reasoning analyst who studies certainty and belief.
Identify how the speaker expresses knowledge, confidence, or doubt.

DEFINITIONS:
- Explicit propositions: Factual or belief statements.
- Implicit propositions: Presuppositions about truth or authority.
- Assumptions: Beliefs about what counts as valid knowledge or evidence.

TASK:
Analyze this text:
"{turn_text}"

OUTPUT (strict JSON):
{
  "explicit_propositions": [...],
  "implicit_propositions": [...],
  "assumptions": [{"text": "...", "confidence": 0.9}, ...]
}"""
]

# ---------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run 5 prompt variants on local SPoRC JSONL files")
    ap.add_argument("--data_dir", type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1")
    ap.add_argument("--output_root", type=str, default="results/prompts")
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    args = ap.parse_args()

    ep_path = Path(args.data_dir) / "episodeLevelData.jsonl.gz"
    turn_path = Path(args.data_dir) / "speakerTurnData.jsonl.gz"

    log.info(f"Loading episode metadata from {ep_path}")
    episodes = load_jsonl_gz(str(ep_path))
    df_ep = pd.DataFrame(episodes)
    log.info("Total episodes loaded: %d", len(df_ep))

    # Detect title, speaker, and mp3 fields automatically
    possible_title_cols = ["epTitle", "title", "episode_title", "name"]
    possible_speaker_cols = ["numMainSpeakers", "num_main_speakers", "numSpeakers"]
    possible_mp3_cols = ["mp3url", "mp3_url", "audio_url"]

    title_col = next((c for c in possible_title_cols if c in df_ep.columns), None)
    speaker_col = next((c for c in possible_speaker_cols if c in df_ep.columns), None)
    mp3_col = next((c for c in possible_mp3_cols if c in df_ep.columns), None)

    if not title_col:
        log.error(f"No title-like column found. Available columns: {list(df_ep.columns)[:30]}")
        return
    if speaker_col:
        df_ep = df_ep[df_ep[speaker_col] == 2]
    log.info(f"Using '{title_col}' for titles and '{speaker_col}' for speaker count.")

    # Select five AI-related episodes
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
        log.warning("No matching episodes found.")
        return

    # Load turns file
    log.info(f"Loading speaker turns from {turn_path}")
    turns = load_jsonl_gz(str(turn_path))
    df_turns = pd.DataFrame(turns)
    log.info("Total turns loaded: %d", len(df_turns))

    # Detect correct turn text and mp3 join keys
    possible_text_cols = ["turnText", "text", "utterance", "turn_text"]
    text_col = next((c for c in possible_text_cols if c in df_turns.columns), None)
    turn_mp3_col = next((c for c in possible_mp3_cols if c in df_turns.columns), None)

    if not text_col or not turn_mp3_col:
        log.error(f"Missing text or mp3 column in turns. Columns: {list(df_turns.columns)[:30]}")
        return
    log.info(f"Using '{text_col}' for text and '{turn_mp3_col}' for joining turns with episodes.")

    llm = LLMInterface(model_name=args.model_name)

    # Process each prompt variant
    for prompt_id, tmpl in enumerate(PROMPTS, start=1):
        out_dir = Path(args.output_root) / f"prompt{prompt_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        total_turns = 0
        for ep in tqdm(selected_eps, desc=f"Prompt {prompt_id} episodes"):
            ep_mp3 = (ep.get(mp3_col) or "").strip()
            if not ep_mp3:
                continue

            ep_turns = df_turns[df_turns[turn_mp3_col] == ep_mp3]
            if ep_turns.empty:
                continue

            prompts, meta = [], []
            for _, row in ep_turns.iterrows():
                text = (row.get(text_col) or "").strip()
                if count_words(text) < args.min_words:
                    continue
                prompts.append(tmpl.format(turn_text=text))
                meta.append({
                    "text": text,
                    "speaker_id": row.get("speaker", []),
                    "inferred_speaker_name": row.get("inferredSpeakerName", "N/A"),
                    "inferred_speaker_role": row.get("inferredSpeakerRole", "N/A")
                })

            if not prompts:
                continue

            outputs = []
            for start in range(0, len(prompts), args.batch_size):
                chunk = prompts[start:start + args.batch_size]
                outs = llm.generate_batch(chunk)
                outputs.extend(outs)

            total_turns += len(outputs)
            results = []
            for m, raw in zip(meta, outputs):
                results.append({**m, "llm_output": raw})

            key = episode_key(ep)
            with open(out_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            gc.collect()

        log.info("Prompt %d complete: %d turns processed. Saved to %s",
                 prompt_id, total_turns, str(out_dir))

    log.info("✅ All 5 prompt variants completed on local JSONL data.")

if __name__ == "__main__":
    main()