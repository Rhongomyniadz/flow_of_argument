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
log = logging.getLogger("local-assumption-prompts")


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
    title = (ep.get("title") or "").strip()
    mp3 = (ep.get("mp3_url") or "").strip()
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
# Prompts (enhanced 30B-optimized)
# ---------------------------------------------------------
PROMPTS = [ ...  # ← insert your detailed five prompts from previous message here
]


# ---------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run 5 prompt variants directly on local SPoRC JSONL files")
    ap.add_argument("--data_dir", type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1")
    ap.add_argument("--output_root", type=str, default="results")
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

    # Filter for 2-speaker episodes only
    if "num_main_speakers" in df_ep.columns:
        df_ep = df_ep[df_ep["num_main_speakers"] == 2]
    log.info("Filtered to %d two-speaker episodes", len(df_ep))

    # Select 5 AI-related episodes
    targets = [
        "Mostafa Elbermawy — on Long-Lasting Work, Self-Development, and Why AI Will Not Replace Us",
        "China's Six Front War With America - How To Weaponise COVID-19, 5G & AI",
        "Al and Rishal talk about Rishal’s book Grokking AI Algorithms",
        "AI and data-driven adaptation with Colin Shearer",
        "Augmented Intelligence with AI in Manufacturing - Paul Boris"
    ]

    mask = df_ep["title"].fillna("").apply(lambda x: any(t.lower() in x.lower() for t in targets))
    selected_eps = df_ep[mask].to_dict(orient="records")
    log.info("Found %d matching episodes.", len(selected_eps))

    if not selected_eps:
        log.warning("No matching episodes found — check spelling or file contents.")
        return

    # Index turns by episode_id
    log.info(f"Loading speaker turns from {turn_path}")
    turns = load_jsonl_gz(str(turn_path))
    df_turns = pd.DataFrame(turns)
    log.info("Total turns loaded: %d", len(df_turns))

    llm = LLMInterface(model_name=args.model_name)

    # Process each prompt variant
    for prompt_id, tmpl in enumerate(PROMPTS, start=1):
        out_dir = Path(args.output_root) / f"prompt{prompt_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        total_turns = 0
        for ep in tqdm(selected_eps, desc=f"Prompt {prompt_id} episodes"):
            ep_id = ep.get("episode_id") or ep.get("id")
            if not ep_id:
                continue

            ep_turns = df_turns[df_turns["episode_id"] == ep_id]
            if ep_turns.empty:
                continue

            prompts, meta = [], []
            for _, row in ep_turns.iterrows():
                text = (row.get("text") or "").strip()
                if count_words(text) < args.min_words:
                    continue
                prompts.append(tmpl.format(turn_text=text))
                meta.append({
                    "text": text,
                    "speaker_id": row.get("speaker_id"),
                    "speaker_name": row.get("inferred_speaker_name", "N/A"),
                    "speaker_role": row.get("inferred_speaker_role", "N/A")
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