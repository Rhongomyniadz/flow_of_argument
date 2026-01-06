from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from vllm import LLM, SamplingParams


# -----------------------------
# vLLM wrapper (your style)
# -----------------------------
class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        max_tokens: int = 16,
        download_dir: str = "/shared/4/models",
    ):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir,
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
        out = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in out]


# -----------------------------
# Prompting + parsing
# -----------------------------
SYSTEM_RULES = """You are a strict binary classifier.

Task: Decide whether QUESTION is a clarification question about PREV_TURN.

Label=1 if QUESTION asks to clarify/confirm/disambiguate/explain something in PREV_TURN.
Label=0 if QUESTION is unrelated, starts a new topic, or does not clarify PREV_TURN.

Output ONLY a single character: 1 or 0.
No extra words, no punctuation, no newlines.
"""

def build_prompt(prev_turn: str, question: str) -> str:
    # Keep prompt short and stable; avoid huge tokens.
    prev_turn = (prev_turn or "").strip()
    question = (question or "").strip()

    return (
        f"{SYSTEM_RULES}\n"
        f"PREV_TURN:\n{prev_turn}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer (1 or 0):"
    )

_label_re = re.compile(r"\b([01])\b")

def parse_label(text: str) -> int:
    """
    Robustly parse model output into 0/1.
    Accepts outputs like "1", "0", "Answer: 1", etc.
    Defaults to 0 if unparseable.
    """
    if not isinstance(text, str):
        return 0
    t = text.strip()
    m = _label_re.search(t)
    return int(m.group(1)) if m else 0


def make_row_id(row: pd.Series) -> str:
    return f'{row["episode_id"]}:{row["turn_idx_prev"]}->{row["turn_idx_next"]}'


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="If >0, label only first N rows")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output if present")
    ap.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size")
    ap.add_argument("--gpu-mem", type=float, default=0.9, help="gpu_memory_utilization")
    ap.add_argument("--download-dir", default="/shared/4/models")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    else:
        df = df.copy()

    # Stable row_id for resume/merge
    df["row_id"] = df.apply(make_row_id, axis=1)

    # Resume support
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "row_id" in old.columns and "is_clarification" in old.columns:
            old_map = dict(zip(old["row_id"].astype(str), old["is_clarification"].astype(int)))
            df["is_clarification"] = df["row_id"].astype(str).map(old_map)
        else:
            df["is_clarification"] = None
    else:
        df["is_clarification"] = None

    # Pre-label empty questions as 0
    q_col = "clarification_sentence"
    prev_col = "turn_text_prev"

    empty_mask = df[q_col].isna() | (df[q_col].astype(str).str.strip() == "")
    df.loc[empty_mask, "is_clarification"] = 0

    # Determine which rows still need labeling
    need_mask = df["is_clarification"].isna()
    need_indices = df.index[need_mask].tolist()
    print(f"[info] Total rows: {len(df)}")
    print(f"[info] Need labeling: {len(need_indices)}")

    if not need_indices:
        df.drop(columns=["row_id"]).to_csv(out_path, index=False)
        print(f"[done] Nothing to do. Wrote: {out_path}")
        return

    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        download_dir=args.download_dir,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        repetition_penalty=1.05,
        max_tokens=16,
    )

    # Batch labeling with checkpoints
    bs = args.batch_size
    for start in tqdm(range(0, len(need_indices), bs), desc="Labeling"):
        batch_idx = need_indices[start:start + bs]

        prompts: List[str] = []
        for i in batch_idx:
            prev_turn = str(df.at[i, prev_col]) if not pd.isna(df.at[i, prev_col]) else ""
            question = str(df.at[i, q_col]) if not pd.isna(df.at[i, q_col]) else ""
            prompts.append(build_prompt(prev_turn, question))

        outputs = llm.generate_batch(prompts)
        labels = [parse_label(o) for o in outputs]

        df.loc[batch_idx, "is_clarification"] = labels

        # checkpoint write every batch
        df_out = df.drop(columns=["row_id"])
        df_out.to_csv(out_path, index=False)

    print(f"[done] Wrote labeled CSV to: {out_path}")


if __name__ == "__main__":
    main()
