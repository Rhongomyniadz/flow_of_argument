import re
import gc
import argparse
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


# -------------------- LLM wrapper (your style) --------------------
class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2,
        temperature: float = 0.0,   # deterministic for labeling
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        download_dir: str = "/shared/4/models",
        max_tokens: int = 8,        # only need 0/1
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


# -------------------- Prompting --------------------
SYSTEM_RULES = """You are a strict binary classifier.

Task: Decide whether QUESTION is a clarification question about PREV_TURN.

Label = 1 if QUESTION asks to clarify, confirm, disambiguate, or request explanation of something in PREV_TURN.
Label = 0 if QUESTION is unrelated, starts a new topic, or does not clarify PREV_TURN.

Output ONLY a single character: 1 or 0.
No words, no punctuation, no extra whitespace.
"""

def build_prompt(prev_turn: str, question: str) -> str:
    prev_turn = (prev_turn or "").strip()
    question = (question or "").strip()
    return (
        f"{SYSTEM_RULES}\n"
        f"PREV_TURN:\n{prev_turn}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer (1 or 0):"
    )


# -------------------- Parsing --------------------
_label_re = re.compile(r"\b([01])\b")

def parse_label(text: str) -> int:
    """Parse model output into 0/1. Defaults to 0 if unparseable."""
    if not isinstance(text, str):
        return 0
    m = _label_re.search(text.strip())
    return int(m.group(1)) if m else 0


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/edenzha/flow_of_argument/results/analysis_charts/clarification_prediction/questions.csv")
    ap.add_argument("--output", default="/home/edenzha/flow_of_argument/results/analysis_charts/clarification_prediction/questions_labeled.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)

    # vLLM / model knobs
    ap.add_argument("--model_name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--download_dir", default="/shared/4/models")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    else:
        df = df.copy()

    q_col = "clarification_sentence"
    prev_col = "turn_text_prev"

    # Pre-label empty questions as 0 (no LLM call)
    df["is_clarification"] = pd.NA
    empty_mask = df[q_col].isna() | (df[q_col].astype(str).str.strip() == "")
    df.loc[empty_mask, "is_clarification"] = 0

    need_idx = df.index[df["is_clarification"].isna()].tolist()
    print(f"[info] total={len(df)} need_label={len(need_idx)}")

    # Init vLLM
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
        max_tokens=8,
    )

    bs = args.batch_size
    for start in tqdm(range(0, len(need_idx), bs), desc="labeling"):
        batch_idx = need_idx[start:start + bs]

        prompts: List[str] = []
        for i in batch_idx:
            prev_turn = "" if pd.isna(df.at[i, prev_col]) else str(df.at[i, prev_col])
            question = "" if pd.isna(df.at[i, q_col]) else str(df.at[i, q_col])
            prompts.append(build_prompt(prev_turn, question))

        outputs = llm.generate_batch(prompts)
        labels = [parse_label(o) for o in outputs]
        df.loc[batch_idx, "is_clarification"] = labels

        # checkpoint every batch
        df.to_csv(out_path, index=False)

        del prompts, outputs, labels
        gc.collect()

    print(f"[done] wrote: {out_path}")


if __name__ == "__main__":
    main()
