import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Iterable, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


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
        max_tokens: int = 8,        # small output (label only)
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


SYSTEM_PROMPT = (
    "You are a strict conversation-turn annotator. "
    "Output ONLY the label string, with no extra words."
)

TURN_TYPE_PROMPT = """\
Choose exactly ONE Turn Type label for the TURN.

Turn Types:
- Substantive: advances topic with new info/claims OR manages topic (includes repair questions).
- Backchannel: minimal attention signal without taking the floor.
- Procedural: meta-talk about the channel/setting (e.g., can you hear me, you're muted).
- Disrupted: cut off before completing a thought; incomplete syntax.

Output ONLY one of:
Substantive
Backchannel
Procedural
Disrupted

TURN:
{turn_text}
"""

ALLOWED = {"Substantive", "Backchannel", "Procedural", "Disrupted"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_episode(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        return obj["turns"]
    raise ValueError(f"Unrecognized JSON format in {path} (expected list or dict with 'turns')")


def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else (s[:max_chars] + " …(truncated)")


def build_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_label(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip().splitlines()[0].strip()
    s = s.strip(" \"'\t")
    s = re.sub(r"[.。]+$", "", s).strip()
    if s in ALLOWED:
        return s
    # If model adds leading/trailing tokens, try match by prefix (robust parsing, not heuristic labeling)
    for lab in sorted(ALLOWED, key=len, reverse=True):
        if s.startswith(lab):
            return lab
    return None


def stream_json_array(out_path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for rec in records:
            if not first:
                f.write(",\n")
            json.dump(rec, f, ensure_ascii=False)
            first = False
        f.write("\n]\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="results/political/parsed")
    ap.add_argument("--output", type=str, default="data/turn_type_labeled.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_turn_chars", type=int, default=3000)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.output) or ".")

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No .json files found under: {args.input_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm_if = LLMInterface()  # <- use defaults exactly as defined

    def labeled_records() -> Iterable[Dict[str, Any]]:
        for fp in tqdm(files, desc="TurnType Episodes"):
            turns = load_episode(fp)
            if not turns:
                continue

            prompts: List[str] = []
            for t in turns:
                txt = truncate_text(t.get("turn_text", ""), args.max_turn_chars)
                user = TURN_TYPE_PROMPT.format(turn_text=txt)
                prompts.append(build_chat_prompt(tokenizer, user))

            outputs: List[str] = []
            for i in range(0, len(prompts), args.batch_size):
                outputs.extend(llm_if.generate_batch(prompts[i:i + args.batch_size]))

            for t, out_text in zip(turns, outputs):
                rec = dict(t)
                lab = normalize_label(out_text)
                rec["turn_type_label"] = lab
                if lab is None:
                    rec["turn_type_label_error"] = out_text[:400]
                yield rec

    stream_json_array(args.output, labeled_records())
    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
