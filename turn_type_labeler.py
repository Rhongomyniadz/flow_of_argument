import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

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
        max_tokens: int = 8,        # label only
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
- Substantive: advances the conversation by adding new information/claims OR managing the topic (includes repair questions).
- Backchannel: a short signal of continued attention without taking the floor.
- Procedural: meta-talk about the channel/setting (e.g., can you hear me, you're muted, hold on).
- Disrupted: cut off before a complete thought is formed; incomplete syntax.

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


def load_episode_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_turns(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        return obj["turns"]
    raise ValueError("Unrecognized JSON format: expected list or dict with 'turns' list.")


def get_episode_id(turns: List[Dict[str, Any]], fallback_path: str) -> str:
    if turns and "episode_id" in turns[0] and turns[0]["episode_id"] is not None:
        return str(turns[0]["episode_id"])
    m = re.search(r"(\d+)\.json$", os.path.basename(fallback_path))
    if m:
        return m.group(1)
    raise ValueError(f"Could not infer episode_id from data or filename: {fallback_path}")


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
    # Robust parsing only (not heuristic labeling)
    for lab in sorted(ALLOWED, key=len, reverse=True):
        if s.startswith(lab):
            return lab
    return None


def save_episode_turns(out_path: str, turns: List[Dict[str, Any]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="results/political/parsed")
    ap.add_argument("--output_dir", type=str, default="data/labeled")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_turn_chars", type=int, default=3000)
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No .json files found under: {args.input_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm_if = LLMInterface()  # use defaults exactly as defined

    for fp in tqdm(files, desc="TurnType Episodes"):
        obj = load_episode_json(fp)
        turns = extract_turns(obj)
        if not turns:
            continue

        episode_id = get_episode_id(turns, fp)
        out_path = os.path.join(args.output_dir, f"{episode_id}.json")

        prompts: List[str] = []
        for t in turns:
            txt = truncate_text(t.get("turn_text", ""), args.max_turn_chars)
            user = TURN_TYPE_PROMPT.format(turn_text=txt)
            prompts.append(build_chat_prompt(tokenizer, user))

        outputs: List[str] = []
        for i in range(0, len(prompts), args.batch_size):
            outputs.extend(llm_if.generate_batch(prompts[i:i + args.batch_size]))

        labeled_turns: List[Dict[str, Any]] = []
        for t, out_text in zip(turns, outputs):
            rec = dict(t)
            lab = normalize_label(out_text)
            rec["turn_type_label"] = lab
            if lab is None:
                rec["turn_type_label_error"] = out_text[:400]
            labeled_turns.append(rec)

        save_episode_turns(out_path, labeled_turns)

    print(f"Done. Wrote per-episode files under: {args.output_dir}")


if __name__ == "__main__":
    main()
