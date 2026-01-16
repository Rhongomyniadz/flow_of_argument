import os
import json
import glob
import argparse
from typing import List, Dict, Any, Tuple

from tqdm import tqdm

# ---- your vLLM wrapper (as provided) ----
from vllm import LLM, SamplingParams

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
        download_dir: str = "/shared/4/models",
        max_tokens: int = 16,
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


# ----------------- helpers -----------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def simple_tokenize(text: str) -> List[str]:
    # lightweight "token" approximation (whitespace split).
    # If you want exact tokenizer counts, plug in the HF tokenizer later.
    return text.strip().split()

def truncate_to_last_k_tokens(text: str, k: int) -> str:
    if k <= 0:
        return ""
    toks = simple_tokenize(text)
    if len(toks) <= k:
        return text
    return " ".join(toks[-k:])

def extract_turn_text(turn: Dict[str, Any]) -> str:
    # prioritize chosen_text_key if available; fallback to transcript / turn_text
    key = turn.get("chosen_text_key")
    if isinstance(key, str) and key in turn and isinstance(turn[key], str):
        return turn[key]
    for cand in ("transcript", "turn_text", "text"):
        if cand in turn and isinstance(turn[cand], str):
            return turn[cand]
    return ""

def format_history_for_prompt(history_turns: List[Tuple[str, str]]) -> str:
    # history_turns: [(speaker, text), ...]
    # keep it compact but explicit
    lines = []
    for spk, txt in history_turns:
        txt = txt.replace("\n", " ").strip()
        lines.append(f"{spk}: {txt}")
    return "\n".join(lines)

def build_prompts_for_episode(
    turns: List[Dict[str, Any]],
    k: int,
    use_speaker: bool = True,
    stance_target: str = "immediately_previous_turn",
) -> List[str]:
    """
    stance_target:
      - "immediately_previous_turn": stance of current turn toward the last turn
      - "previous_nontrivial_turn": stance toward last non-empty turn text
    """
    prompts = []
    history_accum: List[Tuple[str, str]] = []  # full growing history (speaker, text)

    for i, t in enumerate(turns):
        cur_text = extract_turn_text(t).replace("\n", " ").strip()
        cur_speaker = t.get("speaker_id") or t.get("speaker") or "SPEAKER"
        cur_label = cur_speaker if use_speaker else "SPEAKER"

        # identify target text (what current turn is responding to)
        target_text = ""
        if i == 0:
            target_text = ""
        else:
            if stance_target == "previous_nontrivial_turn":
                j = i - 1
                while j >= 0:
                    cand = extract_turn_text(turns[j]).replace("\n", " ").strip()
                    if cand:
                        target_text = cand
                        break
                    j -= 1
            else:
                target_text = extract_turn_text(turns[i - 1]).replace("\n", " ").strip()

        # build context from full conversation so far (excluding current)
        # then truncate to last k tokens
        history_str_full = format_history_for_prompt(history_accum)
        history_str = truncate_to_last_k_tokens(history_str_full, k)

        # prompt: force single-digit output
        prompt = f"""You are doing stance detection in a conversation.

Task:
Given the conversation context and the current turn, rate the CURRENT turn's stance toward the IMMEDIATELY PRECEDING turn's content.

Scale (1-5):
1 = disagree / contradict
2 = mostly disagree
3 = neutral / unclear / unrelated / no stance
4 = mostly agree
5 = agree / support

Rules:
- Output ONLY a single digit: 1, 2, 3, 4, or 5.
- If the current turn is an ad, narration, or does not respond to the previous turn, output 3.

Conversation context (most recent up to {k} tokens):
{history_str if history_str else "[NO PRIOR CONTEXT]"}

Immediately preceding turn (target):
{target_text if target_text else "[NO PRECEDING TURN]"}

Current turn:
{cur_label}: {cur_text}

Answer (single digit 1-5):"""
        prompts.append(prompt)

        # update history accumulator AFTER building prompt
        if cur_text:
            history_accum.append((cur_label, cur_text))
        else:
            history_accum.append((cur_label, ""))

    return prompts

def parse_stance_digit(raw: str) -> int:
    raw = (raw or "").strip()
    # common model outputs: "5", "5.", "Answer: 5", etc.
    for ch in raw:
        if ch in "12345":
            return int(ch)
    return 3  # safe fallback


# ----------------- main pipeline -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/conversation_moves_labeled")
    ap.add_argument("--out_dir", type=str, default="data/stance_labeled")
    ap.add_argument("--k", type=int, default=512, help="max prior-context tokens (whitespace tokens)")
    ap.add_argument("--glob", type=str, default="*.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    llm = LLMInterface(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    in_paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not in_paths:
        raise FileNotFoundError(f"No input files found under {args.in_dir} matching {args.glob}")

    k_dir = os.path.join(args.out_dir, str(args.k))
    ensure_dir(k_dir)

    for path in tqdm(in_paths, desc="Episodes"):
        episode_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(k_dir, f"{episode_id}.json")

        turns = load_json(path)
        if not isinstance(turns, list):
            raise ValueError(f"Expected a list in {path}, got {type(turns)}")

        prompts = build_prompts_for_episode(turns, k=args.k)

        # run vLLM
        outputs: List[str] = []
        if args.batch_size and args.batch_size > 0:
            for i in range(0, len(prompts), args.batch_size):
                outputs.extend(llm.generate_batch(prompts[i:i + args.batch_size]))
        else:
            outputs = llm.generate_batch(prompts)

        if len(outputs) != len(turns):
            raise RuntimeError(f"Output length mismatch for {episode_id}: {len(outputs)} vs {len(turns)}")

        # attach stance results
        stance_labeled = []
        for t, raw in zip(turns, outputs):
            stance = parse_stance_digit(raw)
            t2 = dict(t)
            t2["stance_5pt"] = stance
            t2["stance_raw"] = raw
            t2["stance_context_k"] = args.k
            stance_labeled.append(t2)

        save_json(out_path, stance_labeled)

if __name__ == "__main__":
    main()
