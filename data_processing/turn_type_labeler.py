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
DEFAULT_DATA_ROOT = "data"
DEFAULT_INPUT_SUBDIR = "parsed"
DEFAULT_OUTPUT_SUBDIR = "turn_type_labeled"
DEFAULT_CATEGORIES = ["business", "commentary", "news", "religion", "sports"]


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

TURN_TYPE_PROMPT = """
You are labeling the TURN TYPE of a single conversation turn.
Your job is to choose exactly ONE label from the list below, using the definitions and decision rules provided.

You MUST output exactly ONE label (verbatim) and NOTHING ELSE.

Allowed output (choose ONE):
- Substantive
- Backchannel
- Procedural
- Disrupted

=== Turn Type Definitions (use these) ===

1) Substantive
- A turn that advances the conversation by adding new information (claims) OR managing the topic.
- Includes: stating facts/opinions, explaining, narrating, arguing, answering questions, asking clarification questions,
  challenging/correcting, proposing actions, or otherwise moving the discussion forward.
- Generally not just a minimal acknowledgment.

2) Backchannel
- A signal of continued attention without taking the floor.
- Typically very short acknowledgments like: "Yeah", "Uh-huh", "Right", "Okay", "Sure", "Mm-hmm".
- Does NOT introduce new claims, does NOT meaningfully steer the topic, does NOT ask a real question.

3) Procedural
- Meta-talk about the channel or setting, not about the topic.
- Examples: "Can you hear me?", "You're muted", "Hold on a sec", "Let me restart", "Connection is bad".

4) Disrupted
- A turn cut off by an interruption before a complete thought is formed.
- Indicators: incomplete syntax, trailing dash/ellipsis suggesting interruption ("I was thinking that-"),
  or clearly unfinished sentence where the intent cannot be completed.

=== Decision Rules / Tie-breakers (IMPORTANT) ===
- Choose ONE label even if multiple patterns appear.
- Priority:
  1) If the turn is about audio/connection/turn-taking mechanics, choose Procedural.
  2) If the turn is clearly cut off / incomplete mid-thought, choose Disrupted.
  3) If the turn is merely a short acknowledgment with no added content, choose Backchannel.
  4) Otherwise choose Substantive (default), including ANY real question (especially clarification), answer, claim, or challenge.

Output rules:
- Output ONLY the label text (one line).
- No punctuation, no quotes, no explanation.

TURN:
{turn_text}
"""


ALLOWED = {"Substantive", "Backchannel", "Procedural", "Disrupted"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_category_dirs(
    data_root: str,
    categories: Optional[List[str]],
    auto_categories: bool,
    input_subdir: str,
) -> List[str]:
    if auto_categories:
        found: List[str] = []
        if not os.path.isdir(data_root):
            return found
        for name in sorted(os.listdir(data_root)):
            category_dir = os.path.join(data_root, name)
            if not os.path.isdir(category_dir):
                continue
            if os.path.isdir(os.path.join(category_dir, input_subdir)):
                found.append(name)
        return found
    return categories if categories else DEFAULT_CATEGORIES


def discover_episode_files(data_root: str, category: str, input_subdir: str) -> List[str]:
    pattern = os.path.join(data_root, category, input_subdir, "*.json")
    return sorted(glob.glob(pattern))


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
    ap.add_argument("--input_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--input_subdir", type=str, default=DEFAULT_INPUT_SUBDIR)
    ap.add_argument("--output_subdir", type=str, default=DEFAULT_OUTPUT_SUBDIR)
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument(
        "--auto_categories",
        action="store_true",
        help="Discover all category folders under --input_root that contain --input_subdir.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Prompt chunk size per llm.generate() call. Set <=0 to label a full episode in one batch.",
    )
    ap.add_argument("--max_turn_chars", type=int, default=3000)
    ap.add_argument("--model_name", type=str, default=MODEL_NAME)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--download_dir", type=str, default="/shared/4/models")
    ap.add_argument("--max_tokens", type=int, default=8)
    args = ap.parse_args()

    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input root not found or not a directory: {args.input_root}")

    ensure_dir(args.output_root)
    categories = discover_category_dirs(
        args.input_root,
        args.categories,
        args.auto_categories,
        args.input_subdir,
    )
    if not categories:
        raise FileNotFoundError(
            f"No categories found under input_root={args.input_root} with input_subdir={args.input_subdir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    llm_if = LLMInterface(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        download_dir=args.download_dir,
        max_tokens=args.max_tokens,
    )

    for category in categories:
        files = discover_episode_files(args.input_root, category, args.input_subdir)
        if not files:
            print(
                f"Skipping category={category}: no .json files found under "
                f"{os.path.join(args.input_root, category, args.input_subdir)}"
            )
            continue

        output_dir = os.path.join(args.output_root, category, args.output_subdir)
        ensure_dir(output_dir)

        for fp in tqdm(files, desc=f"{category}: TurnType Episodes"):
            obj = load_episode_json(fp)
            turns = extract_turns(obj)
            if not turns:
                continue

            episode_id = get_episode_id(turns, fp)
            out_path = os.path.join(output_dir, f"{episode_id}.json")

            prompts: List[str] = []
            for t in turns:
                txt = truncate_text(t.get("turn_text", ""), args.max_turn_chars)
                user = TURN_TYPE_PROMPT.format(turn_text=txt)
                prompts.append(build_chat_prompt(tokenizer, user))

            if args.batch_size and args.batch_size > 0:
                outputs: List[str] = []
                for i in range(0, len(prompts), args.batch_size):
                    outputs.extend(llm_if.generate_batch(prompts[i:i + args.batch_size]))
            else:
                outputs = llm_if.generate_batch(prompts)

            labeled_turns: List[Dict[str, Any]] = []
            for t, out_text in zip(turns, outputs):
                rec = dict(t)
                rec.setdefault("category", category)
                lab = normalize_label(out_text)
                rec["turn_type_label"] = lab
                if lab is None:
                    rec["turn_type_label_error"] = out_text[:400]
                labeled_turns.append(rec)

            save_episode_turns(out_path, labeled_turns)

    print(
        "Done. Wrote per-episode files under: "
        f"{args.output_root}/{{category}}/{args.output_subdir}"
    )


if __name__ == "__main__":
    main()
