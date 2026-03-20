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
DEFAULT_INPUT_SUBDIR = "turn_type_labeled"
DEFAULT_OUTPUT_SUBDIR = "conversation_moves_labeled"
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

ALLOWED = [
    "Assert / Elaborate",
    "Agree / Align",
    "Answer",
    "Clarification Request (Generic)",
    "Clarification Request (Specific)",
    "Correction / Challenge",
    "Self-Correction",
    "Topic Shift",
    "Stonewalling / Non-Response",
]
ALLOWED_SET = set(ALLOWED)

MOVE_PROMPT = """
You are an expert conversation analyst. Your job is to label the PRIMARY Conversation Move of the CURRENT TURN,
using the PREVIOUS TURN only as context.

Conversation Moves are grouped into:
A) Constructive (Building the World): adds to the "Explicit Claims" pile
B) Repair (Fixing the World): signals a Gricean Maxim violation that must be repaired
C) Disengagement (Abandoning the World): negative outcomes; avoids engaging the world

You MUST output exactly ONE label from the list below, and output NOTHING ELSE.

Allowed labels (output EXACTLY one):
- Assert / Elaborate
- Agree / Align
- Answer
- Clarification Request (Generic)
- Clarification Request (Specific)
- Correction / Challenge
- Self-Correction
- Topic Shift
- Stonewalling / Non-Response

=== Official Definitions (use these) ===

A. Constructive Moves (Building the World)
1) Assert / Elaborate
   - Stating a fact, opinion, or description (adds new explicit content).
   - Includes explaining, justifying, narrating, or elaborating, when not primarily answering a question.
2) Agree / Align
   - Explicit agreement with the previous speaker's claim or assumption.
   - Examples: "That's a good point", "Exactly", "Right", "I agree".
3) Answer
   - Providing information solicited by a previous question (directly addresses what was asked).

B. Repair Moves (Fixing the World)
4) Clarification Request (Generic)
   - Signaling a failure to understand Manner or Relation (unclear meaning or relevance).
   - Examples: "What do you mean?", "I don't follow."
   - Does NOT specify what exact piece is missing.
5) Clarification Request (Specific)
   - Signaling a missing Claim or Assumption (Quantity violation).
   - Asks for a specific missing element needed to interpret prior content.
   - Examples: "Which report are you referring to?", "Who is 'he'?"
6) Correction / Challenge
   - Explicitly rejecting a previous Claim or Assumption (Quality violation).
   - Examples: "No, that's not right", "I disagree with your premise."
7) Self-Correction
   - The speaker catches their own violation mid-turn and corrects/rephrases themselves.
   - Examples: "I mean, well, let me rephrase that..."

C. Disengagement Moves (Abandoning the World)
8) Topic Shift
   - Abruptly changing the subject without a bridging assumption.
   - Example: "Anyway, did you see the game?"
9) Stonewalling / Non-Response
   - Deliberately short, non-committal answers to open questions.
   - Examples: "Maybe", "I guess."
   - Use when the turn avoids engaging rather than genuinely answering.

=== Primary-move selection rules (IMPORTANT) ===
- Choose ONE label even if multiple moves appear.
- Prefer the move that best describes the turn's main function in the interaction.

Tie-breakers (apply in order):
1) If the CURRENT TURN is primarily asking for clarification, choose Clarification Request (Specific/Generic),
   even if it also contains commentary.
2) If the CURRENT TURN primarily rejects/disputes prior content, choose Correction / Challenge.
3) If the PREVIOUS TURN asked a question and the CURRENT TURN provides the requested information,
   choose Answer (even if it also elaborates).
4) If it begins with explicit agreement and then adds detail, choose Agree / Align unless the agreement is minor
   and the bulk is new assertion (then Assert / Elaborate).
5) If it avoids engaging with an open question via short non-committal language, choose Stonewalling / Non-Response.
6) If it abruptly changes topic without a bridge, choose Topic Shift.

Output rules:
- Output ONLY one label from the Allowed labels list.
- No quotes, no punctuation, no explanations.

PREVIOUS TURN:
{prev_turn}

CURRENT TURN:
{cur_turn}
"""



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_category_dirs(
    data_root: str,
    categories: Optional[List[str]],
    auto_categories: bool,
    input_subdir: str,
) -> List[str]:
    category_root = os.path.join(data_root, input_subdir)
    if auto_categories:
        found: List[str] = []
        if not os.path.isdir(category_root):
            return found
        for name in sorted(os.listdir(category_root)):
            category_dir = os.path.join(category_root, name)
            if os.path.isdir(category_dir):
                found.append(name)
        return found
    return categories if categories else DEFAULT_CATEGORIES


def discover_episode_files(data_root: str, input_subdir: str, category: str) -> List[str]:
    pattern = os.path.join(data_root, input_subdir, category, "*.json")
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

    if s in ALLOWED_SET:
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
        help="Discover all category folders under --input_root/--input_subdir.",
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
        files = discover_episode_files(args.input_root, args.input_subdir, category)
        if not files:
            print(
                f"Skipping category={category}: no .json files found under "
                f"{os.path.join(args.input_root, args.input_subdir, category)}"
            )
            continue

        output_dir = os.path.join(args.output_root, args.output_subdir, category)
        ensure_dir(output_dir)

        for fp in tqdm(files, desc=f"{category}: Move Episodes"):
            obj = load_episode_json(fp)
            turns = extract_turns(obj)
            if not turns:
                continue

            episode_id = get_episode_id(turns, fp)
            out_path = os.path.join(output_dir, f"{episode_id}.json")

            prompts: List[str] = []
            for i, t in enumerate(turns):
                prev_txt = turns[i - 1].get("turn_text", "") if i > 0 else ""
                cur_txt = t.get("turn_text", "")

                prev_txt = truncate_text(prev_txt, args.max_turn_chars)
                cur_txt = truncate_text(cur_txt, args.max_turn_chars)

                user = MOVE_PROMPT.format(prev_turn=prev_txt, cur_turn=cur_txt)
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
                rec["conversation_move_label"] = lab
                if lab is None:
                    rec["conversation_move_label_error"] = out_text[:400]
                labeled_turns.append(rec)

            save_episode_turns(out_path, labeled_turns)

    print(
        "Done. Updated per-episode files under: "
        f"{args.output_root}/{args.output_subdir}/{{category}}"
    )


if __name__ == "__main__":
    main()
