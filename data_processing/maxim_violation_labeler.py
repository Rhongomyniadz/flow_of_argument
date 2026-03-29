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
DEFAULT_INPUT_SUBDIR = "conversation_moves_labeled"
DEFAULT_OUTPUT_SUBDIR = "maxim_violations_labeled"
DEFAULT_CATEGORIES = ["business", "commentary", "news", "political", "religion", "sports"]
MAXIM_VIOLATION_SCHEME = "exp7_grounding_v1"


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
        max_tokens: int = 8,
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
    "No Violation",
    "Quantity",
    "Relation",
    "Manner",
]
ALLOWED_SET = set(ALLOWED)

MAXIM_PROMPT = """
You are an expert conversation analyst.

Your task is to decide whether the CURRENT TURN is a maxim violation.

Use ONLY the PREVIOUS TURN and the CURRENT TURN.
Judge only from the source turn and its immediate local context.
Do NOT use any later turn.
Do NOT decide based on whether anyone later asked for repair or clarification.

Main idea:
A maxim violation happens when the CURRENT TURN creates a clear local interaction problem, compared with the PREVIOUS TURN and what it makes relevant next.
The problem must be strong enough that a cooperative listener would reasonably pause normal uptake and instead ask for clarification, ask for a better answer, or challenge the turn.

Allowed labels (output EXACTLY one):
- No Violation
- Quantity
- Relation
- Manner

Label meanings:
1) Quantity
   - The turn gives too little or too much for what is needed right now.
   - Common cases:
     - incomplete or underinformative answer
     - evasive answer
     - overly long answer that gets in the way of the projected task

2) Relation
   - The turn does not respond well enough to what the previous turn makes relevant.
   - Use this when the current speaker is expected to address something specific, but instead shifts away, answers a different issue, or does not address the main point on the table.

3) Manner
   - The turn is too hard to understand for current purposes.
   - Use this when it is unclear, incomplete, vague, disorganized, ambiguous, or too under-specified to interpret straightforwardly.

4) No Violation
   - The turn is usable enough for the local purpose, even if it is brief, awkward, rude, disagreeing, or socially dispreferred.
   - Normal topic management is not automatically a violation.
   - Self-correction is not automatically a violation if the speaker fixes the problem within the same turn.
   - Backchannels and procedural turns are usually No Violation.

How to decide:
- Focus on what the immediately previous turn makes relevant next.
- Ask: could a cooperative listener continue normally, or would they reasonably need to stop and ask for clarification, ask for more, or challenge it?
- If the turn works well enough as a normal next contribution, choose No Violation.
- If more than one label seems possible, use this priority:
  1) Manner if the turn is too unclear to use at all
  2) Relation if the main problem is not addressing the relevant issue
  3) Quantity if it is responsive but gives too little or too much

Important:
- The turn-type and move labels below are only hints.
- Do NOT automatically treat Topic Shift, Stonewalling / Non-Response, or Self-Correction as violations.
- Judge the local interaction problem, not overall conversation quality.

Turn type meanings:
- Substantive:
  A turn that moves the conversation forward by adding information, making a claim, giving an opinion, answering, or managing the topic.
  Common signs: usually longer than a few words, contains content words, carries the main discussion.
- Backchannel:
  A short signal of attention that does not take the floor.
  Common signs: very short turns like “yeah,” “uh-huh,” “right,” “okay.”
- Procedural:
  Meta-talk about the conversation channel or situation, not the topic itself.
  Common signs: “Can you hear me?”, “You're muted,” “Hold on.”
- Disrupted:
  A turn that gets cut off before a complete thought is finished.
  Common signs: broken syntax, interruption, trailing dash.

Move type meanings:
Constructive moves:
- Assert / Elaborate:
  States a fact, opinion, claim, explanation, or added detail.
- Agree / Align:
  Clearly agrees with or supports the previous speaker.
- Answer:
  Gives information requested by a previous question.

Repair moves:
- Clarification Request (Generic):
  Signals general difficulty understanding what the other person means.
  Examples: “What do you mean?” “I don't follow.”
- Clarification Request (Specific):
  Signals that some specific missing piece is needed.
  Examples: “Which report?” “Who is 'he'?”
- Correction / Challenge:
  Rejects or disputes a previous claim, assumption, or framing.
- Self-Correction:
  The speaker fixes or revises their own turn while still speaking.

Disengagement moves:
- Topic Shift:
  Changes the subject without clearly linking it to the current one.
- Stonewalling / Non-Response:
  Gives a very short, non-committal, or minimal response where more engagement might have been expected.

Output rules:
- Output ONLY one label from the allowed list
- No quotes
- No punctuation
- No explanation

PREVIOUS TURN TYPE:
{prev_type}

PREVIOUS TURN MOVE:
{prev_move}

PREVIOUS TURN:
{prev_turn}

CURRENT TURN TYPE:
{cur_type}

CURRENT TURN MOVE:
{cur_move}

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
    match = re.search(r"(\d+)\.json$", os.path.basename(fallback_path))
    if match:
        return match.group(1)
    raise ValueError(f"Could not infer episode_id from data or filename: {fallback_path}")


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    return text if len(text) <= max_chars else (text[:max_chars] + " …(truncated)")


def build_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_label(raw: str) -> Optional[str]:
    if raw is None:
        return None
    text = raw.strip().splitlines()[0].strip()
    text = text.strip(" \"'\t")
    text = re.sub(r"[.。]+$", "", text).strip()

    if text in ALLOWED_SET:
        return text

    for label in sorted(ALLOWED, key=len, reverse=True):
        if text.startswith(label):
            return label
    return None


def save_episode_turns(out_path: str, turns: List[Dict[str, Any]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)


def main() -> None:
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
    ap.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on number of episode files processed per category.",
    )
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
        if args.max_episodes is not None and args.max_episodes > 0:
            files = files[: args.max_episodes]
        if not files:
            print(
                f"Skipping category={category}: no .json files found under "
                f"{os.path.join(args.input_root, args.input_subdir, category)}"
            )
            continue

        output_dir = os.path.join(args.output_root, args.output_subdir, category)
        ensure_dir(output_dir)

        for fp in tqdm(files, desc=f"{category}: Maxim Episodes"):
            obj = load_episode_json(fp)
            turns = extract_turns(obj)
            if not turns:
                continue

            episode_id = get_episode_id(turns, fp)
            out_path = os.path.join(output_dir, f"{episode_id}.json")

            prompts: List[str] = []
            for i, turn in enumerate(turns):
                prev_turn = turns[i - 1] if i > 0 else {}

                prev_text = truncate_text(prev_turn.get("turn_text", ""), args.max_turn_chars)
                cur_text = truncate_text(turn.get("turn_text", ""), args.max_turn_chars)

                user = MAXIM_PROMPT.format(
                    prev_type=normalize_space(prev_turn.get("turn_type_label")) or "[NONE]",
                    prev_move=normalize_space(prev_turn.get("conversation_move_label")) or "[NONE]",
                    prev_turn=prev_text or "[NONE]",
                    cur_type=normalize_space(turn.get("turn_type_label")) or "[UNKNOWN]",
                    cur_move=normalize_space(turn.get("conversation_move_label")) or "[UNKNOWN]",
                    cur_turn=cur_text or "[EMPTY]",
                )
                prompts.append(build_chat_prompt(tokenizer, user))

            if args.batch_size and args.batch_size > 0:
                outputs: List[str] = []
                for i in range(0, len(prompts), args.batch_size):
                    outputs.extend(llm_if.generate_batch(prompts[i:i + args.batch_size]))
            else:
                outputs = llm_if.generate_batch(prompts)

            labeled_turns: List[Dict[str, Any]] = []
            for turn, out_text in zip(turns, outputs):
                rec = dict(turn)
                rec.setdefault("category", category)
                rec["maxim_violation_scheme"] = MAXIM_VIOLATION_SCHEME
                label = normalize_label(out_text)
                rec["maxim_violation_label"] = label
                if label is None:
                    rec["maxim_violation_label_error"] = out_text[:400]
                labeled_turns.append(rec)

            save_episode_turns(out_path, labeled_turns)

    print(
        "Done. Updated per-episode files under: "
        f"{args.output_root}/{args.output_subdir}/{{category}}"
    )


if __name__ == "__main__":
    main()
