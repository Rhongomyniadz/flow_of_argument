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
NON_VIOLATION_TURN_TYPES = {"Backchannel", "Procedural", "Disrupted"}
NON_VIOLATION_MOVES = {
    "Clarification Request (Generic)",
    "Clarification Request (Specific)",
    "Agree / Align",
}
ANSWERISH_MOVES = {
    "Answer",
    "Assert / Elaborate",
}
PROMO_PATTERNS = [
    r"\bsponsored by\b",
    r"\bbrought to you by\b",
    r"\bpromo code\b",
    r"\bdownload the app\b",
    r"\bfree trial\b",
    r"\bprice estimates\b",
    r"\btop-rated pros\b",
    r"\bfree samsung\b",
    r"\bthumbtack\b",
    r"\bhomes\.com\b",
    r"\bmetro\b",
    r"\b5g data\b",
]
SPLICE_START_PATTERNS = (
    "and ",
    "but ",
    "because ",
    "with ",
    "from ",
    "to ",
    "of ",
    "for ",
    "in ",
    "on ",
    "at ",
    "some ",
    "ones ",
    "the difference ",
)
DISCOURSE_CONTINUATION_PATTERNS = (
    "yeah",
    "yes",
    "no",
    "so",
    "i mean",
    "well",
    "right",
    "exactly",
    "actually",
    "definitely",
    "and",
    "but",
)

MAXIM_PROMPT = """
You are an expert conversation analyst.

Your task is to decide whether the CURRENT TURN is a maxim violation. Judge from the source turn and its immediate local context.

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
- Do NOT label obvious advertisement copy, sponsor reads, station IDs, or promo segments as maxim violations just because they are abrupt or off-topic. If the text looks like inserted non-conversational material, choose No Violation.
- Do NOT label transcript splice boundaries, broken segment joins, or metadata/captioning artifacts as Relation or Quantity violations. If the discontinuity looks like a recording or transcription artifact rather than a cooperative speaker choice, choose No Violation.
- Do NOT label Disrupted turns as maxim violations unless the remaining text itself creates a clear local Manner problem beyond simple truncation. Simple cut-offs, interruptions, or partial fragments caused by segmentation should usually be No Violation.
- Do NOT label normal interview progression as a maxim violation. Follow-up questions, topic elaborations, requests to tell us more, and ordinary speaker handoffs are usually No Violation.
- Do NOT label a turn as Relation just because it introduces a subtopic, example, comparison, anecdote, or partial reframing that still connects naturally to the previous turn.
- Do NOT label a turn as Quantity just because it is long, detailed, enthusiastic, or somewhat indirect. Use Quantity only when it is clearly too little or too much for the immediate local need.
- Clarification requests, agreement moves, and ordinary answer-prefaces like "yeah", "so", or "I mean" are usually No Violation unless they create a clear local breakdown.
- For interviews, podcasts, and conversational Q&A, long informative answers are usually No Violation. Only use Quantity when the answer is plainly evasive, drastically under-informative, or so excessive that it blocks the projected task.
- Do NOT use Relation when the current turn still answers, elaborates, exemplifies, or extends the previous turn's topic, even if the connection is loose.
- Prefer No Violation over Relation when both turns share obvious topical vocabulary, named entities, or a clear discourse continuation marker.
- Use Relation only for a genuine relevance breakdown: the current turn should fail to address what the prior turn made relevant in a way that a cooperative listener would likely challenge.
- Use Manner only when the current turn itself is seriously hard to interpret. Spoken-style disfluency, casual wording, or compressed syntax are not enough by themselves.
- Prefer No Violation over Manner when the main point of the turn is still recoverable.
- Be especially conservative with Relation. If there is any reasonable reading on which the current turn still connects to the prior topic, prefer No Violation.
- For Manner, brief fragmentary or broken turns can still count when a listener would genuinely struggle to recover the intended meaning from the local context.

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


def looks_like_promo(text: str) -> bool:
    text = normalize_space(text).lower()
    return any(re.search(pattern, text) for pattern in PROMO_PATTERNS)


def looks_like_splice_artifact(prev_text: str, cur_text: str) -> bool:
    prev_text = normalize_space(prev_text)
    cur_text = normalize_space(cur_text)
    if not prev_text or not cur_text:
        return False
    prev_tail = prev_text[-1]
    cur_head = cur_text[0]
    return (
        prev_tail not in ".?!"
        and cur_head.islower()
        and cur_text.lower().startswith(SPLICE_START_PATTERNS)
    )


def has_continuation_marker(text: str) -> bool:
    text = normalize_space(text).lower()
    return text.startswith(DISCOURSE_CONTINUATION_PATTERNS)


def lexical_overlap(prev_text: str, cur_text: str) -> float:
    prev_tokens = {
        tok for tok in re.findall(r"[a-zA-Z]{3,}", normalize_space(prev_text).lower())
        if tok not in {"that", "this", "with", "from", "have", "been", "they", "them", "their", "about", "would", "could", "should"}
    }
    cur_tokens = {
        tok for tok in re.findall(r"[a-zA-Z]{3,}", normalize_space(cur_text).lower())
        if tok not in {"that", "this", "with", "from", "have", "been", "they", "them", "their", "about", "would", "could", "should"}
    }
    if not prev_tokens or not cur_tokens:
        return 0.0
    return len(prev_tokens & cur_tokens) / min(len(prev_tokens), len(cur_tokens))


def looks_like_reasonable_answer(prev_turn: Dict[str, Any], turn: Dict[str, Any]) -> bool:
    prev_text = normalize_space(prev_turn.get("turn_text", ""))
    cur_text = normalize_space(turn.get("turn_text", ""))
    if not prev_text or not cur_text:
        return False
    prev_is_question = "?" in prev_text
    cur_len = len(cur_text.split())
    overlap = lexical_overlap(prev_text, cur_text)
    return (
        prev_is_question
        and cur_len >= 12
        and (has_continuation_marker(cur_text) or overlap >= 0.12)
    )


def severe_manner_signal(text: str) -> bool:
    text = normalize_space(text)
    low = text.lower()
    words = text.split()
    if not text:
        return False
    if text.strip().startswith((".", ",", "?", "'")):
        return True
    if "..." in text and len(words) <= 8:
        return True
    if len(words) <= 4 and any(ch.isalpha() for ch in text):
        return True
    if low.count(" uh") + low.count(" um") >= 2:
        return True
    return False


def apply_hard_filter(i: int, prev_turn: Dict[str, Any], turn: Dict[str, Any], label: Optional[str]) -> str:
    turn_type = normalize_space(turn.get("turn_type_label"))
    move = normalize_space(turn.get("conversation_move_label"))
    prev_text = str(prev_turn.get("turn_text", ""))
    cur_text = str(turn.get("turn_text", ""))
    if i == 0:
        return "No Violation"
    if turn_type in NON_VIOLATION_TURN_TYPES:
        return "No Violation"
    if move in NON_VIOLATION_MOVES:
        return "No Violation"
    if looks_like_promo(cur_text) or looks_like_promo(prev_text):
        return "No Violation"
    if looks_like_splice_artifact(prev_text, cur_text):
        return "No Violation"
    filtered = label or "No Violation"
    overlap = lexical_overlap(prev_text, cur_text)
    if filtered == "Relation":
        if (
            has_continuation_marker(cur_text)
            or overlap >= 0.04
            or looks_like_reasonable_answer(prev_turn, turn)
            or move in {"Topic Shift", "Stonewalling / Non-Response"}
        ):
            return "No Violation"
        if "?" in prev_text and len(normalize_space(cur_text).split()) >= 3:
            return "No Violation"
        if move == "Assert / Elaborate" and len(normalize_space(cur_text).split()) >= 8:
            return "No Violation"
    if filtered == "Quantity":
        if move in ANSWERISH_MOVES and len(normalize_space(cur_text).split()) >= 12:
            return "No Violation"
        if has_continuation_marker(cur_text) and overlap >= 0.12:
            return "No Violation"
    if filtered == "Manner":
        words = normalize_space(cur_text).split()
        if severe_manner_signal(cur_text):
            return "Manner"
        if (
            len(words) >= 6
            and (overlap >= 0.06 or has_continuation_marker(cur_text))
            and not looks_like_splice_artifact(prev_text, cur_text)
        ):
            return "No Violation"
        if cur_text.count("...") == 0 and cur_text.count("uh") + cur_text.count("um") <= 1 and len(words) >= 7:
            return "No Violation"
    return filtered


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
            for i, (turn, out_text) in enumerate(zip(turns, outputs)):
                rec = dict(turn)
                rec.setdefault("category", category)
                rec["maxim_violation_scheme"] = MAXIM_VIOLATION_SCHEME
                label = normalize_label(out_text)
                prev_turn = turns[i - 1] if i > 0 else {}
                rec["maxim_violation_label"] = apply_hard_filter(i, prev_turn, turn, label)
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
