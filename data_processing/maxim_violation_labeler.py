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
- Make the conservative decision yourself from the prompt. Do not assume a later cleanup step will correct an over-aggressive label.
- Do NOT automatically treat Topic Shift, Stonewalling / Non-Response, or Self-Correction as violations.
- Judge the local interaction problem, not overall conversation quality.
- The first turn in an episode should be No Violation because there is no prior turn whose conversational demand it can fail.
- Do NOT label obvious advertisement copy, sponsor reads, station IDs, or promo segments as maxim violations just because they are abrupt or off-topic. If the text looks like inserted non-conversational material, choose No Violation.
- Do NOT label transcript splice boundaries, broken segment joins, or metadata/captioning artifacts as Relation or Quantity violations. If the discontinuity looks like a recording or transcription artifact rather than a cooperative speaker choice, choose No Violation.
- Do NOT label Disrupted turns as maxim violations unless the remaining text itself creates a clear local Manner problem beyond simple truncation. Simple cut-offs, interruptions, or partial fragments caused by segmentation should usually be No Violation.
- Backchannel and Procedural turns should be No Violation.
- Clarification requests and agreement/alignment moves should be No Violation unless the text itself still creates a clear local Manner problem.
- Do NOT label normal interview progression as a maxim violation. Follow-up questions, topic elaborations, requests to tell us more, and ordinary speaker handoffs are usually No Violation.
- Do NOT label a turn as Relation just because it introduces a subtopic, example, comparison, anecdote, or partial reframing that still connects naturally to the previous turn.
- Do NOT label a turn as Quantity just because it is long, detailed, enthusiastic, or somewhat indirect. Use Quantity only when it is clearly too little or too much for the immediate local need.
- Clarification requests, agreement moves, and ordinary answer-prefaces like "yeah", "so", or "I mean" are usually No Violation unless they create a clear local breakdown.
- For interviews, podcasts, and conversational Q&A, long informative answers are usually No Violation. Only use Quantity when the answer is plainly evasive, drastically under-informative, or so excessive that it blocks the projected task.
- Do NOT use Relation when the current turn still answers, elaborates, exemplifies, or extends the previous turn's topic, even if the connection is loose.
- Prefer No Violation over Relation when both turns share obvious topical vocabulary, named entities, or a clear discourse continuation marker.
- If the previous turn is a question and the current turn gives a minimally contentful answer, completion, or correction, prefer No Violation over Relation unless the reply is clearly off-track.
- For Assert / Elaborate turns, use Relation only when the previous turn made a specific issue relevant and the current turn clearly bypasses that issue rather than continuing it.
- Openings, host setup lines, live commentary, narrative continuation, and ordinary segue language are usually No Violation, not Relation.
- For Topic Shift turns, choose Relation only when the shift clearly abandons an immediately relevant question, challenge, or request.
- Broad topical continuation, wrap-up banter, sign-offs, and play-by-play narration are usually No Violation unless they clearly dodge a specific local demand.
- Use Relation only for a genuine relevance breakdown: the current turn should fail to address what the prior turn made relevant in a way that a cooperative listener would likely challenge.
- Use Manner only when the current turn itself is seriously hard to interpret. Spoken-style disfluency, casual wording, or compressed syntax are not enough by themselves.
- Prefer No Violation over Manner when the main point of the turn is still recoverable.
- Be especially conservative with Relation. If there is any reasonable reading on which the current turn still connects to the prior topic, prefer No Violation.
- For Manner, brief fragmentary or broken turns can still count when a listener would genuinely struggle to recover the intended meaning from the local context.
- Short spoken continuations or completions such as brief answer fragments, repairs, or follow-throughs are usually No Violation if their intended meaning is recoverable from the previous turn.
- Use Manner for short fragments only when the intended proposition is still hard to recover from the previous turn; a short but understandable completion should be No Violation.
- Single-word or short echoic phrases that repeat or complete material already present in the previous turn are usually No Violation, not Manner.
- Use Quantity for short replies only when they are clearly under-informative for the specific question or task, not merely brief.

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


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_category_set(
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


def collect_episode_paths(data_root: str, input_subdir: str, category: str) -> List[str]:
    pattern = os.path.join(data_root, input_subdir, category, "*.json")
    return sorted(glob.glob(pattern))


def read_episode_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_turn_sequence(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        return obj["turns"]
    raise ValueError("Unrecognized JSON format: expected list or dict with 'turns' list.")


def resolve_episode_id(turns: List[Dict[str, Any]], fallback_path: str) -> str:
    if turns and "episode_id" in turns[0] and turns[0]["episode_id"] is not None:
        return str(turns[0]["episode_id"])
    match = re.search(r"(\d+)\.json$", os.path.basename(fallback_path))
    if match:
        return match.group(1)
    raise ValueError(f"Could not infer episode_id from data or filename: {fallback_path}")


def canonicalize_whitespace(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_turn_text(text: str, max_chars: int) -> str:
    text = canonicalize_whitespace(text)
    return text if len(text) <= max_chars else (text[:max_chars] + " …(truncated)")


def render_annotation_prompt(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_maxim_label(raw: str) -> Optional[str]:
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


def write_annotated_episode(out_path: str, turns: List[Dict[str, Any]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)


def is_promotional_segment(text: str) -> bool:
    text = canonicalize_whitespace(text).lower()
    return any(re.search(pattern, text) for pattern in PROMO_PATTERNS)


def is_transcript_splice_artifact(prev_text: str, cur_text: str) -> bool:
    prev_text = canonicalize_whitespace(prev_text)
    cur_text = canonicalize_whitespace(cur_text)
    if not prev_text or not cur_text:
        return False
    prev_tail = prev_text[-1]
    cur_head = cur_text[0]
    return (
        prev_tail not in ".?!"
        and cur_head.islower()
        and cur_text.lower().startswith(SPLICE_START_PATTERNS)
    )


def has_discourse_continuation(text: str) -> bool:
    text = canonicalize_whitespace(text).lstrip(".,?!'\"-:;()[] ").lower()
    return text.startswith(DISCOURSE_CONTINUATION_PATTERNS)


def compute_lexical_overlap(prev_text: str, cur_text: str) -> float:
    stopwords = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "they",
        "them",
        "their",
        "about",
        "would",
        "could",
        "should",
    }
    prev_tokens = {
        tok for tok in re.findall(r"[a-zA-Z]{3,}", canonicalize_whitespace(prev_text).lower()) if tok not in stopwords
    }
    cur_tokens = {
        tok for tok in re.findall(r"[a-zA-Z]{3,}", canonicalize_whitespace(cur_text).lower()) if tok not in stopwords
    }
    if not prev_tokens or not cur_tokens:
        return 0.0
    return len(prev_tokens & cur_tokens) / min(len(prev_tokens), len(cur_tokens))


def is_reasonable_answer_continuation(prev_turn: Dict[str, Any], turn: Dict[str, Any]) -> bool:
    prev_text = canonicalize_whitespace(prev_turn.get("turn_text", ""))
    cur_text = canonicalize_whitespace(turn.get("turn_text", ""))
    move = canonicalize_whitespace(turn.get("conversation_move_label", ""))
    if not prev_text or not cur_text:
        return False
    prev_is_question = "?" in prev_text
    cur_len = len(cur_text.split())
    overlap = compute_lexical_overlap(prev_text, cur_text)
    return prev_is_question and cur_len >= 4 and (
        has_discourse_continuation(cur_text)
        or overlap >= 0.06
        or move in {"Answer", "Correction / Challenge"}
    )


def has_severe_manner_signal(text: str) -> bool:
    text = canonicalize_whitespace(text)
    low = text.lower()
    words = text.split()
    if not text:
        return False
    if text.strip().startswith((".", ",", "?", "'")) and len(words) <= 5:
        return True
    if "..." in text and len(words) <= 7:
        return True
    if len(words) <= 2 and any(ch.isalpha() for ch in text):
        return True
    if len(words) <= 8 and low.count(" uh") + low.count(" um") >= 2:
        return True
    if low.count(" uh") + low.count(" um") >= 4:
        return True
    return False


def is_fragmentary_continuation(prev_text: str, cur_text: str) -> bool:
    prev_text = canonicalize_whitespace(prev_text)
    cur_text = canonicalize_whitespace(cur_text)
    if not prev_text or not cur_text:
        return False
    cur_words = cur_text.split()
    if len(cur_words) > 4:
        return False
    prev_tail = prev_text[-1]
    cur_head = cur_text.lstrip(".,?!'\"-:;()[] ")
    if not cur_head:
        return False
    return prev_tail not in ".?!" and (cur_head[0].islower() or has_discourse_continuation(cur_text))


def is_recoverable_short_completion(prev_turn: Dict[str, Any], turn: Dict[str, Any]) -> bool:
    prev_text = canonicalize_whitespace(prev_turn.get("turn_text", ""))
    prev_move = canonicalize_whitespace(prev_turn.get("conversation_move_label", ""))
    cur_text = canonicalize_whitespace(turn.get("turn_text", ""))
    move = canonicalize_whitespace(turn.get("conversation_move_label", ""))
    if not prev_text or not cur_text:
        return False
    cur_len = len(cur_text.split())
    if cur_len > 6:
        return False
    overlap = compute_lexical_overlap(prev_text, cur_text)
    prev_requires_response = "?" in prev_text or prev_move in {
        "Clarification Request (Generic)",
        "Clarification Request (Specific)",
        "Correction / Challenge",
    }
    if is_fragmentary_continuation(prev_text, cur_text):
        return True
    if overlap >= 0.18:
        return True
    if prev_requires_response and move in {"Answer", "Correction / Challenge", "Stonewalling / Non-Response"} and cur_len <= 8:
        return True
    if prev_requires_response and "?" in cur_text and cur_len <= 8:
        return True
    if has_discourse_continuation(cur_text) and cur_len <= 5:
        return True
    if prev_requires_response and overlap >= 0.05:
        return True
    return False


def apply_gricean_postfilter(i: int, prev_turn: Dict[str, Any], turn: Dict[str, Any], label: Optional[str]) -> str:
    turn_type = canonicalize_whitespace(turn.get("turn_type_label"))
    move = canonicalize_whitespace(turn.get("conversation_move_label"))
    prev_move = canonicalize_whitespace(prev_turn.get("conversation_move_label"))
    prev_text = str(prev_turn.get("turn_text", ""))
    cur_text = str(turn.get("turn_text", ""))
    if i == 0:
        return "No Violation"
    if turn_type in NON_VIOLATION_TURN_TYPES:
        return "No Violation"
    if move in NON_VIOLATION_MOVES:
        return "No Violation"
    if is_promotional_segment(cur_text) or is_promotional_segment(prev_text):
        return "No Violation"
    if is_transcript_splice_artifact(prev_text, cur_text):
        return "No Violation"

    filtered = label or "No Violation"
    overlap = compute_lexical_overlap(prev_text, cur_text)
    cur_len = len(canonicalize_whitespace(cur_text).split())
    prev_requires_specific_response = "?" in prev_text or prev_move in {
        "Clarification Request (Generic)",
        "Clarification Request (Specific)",
        "Correction / Challenge",
    }

    if filtered == "Relation":
        if is_fragmentary_continuation(prev_text, cur_text):
            return "No Violation"
        if has_discourse_continuation(cur_text) or overlap >= 0.06 or is_reasonable_answer_continuation(prev_turn, turn):
            return "No Violation"
        if move in {"Topic Shift", "Stonewalling / Non-Response"} and not prev_requires_specific_response:
            return "No Violation"
        if move == "Assert / Elaborate" and prev_requires_specific_response and cur_len >= 8:
            return "No Violation"
        if move == "Assert / Elaborate" and not prev_requires_specific_response and cur_len >= 4:
            return "No Violation"
        if move == "Assert / Elaborate" and cur_len >= 6 and overlap >= 0.03:
            return "No Violation"
        if move == "Topic Shift" and overlap >= 0.04:
            return "No Violation"
        if move in {"Answer", "Correction / Challenge"} and cur_len >= 2:
            return "No Violation"
        if "?" in prev_text and cur_len >= 4:
            return "No Violation"

    if filtered == "Quantity":
        if move in {"Assert / Elaborate", "Answer"} and cur_len >= 12:
            return "No Violation"
        if has_discourse_continuation(cur_text) and overlap >= 0.12:
            return "No Violation"

    if filtered == "Manner":
        if is_recoverable_short_completion(prev_turn, turn):
            return "No Violation"
        if has_severe_manner_signal(cur_text):
            return "Manner"
        if cur_len >= 8 and cur_text.count("...") == 0 and cur_text.count("uh") + cur_text.count("um") <= 1:
            return "No Violation"
        if prev_requires_specific_response and move in {"Answer", "Stonewalling / Non-Response"} and cur_len <= 10:
            return "No Violation"
        if cur_len >= 3 and (overlap >= 0.08 or has_discourse_continuation(cur_text)):
            return "No Violation"
        if move in {"Answer", "Assert / Elaborate"} and cur_len >= 3 and overlap >= 0.04:
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

    ensure_output_dir(args.output_root)
    categories = resolve_category_set(
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
        files = collect_episode_paths(args.input_root, args.input_subdir, category)
        if args.max_episodes is not None and args.max_episodes > 0:
            files = files[: args.max_episodes]
        if not files:
            print(
                f"Skipping category={category}: no .json files found under "
                f"{os.path.join(args.input_root, args.input_subdir, category)}"
            )
            continue

        output_dir = os.path.join(args.output_root, args.output_subdir, category)
        ensure_output_dir(output_dir)

        for fp in tqdm(files, desc=f"{category}: Maxim Episodes"):
            obj = read_episode_json(fp)
            turns = extract_turn_sequence(obj)
            if not turns:
                continue

            episode_id = resolve_episode_id(turns, fp)
            out_path = os.path.join(output_dir, f"{episode_id}.json")

            prompts: List[str] = []
            for i, turn in enumerate(turns):
                prev_turn = turns[i - 1] if i > 0 else {}

                prev_text = truncate_turn_text(prev_turn.get("turn_text", ""), args.max_turn_chars)
                cur_text = truncate_turn_text(turn.get("turn_text", ""), args.max_turn_chars)

                user = MAXIM_PROMPT.format(
                    prev_type=canonicalize_whitespace(prev_turn.get("turn_type_label")) or "[NONE]",
                    prev_move=canonicalize_whitespace(prev_turn.get("conversation_move_label")) or "[NONE]",
                    prev_turn=prev_text or "[NONE]",
                    cur_type=canonicalize_whitespace(turn.get("turn_type_label")) or "[UNKNOWN]",
                    cur_move=canonicalize_whitespace(turn.get("conversation_move_label")) or "[UNKNOWN]",
                    cur_turn=cur_text or "[EMPTY]",
                )
                prompts.append(render_annotation_prompt(tokenizer, user))

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
                label = parse_maxim_label(out_text)
                prev_turn = turns[i - 1] if i > 0 else {}
                rec["maxim_violation_label"] = apply_gricean_postfilter(i, prev_turn, turn, label)
                if label is None:
                    rec["maxim_violation_label_error"] = out_text[:400]
                labeled_turns.append(rec)

            write_annotated_episode(out_path, labeled_turns)

    print(
        "Done. Updated per-episode files under: "
        f"{args.output_root}/{args.output_subdir}/{{category}}"
    )


if __name__ == "__main__":
    main()
