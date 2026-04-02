import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_DATA_ROOT = "data"
DEFAULT_INPUT_SUBDIR = "conversation_moves_labeled"
DEFAULT_OUTPUT_SUBDIR = "stance_labeled"
DEFAULT_CATEGORIES = ["business", "commentary", "news", "political", "religion", "sports"]
STANCE_LABEL_SCHEME = "signed_5pt_with_zero_v2"


class StanceAnnotationInterface:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
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


def ensure_output_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_analysis_categories(
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


def read_episode_record(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_turn_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("turns"), list):
        return payload["turns"]
    raise ValueError("Unrecognized JSON format: expected list or dict with 'turns' list.")


def resolve_episode_identifier(turn_records: List[Dict[str, Any]], fallback_path: str) -> str:
    if turn_records and turn_records[0].get("episode_id") is not None:
        return str(turn_records[0]["episode_id"])
    return os.path.splitext(os.path.basename(fallback_path))[0]


def write_labeled_episode(output_path: str, labeled_turns: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_turns, f, ensure_ascii=False, indent=2)


def whitespace_tokenize(text: str) -> List[str]:
    return text.strip().split()


def retain_last_k_tokens(text: str, k: int) -> str:
    if k <= 0:
        return ""
    tokens = whitespace_tokenize(text)
    if len(tokens) <= k:
        return text
    return " ".join(tokens[-k:])


def resolve_turn_text(turn_record: Dict[str, Any]) -> str:
    key = turn_record.get("chosen_text_key")
    if isinstance(key, str) and isinstance(turn_record.get(key), str):
        return turn_record[key]
    for cand in ("transcript", "turn_text", "text"):
        if isinstance(turn_record.get(cand), str):
            return turn_record[cand]
    return ""


def render_context_window(history_turns: List[Tuple[str, str]]) -> str:
    lines = []
    for speaker, text in history_turns:
        clean_text = text.replace("\n", " ").strip()
        lines.append(f"{speaker}: {clean_text}")
    return "\n".join(lines)


def build_episode_stance_prompts(
    turn_records: List[Dict[str, Any]],
    k: int,
    use_speaker: bool = True,
    stance_target: str = "immediately_previous_turn",
) -> List[str]:
    prompts: List[str] = []
    discourse_history: List[Tuple[str, str]] = []

    for turn_index, turn_record in enumerate(turn_records):
        current_text = resolve_turn_text(turn_record).replace("\n", " ").strip()
        current_speaker = turn_record.get("speaker_id") or turn_record.get("speaker") or "SPEAKER"
        current_speaker_label = current_speaker if use_speaker else "SPEAKER"

        preceding_turn_text = ""
        if turn_index > 0:
            if stance_target == "previous_nontrivial_turn":
                previous_index = turn_index - 1
                while previous_index >= 0:
                    candidate_text = resolve_turn_text(turn_records[previous_index]).replace("\n", " ").strip()
                    if candidate_text:
                        preceding_turn_text = candidate_text
                        break
                    previous_index -= 1
            else:
                preceding_turn_text = resolve_turn_text(turn_records[turn_index - 1]).replace("\n", " ").strip()

        context_window = retain_last_k_tokens(render_context_window(discourse_history), k)
        annotation_prompt = f"""You are doing stance detection in a conversation.

Task:
Given the conversation context and the current turn, rate the CURRENT turn's stance toward the IMMEDIATELY PRECEDING turn's content.

Scale (-5 to +5):
-5 = explicit, extreme disagreement or contradiction
-4 = strong disagreement
-3 = clear disagreement
-2 = mild disagreement or challenge
-1 = very slight pushback or correction
0 = neutral / unclear / unrelated / no stance
1 = very slight agreement or weak alignment
2 = mild agreement
3 = clear agreement
4 = strong agreement
5 = explicit, extreme agreement or support

Rules:
- Output ONLY a single integer from -5 to 5.
- If there is no clear agreement or disagreement, you MUST output 0.
- Treat 0 as the default choice. Move away from 0 only when the stance direction is genuinely clear.
- Be conservative with negative scores. Use a negative score only when the current turn actually pushes back against, corrects, disputes, or contradicts the immediately previous turn.
- Do NOT use negative scores for ordinary completions, elaborations, preferences, examples, topic continuation, joking banter, or answer fragments unless they clearly oppose the previous turn.
- Use 0 for neutral, unclear, unrelated, ad-like, narration-like, or no-response-to-target cases.
- Do NOT guess a direction from weak cues. If the stance is ambiguous, mixed, or only weakly implied, output 0.
- Reserve larger absolute values for clearer stance. Use -1, 0, or 1 for weak or ambiguous cases.
- If the current turn just continues, clarifies, or weakly aligns without clear support, prefer 0 or 1 rather than a stronger positive score.

Conversation context (most recent up to {k} tokens):
{context_window if context_window else "[NO PRIOR CONTEXT]"}

Immediately preceding turn (target):
{preceding_turn_text if preceding_turn_text else "[NO PRECEDING TURN]"}

Current turn:
{current_speaker_label}: {current_text}

Answer (single integer from -5 to 5):"""
        prompts.append(annotation_prompt)

        discourse_history.append((current_speaker_label, current_text))

    return prompts


def parse_stance_score(raw_output: str) -> int:
    raw_output = (raw_output or "").strip()
    match = re.search(r"(?<!\d)(-?[0-5])(?!\d)", raw_output)
    if match:
        return int(match.group(1))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input_subdir", type=str, default=DEFAULT_INPUT_SUBDIR)
    parser.add_argument("--output_subdir", type=str, default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--auto_categories",
        action="store_true",
        help="Discover all category folders under --input_root/--input_subdir.",
    )
    parser.add_argument("--k", type=int, default=512, help="Max prior-context whitespace tokens.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Prompt chunk size per llm.generate() call. Set <=0 to label a full episode in one batch.",
    )
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--download_dir", type=str, default="/shared/4/models")
    parser.add_argument("--max_tokens", type=int, default=16)
    args = parser.parse_args()

    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input root not found or not a directory: {args.input_root}")

    ensure_output_directory(args.output_root)
    analysis_categories = resolve_analysis_categories(
        args.input_root,
        args.categories,
        args.auto_categories,
        args.input_subdir,
    )
    if not analysis_categories:
        raise FileNotFoundError(
            f"No categories found under input_root={args.input_root} with input_subdir={args.input_subdir}"
        )

    annotator = StanceAnnotationInterface(
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

    output_dir = os.path.join(args.output_root, args.output_subdir, str(args.k))
    ensure_output_directory(output_dir)

    seen_output_paths: Dict[str, str] = {}
    for category in analysis_categories:
        episode_paths = collect_episode_paths(args.input_root, args.input_subdir, category)
        if not episode_paths:
            print(
                f"Skipping category={category}: no .json files found under "
                f"{os.path.join(args.input_root, args.input_subdir, category)}"
            )
            continue

        for episode_path in tqdm(episode_paths, desc=f"{category}: Stance Episodes"):
            episode_payload = read_episode_record(episode_path)
            turn_records = extract_turn_records(episode_payload)
            if not turn_records:
                continue

            episode_id = resolve_episode_identifier(turn_records, episode_path)
            output_path = os.path.join(output_dir, f"{episode_id}.json")
            previous_source_path = seen_output_paths.get(output_path)
            if previous_source_path is not None and previous_source_path != episode_path:
                raise ValueError(
                    "Multiple category inputs resolve to the same stance output path: "
                    f"{previous_source_path} and {episode_path} -> {output_path}"
                )
            seen_output_paths[output_path] = episode_path

            prompt_batch = build_episode_stance_prompts(turn_records, k=args.k)

            if args.batch_size and args.batch_size > 0:
                model_outputs: List[str] = []
                for batch_start in range(0, len(prompt_batch), args.batch_size):
                    model_outputs.extend(annotator.generate_batch(prompt_batch[batch_start:batch_start + args.batch_size]))
            else:
                model_outputs = annotator.generate_batch(prompt_batch)

            if len(model_outputs) != len(turn_records):
                raise RuntimeError(
                    f"Output length mismatch for episode_id={episode_id}: {len(model_outputs)} vs {len(turn_records)}"
                )

            labeled_turns: List[Dict[str, Any]] = []
            for turn_record, raw_model_output in zip(turn_records, model_outputs):
                labeled_turn = dict(turn_record)
                labeled_turn.setdefault("category", category)
                labeled_turn["stance_pt"] = parse_stance_score(raw_model_output)
                labeled_turn["stance_raw"] = raw_model_output
                labeled_turn["stance_context_k"] = args.k
                labeled_turn["stance_scheme"] = STANCE_LABEL_SCHEME
                labeled_turns.append(labeled_turn)

            write_labeled_episode(output_path, labeled_turns)

    print(
        "Done. Updated per-episode files under: "
        f"{args.output_root}/{args.output_subdir}/{args.k}"
    )


if __name__ == "__main__":
    main()
