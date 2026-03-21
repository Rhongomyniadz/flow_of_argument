import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_DATA_ROOT = "data"
DEFAULT_INPUT_SUBDIR = "conversation_moves_labeled"
DEFAULT_OUTPUT_SUBDIR = "stance_labeled"
DEFAULT_CATEGORIES = ["business", "commentary", "news", "political", "religion", "sports"]


class LLMInterface:
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
    if turns and turns[0].get("episode_id") is not None:
        return str(turns[0]["episode_id"])
    return os.path.splitext(os.path.basename(fallback_path))[0]


def save_episode_turns(out_path: str, turns: List[Dict[str, Any]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)


def simple_tokenize(text: str) -> List[str]:
    return text.strip().split()


def truncate_to_last_k_tokens(text: str, k: int) -> str:
    if k <= 0:
        return ""
    toks = simple_tokenize(text)
    if len(toks) <= k:
        return text
    return " ".join(toks[-k:])


def extract_turn_text(turn: Dict[str, Any]) -> str:
    key = turn.get("chosen_text_key")
    if isinstance(key, str) and isinstance(turn.get(key), str):
        return turn[key]
    for cand in ("transcript", "turn_text", "text"):
        if isinstance(turn.get(cand), str):
            return turn[cand]
    return ""


def format_history_for_prompt(history_turns: List[Tuple[str, str]]) -> str:
    lines = []
    for speaker, text in history_turns:
        clean_text = text.replace("\n", " ").strip()
        lines.append(f"{speaker}: {clean_text}")
    return "\n".join(lines)


def build_prompts_for_episode(
    turns: List[Dict[str, Any]],
    k: int,
    use_speaker: bool = True,
    stance_target: str = "immediately_previous_turn",
) -> List[str]:
    prompts: List[str] = []
    history_accum: List[Tuple[str, str]] = []

    for i, turn in enumerate(turns):
        cur_text = extract_turn_text(turn).replace("\n", " ").strip()
        cur_speaker = turn.get("speaker_id") or turn.get("speaker") or "SPEAKER"
        cur_label = cur_speaker if use_speaker else "SPEAKER"

        target_text = ""
        if i > 0:
            if stance_target == "previous_nontrivial_turn":
                j = i - 1
                while j >= 0:
                    candidate_text = extract_turn_text(turns[j]).replace("\n", " ").strip()
                    if candidate_text:
                        target_text = candidate_text
                        break
                    j -= 1
            else:
                target_text = extract_turn_text(turns[i - 1]).replace("\n", " ").strip()

        history_str = truncate_to_last_k_tokens(format_history_for_prompt(history_accum), k)
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

        history_accum.append((cur_label, cur_text))

    return prompts


def parse_stance_digit(raw: str) -> int:
    raw = (raw or "").strip()
    for ch in raw:
        if ch in "12345":
            return int(ch)
    return 3


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
    ap.add_argument("--k", type=int, default=512, help="Max prior-context whitespace tokens.")
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Prompt chunk size per llm.generate() call. Set <=0 to label a full episode in one batch.",
    )
    ap.add_argument("--model_name", type=str, default=MODEL_NAME)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--download_dir", type=str, default="/shared/4/models")
    ap.add_argument("--max_tokens", type=int, default=16)
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

    output_dir = os.path.join(args.output_root, args.output_subdir, str(args.k))
    ensure_dir(output_dir)

    seen_output_paths: Dict[str, str] = {}
    for category in categories:
        files = discover_episode_files(args.input_root, args.input_subdir, category)
        if not files:
            print(
                f"Skipping category={category}: no .json files found under "
                f"{os.path.join(args.input_root, args.input_subdir, category)}"
            )
            continue

        for fp in tqdm(files, desc=f"{category}: Stance Episodes"):
            obj = load_episode_json(fp)
            turns = extract_turns(obj)
            if not turns:
                continue

            episode_id = get_episode_id(turns, fp)
            out_path = os.path.join(output_dir, f"{episode_id}.json")
            prev_source = seen_output_paths.get(out_path)
            if prev_source is not None and prev_source != fp:
                raise ValueError(
                    "Multiple category inputs resolve to the same stance output path: "
                    f"{prev_source} and {fp} -> {out_path}"
                )
            seen_output_paths[out_path] = fp

            prompts = build_prompts_for_episode(turns, k=args.k)

            if args.batch_size and args.batch_size > 0:
                outputs: List[str] = []
                for i in range(0, len(prompts), args.batch_size):
                    outputs.extend(llm_if.generate_batch(prompts[i:i + args.batch_size]))
            else:
                outputs = llm_if.generate_batch(prompts)

            if len(outputs) != len(turns):
                raise RuntimeError(
                    f"Output length mismatch for episode_id={episode_id}: {len(outputs)} vs {len(turns)}"
                )

            labeled_turns: List[Dict[str, Any]] = []
            for turn, out_text in zip(turns, outputs):
                rec = dict(turn)
                rec.setdefault("category", category)
                rec["stance_5pt"] = parse_stance_digit(out_text)
                rec["stance_raw"] = out_text
                rec["stance_context_k"] = args.k
                labeled_turns.append(rec)

            save_episode_turns(out_path, labeled_turns)

    print(
        "Done. Updated per-episode files under: "
        f"{args.output_root}/{args.output_subdir}/{args.k}"
    )


if __name__ == "__main__":
    main()
