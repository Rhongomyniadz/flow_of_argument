import argparse
import random
import json
import logging
from sporc import SPORCDataset
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load Qwen3 model and tokenizer
MODEL_NAME = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Helper to generate using Transformers
def transformer_generate(prompt: str, max_new_tokens: int = 2048, enable_thinking: bool = True) -> str:
    # prepare chat template messages
    messages = [{"role": "user", "content": prompt}]
    # apply chat template (handles <think> blocks)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )[0][len(inputs.input_ids[0]):]
    # decode full output
    gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return gen_text.strip()


def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    without_think = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.DOTALL)
    json_start = without_think.find('{')
    json_end = without_think.rfind('}')
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return {}
    json_str = without_think[json_start:json_end + 1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return {}
    cleaned = {}
    if "key_points_discussed_or_proposed" in parsed:
        cleaned["key_points_discussed_or_proposed"] = parsed["key_points_discussed_or_proposed"]
    if "key_points_assumed" in parsed:
        cleaned["key_points_assumed"] = parsed["key_points_assumed"]
    return cleaned


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", "-c",
        nargs='+',
        required=True,
        help="List of podcast categories to load"
    )
    args = parser.parse_args()

    # Initialize SPORC in streaming mode
    data_dir = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)

    # Search episodes by category
    episodes = []
    seen = set()
    for cat in args.categories:
        logger.info(f"Searching episodes in category: {cat}")
        found = sporc.search_episodes(category=cat)
        for ep in found:
            if ep.title not in seen:
                episodes.append(ep)
                seen.add(ep.title)
    logger.info(f"Collected {len(episodes)} unique episodes for categories {args.categories}")

    # Gather turns
    all_turns = []
    for ep in episodes:
        for turn in ep.get_all_turns():
            if count_words(turn.text) < 30:
                continue
            rec = {
                "episodeTitle": ep.title,
                "turnText": turn.text,
                "speaker": turn.speaker,
                "startTime": turn.start_time,
                "duration": turn.duration
            }
            all_turns.append(rec)

    if not all_turns:
        logger.warning("No speaker turns found for specified categories!")
        return

    sample_size = min(10, len(all_turns))
    sampled = random.sample(all_turns, sample_size)
    logger.info(f"Sampling {sample_size} turns from {len(all_turns)} total turns")

    # Process turns with Transformers
    for rec in sampled:
        text = rec["turnText"].strip()
        prompt = (
            "Please analyze the following text and return a JSON with:\n"
            "- key_points_discussed_or_proposed\n"
            "- key_points_assumed\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
        raw = transformer_generate(prompt)
        cleaned = normalize_output(raw)
        output = {
            "Podcast": rec["episodeTitle"],
            "Speaker": rec["speaker"],
            "Turn": text,
            "KeyPoints": cleaned.get("key_points_discussed_or_proposed", []),
            "Assumptions": cleaned.get("key_points_assumed", [])
        }
        print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
