import os
os.environ['HF_HOME'] = '/shared/3/edenzhang'

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

# Load Qwen3 model and tokenizer, using shared HF_HOME as cache location
MODEL_NAME = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=os.environ['HF_HOME']
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=os.environ['HF_HOME'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Helper to generate using Transformers
def transformer_generate(prompt: str, max_new_tokens: int = 2048, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": prompt}]
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
    parser = argparse.ArgumentParser(
        description="Sample speaker turns via Transformers with Qwen3"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs='+',
        required=True,
        help="List of podcast categories to load"
    )
    args = parser.parse_args()

    # Initialize SPORC in streaming mode
    d = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=d, streaming=True)
    sporc.load_podcast_subset(categories=args.categories)

    episodes = sporc.get_all_episodes()
    logger.info(f"Loaded {len(episodes)} episodes for categories {args.categories}")

    # Gather all turns from these episodes
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

    # Sample up to 10 turns
    sample_size = min(10, len(all_turns))
    sampled = random.sample(all_turns, sample_size)
    logger.info(f"Sampling {sample_size} turns from {len(all_turns)} total turns")

    # Process turns with Transformers
    results = []
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
        results.append(output)

    output_path = 'results/news_sample.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    main()
