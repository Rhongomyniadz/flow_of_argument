import os
os.environ['HF_HOME'] = '/shared/3/edenzhang'

import argparse
import random
import logging
import csv
import re
import json
from typing import List, Dict
from tqdm import tqdm
from sporc import SPORCDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Model and tokenizer setup
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

# Generation helper
def transformer_generate(prompt: str, max_new_tokens: int = 20000, enable_thinking: bool = True) -> str:
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
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# Normalize raw model output to JSON fields
def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    cleaned = {}
    without_think = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.DOTALL)
    start, end = without_think.find('{'), without_think.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return cleaned
    try:
        parsed = json.loads(without_think[start:end+1])
    except json.JSONDecodeError:
        return cleaned
    for key in ("key_points_discussed_or_proposed", "key_points_assumed"):
        if key in parsed:
            cleaned[key] = parsed[key]
    return cleaned

# Word count utility
def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

# Main processing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", "-c", nargs='+', required=True,
        help="Podcast categories to load"
    )
    parser.add_argument(
        "--min_words", type=int, default=50,
        help="Minimum word count threshold for turns"
    )
    args = parser.parse_args()

    data_dir = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
    sporc.load_podcast_subset(categories=args.categories)

    # Use built-in search to find two-speaker episodes
    two_speaker_eps = sporc.search_episodes(min_speakers=2, max_speakers=2)
    if not two_speaker_eps:
        logger.warning("No two-speaker episodes found.")
        return

    # Randomly sample up to 10 episodes
    sample_eps = random.sample(two_speaker_eps, k=min(10, len(two_speaker_eps)))
    logger.info(f"Selected {len(sample_eps)} two-speaker episodes.")

    results = []
    for ep in tqdm(sample_eps, desc="Episodes", unit="episode"):
        title = ep.title
        turns = [t for t in ep.get_all_turns() if count_words(t.text.strip()) > args.min_words]
        for turn in tqdm(turns, desc=f"  Turns in {title}", leave=False, unit="turn"):
            text = turn.text.strip()
            prompt = (
                "Please analyze the following text and return a JSON with two keys:\n"
                "- key_points_discussed_or_proposed\n"
                "- key_points_assumed\n\n"
                f"Text:\n\"\"\"{text}\"\"\""
                "\n\nReturn the JSON without any additional text or explanation."
            )
            raw = transformer_generate(prompt)
            data = normalize_output(raw)
            results.append({
                "Podcast": title,
                "Speaker": ','.join(turn.speaker) if isinstance(turn.speaker, list) else turn.speaker,
                "Turn": text,
                "KeyPoints": '; '.join(data.get("key_points_discussed_or_proposed", [])),
                "Assumptions": '; '.join(data.get("key_points_assumed", []))
            })

    csv_path = 'results/news_sample.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Podcast", "Speaker", "Turn", "KeyPoints", "Assumptions"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Analysis complete. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
