import argparse
import random
import json
import logging
from sporc import SPORCDataset
import re
from typing import List, Optional, Dict

# vllm-based LLM interface (do not modify)
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
import os

class LLMInterface:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", temperature: float = 0.1, top_p: float = 0.95, gpu_id: int = 0, gpu_memory_utilization: float = 0.9):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            device=f"cuda:{gpu_id}"
        )
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens:
            self.sampling_params.max_tokens = max_tokens
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text.strip()

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    """
    Remove <think> blocks and parse out the two keys (arrays of strings).
    """
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
    return len(re.findall(r'\w+', text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", "-c",
        nargs='+',
        required=True,
        help="List of podcast categories to load"
    )
    args = parser.parse_args()

    d = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=d, streaming=True)
    sporc.load_podcast_subset(categories=args.categories)

    episodes = sporc.get_all_episodes()
    logging.info(f"Loaded {len(episodes)} episodes for categories {args.categories}")

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
        logging.warning("No speaker turns found for specified categories!")
        return

    sample_size = min(10, len(all_turns))
    sampled = random.sample(all_turns, sample_size)
    logging.info(f"Sampling {sample_size} turns from {len(all_turns)} total turns")

    # Instantiate the vllm interface
    llm = LLMInterface(model_name="Qwen/Qwen3-0.6B", gpu_id=1)

    for rec in sampled:
        text = rec["turnText"].strip()
        prompt = (
            "Please analyze the following text and return a JSON with:\n"
            "- key_points_discussed_or_proposed\n"
            "- key_points_assumed\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
        raw = llm.generate_response(prompt)
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
