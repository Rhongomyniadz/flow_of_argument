import argparse
import csv
import gzip
import json
import logging
import re
import random
from collections import defaultdict
from typing import List, Optional, Dict

import requests

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OllamaGeneration:
    """Class for generating text using Ollama models via the local API."""

    def __init__(self, model_name: str = "qwen3:1.7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 20000) -> str:
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            request_data["system"] = system_prompt

        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=request_data, timeout=60)
            if resp.status_code != 200:
                logger.error(f"Ollama generation error {resp.status_code}: {resp.text}")
                return ""
            result = resp.json()
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Exception during Ollama generate call: {e}")
            return ""

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

def get_input_path(env: str) -> str:
    if env == "local":
        return "data/speakerTurnData.jsonl.gz"
    elif env == "cluster":
        return "/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/speakerTurnData.jsonl.gz"
    else:
        raise ValueError(f"Unknown environment: {env}. Choose 'local' or 'cluster'.")

def count_words(text: str) -> int:
    return len(re.findall(r'\w+', text))

def load_politics_episodes(env: str) -> set:
    """Returns the set of episode titles whose category1..10 includes 'politics'."""
    path = (
        "data/episodeLevelData.jsonl.gz"
        if env == "local"
        else "/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/episodeLevelData.jsonl.gz"
    )
    logging.info(f"Loading episode metadata from {path}")
    politics_eps = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            title = rec.get("epTitle") or rec.get("episodeTitle")
            for i in range(1, 11):
                cat = rec.get(f"category{i}")
                if cat and cat.lower() == "politics":
                    politics_eps.add(title)
                    break
    logging.info(f"Found {len(politics_eps)} politics-category episodes")
    return politics_eps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["local", "cluster"],
        default="local",
        help="Choose 'local' or 'cluster' to pick data paths"
    )
    args = parser.parse_args()
    input_path = get_input_path(args.env)
    output_csv = "results/episode_analysis.csv"

    politics_eps = load_politics_episodes(args.env)

    # 2) read speaker‐turn data and filter to politics episodes
    input_path = get_input_path(args.env)
    all_turns = []
    with gzip.open(input_path, "rt", encoding="utf-8") as infile:
        for line in infile:
            rec = json.loads(line)
            ep = rec.get("episodeTitle")
            if ep not in politics_eps:
                continue
            if count_words(rec.get("turnText", "")) < 30:
                continue
            all_turns.append(rec)

    if not all_turns:
        logging.warning("No speaker turns found for politics episodes!")
        return

    # 3) sample 10 turns
    sample_size = min(10, len(all_turns))
    sampled_turns = random.sample(all_turns, sample_size)
    logging.info(f"Sampling {sample_size} turns from {len(all_turns)} total politics turns")

    # 4) process each sampled turn with the model
    ollama = OllamaGeneration()
    results = []
    for turn in sampled_turns:
        text = turn["turnText"].strip()
        prompt = (
            "Please analyze the following text and output a JSON object with two keys:\n"
            "\"key_points_discussed_or_proposed\": [...],\n"
            "\"key_points_assumed\": [...]\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
        raw = ollama.generate(prompt)
        cleaned = normalize_output(raw)
        results.append({
            "Podcast": turn["episodeTitle"],
            "Turn Text": text,
            "Key Points": "; ".join(cleaned.get("key_points_discussed_or_proposed", [])),
            "Assumptions": "; ".join(cleaned.get("key_points_assumed", [])),
        })


    # Step 5: Write results to CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["Podcast", "Turn Text", "Turn Number", "Key Points", "Assumptions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Output saved to: {output_csv}")

if __name__ == "__main__":
    main()
