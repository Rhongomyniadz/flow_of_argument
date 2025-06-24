import argparse
import random
import json
import logging
from sporc import SPORCDataset
import re
from typing import List, Optional, Dict

import requests

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OllamaGeneration:
    """Class for generating text using Ollama models via the local API."""

    def __init__(self, model_name: str = "qwen3:1.7b", base_url: str = "http://localhost:8889"):
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
        logging.warning("No speaker turns found for specified categories!")
        return

    # Sample up to 10 turns
    sample_size = min(10, len(all_turns))
    sampled = random.sample(all_turns, sample_size)
    logging.info(f"Sampling {sample_size} turns from {len(all_turns)} total turns")

    # Process each sampled turn with the model
    ollama = OllamaGeneration()
    for rec in sampled:
        text = rec["turnText"].strip()
        prompt = (
            "Please analyze the following text and return a JSON with:\n"
            "- key_points_discussed_or_proposed\n"
            "- key_points_assumed\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
        raw = ollama.generate(prompt)
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
