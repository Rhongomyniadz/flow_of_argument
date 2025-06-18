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

def main():
    parser = argparse.ArgumentParser(description="Run speaker turn processing with environment option.")
    parser.add_argument(
        "--env",
        type=str,
        choices=["local", "cluster"],
        default="local",
        help="Environment where code is run: 'local' (default) or 'cluster'"
    )
    args = parser.parse_args()
    input_path = get_input_path(args.env)
    output_csv = "results/episode_analysis.csv"

    ollama_client = OllamaGeneration()
    min_word_threshold = 30

    episode_buffer = defaultdict(list)
    episode_speakers = defaultdict(set)

    with gzip.open(input_path, "rt", encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
                ep_title = record.get("episodeTitle")
                speaker_ids = record.get("speaker", [])
                if not ep_title:
                    continue
                episode_buffer[ep_title].append(record)
                episode_speakers[ep_title].update(speaker_ids)
            except Exception:
                continue

            if len(episode_buffer) >= 30:
                break

    # Filter down to those with 2 speakers and at least 1 long turn
    filtered = []
    for ep, turns in episode_buffer.items():
        if len(episode_speakers[ep]) != 2:
            continue
        if any(count_words(t.get("turnText", "")) >= 30 for t in turns):
            filtered.append((ep, turns))

    selected_episodes = random.sample(filtered, min(10, len(filtered)))

    results = []

    # Process each qualifying turn in selected episodes
    for ep_title, turns in selected_episodes:
        for turn in turns:
            turn_text = turn.get("turnText", "").strip()
            if count_words(turn_text) < min_word_threshold:
                continue

            prompt = (
                "Please analyze the following text and output a JSON object with two keys:\n"
                "\"key_points_discussed_or_proposed\": an array of strings, each string being a main idea, "
                "argument, or proposal explicitly presented in the text.\n"
                "\"key_points_assumed\": an array of strings, each string being an underlying assumption or "
                "implicit premise taken for granted by the text.\n\n"
                f"Text:\n\"\"\"{turn_text}\"\"\""
            )

            raw_response = ollama_client.generate(prompt)
            cleaned = normalize_output(raw_response)

            results.append({
                "Podcast": ep_title,
                "Turn Text": turn_text,
                "Turn Number": turn.get("turnCount", ""),
                "Key Points": "; ".join(cleaned.get("key_points_discussed_or_proposed", [])),
                "Assumptions": "; ".join(cleaned.get("key_points_assumed", []))
            })

    # Step 5: Write results to CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["Podcast", "Turn Text", "Turn Number", "Key Points", "Assumptions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Processed {len(results)} long turns from {len(selected_episodes)} episodes.")
    print(f"📁 Output saved to: {output_csv}")

if __name__ == "__main__":
    main()
