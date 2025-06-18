import argparse
import gzip
import json
import logging
import re
from typing import List, Optional, Dict

import requests

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


def split_transcript_sliding(text: str,
                             window_size: int = 3,
                             stride: int = 2) -> List[str]:
    """
    Split `text` into overlapping chunks of `window_size` sentences,
    advancing by `stride` sentences each time.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    n = len(sentences)
    
    if n == 0:
        return chunks
    
    for start in range(0, n, stride):
        end = start + window_size
        window = sentences[start:end]
        if not window:
            break
        chunks.append(" ".join(window))
        if end >= n:
            break

    return chunks


def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    """
    Remove <think> blocks and parse out the two keys (arrays of strings).
    """
    without_think = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.DOTALL)

    json_start = without_think.find('{')
    json_end = without_think.rfind('}')
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return {}

    json_str = without_think[json_start : json_end + 1]

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
    output_path = "results/turn_level_stride2.json"

    ollama_client = OllamaGeneration()
    cleaned_entries = []
    total_count = 0
    success_count = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            if line_num > 10:
                break

            total_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Line {line_num}] Skipping: invalid JSON.")
                cleaned_entries.append({})
                continue

            turn_text = record.get("turnText", "").strip()
            if not turn_text:
                print(f"[Line {line_num}] Skipped: empty turnText.")
                cleaned_entries.append(record)
                continue

            print(f"[Line {line_num}] Processing speaker turn by {record.get('inferredSpeakerName', 'unknown')}")

            # Split into sliding windows
            chunks = split_transcript_sliding(turn_text, window_size=3, stride=2)
            print(f"  -> Turn split into {len(chunks)} sliding-window chunks.")

            all_discussed = []
            all_assumed = []
            any_success = False

            for idx, chunk in enumerate(chunks, start=1):
                print(f"    -> Chunk {idx}/{len(chunks)} length: {len(chunk)} chars")
                prompt = (
                    "Please analyze the following text and output a JSON object with two keys:\n"
                    "\"key_points_discussed_or_proposed\": an array of strings, each string being a main idea, "
                    "argument, or proposal explicitly presented in the text.\n"
                    "\"key_points_assumed\": an array of strings, each string being an underlying assumption or "
                    "implicit premise taken for granted by the text.\n\n"
                    f"Text:\n\"\"\"{chunk}\"\"\""
                )
                raw_response = ollama_client.generate(prompt)
                if not raw_response:
                    print(f"      -> Warning: empty response for chunk {idx}.")
                    continue

                cleaned = normalize_output(raw_response)
                if cleaned:
                    any_success = True
                    all_discussed.extend(cleaned.get("key_points_discussed_or_proposed", []))
                    all_assumed.extend(cleaned.get("key_points_assumed", []))
                    print(f"      -> Extracted {len(cleaned.get('key_points_discussed_or_proposed', []))} discussed and {len(cleaned.get('key_points_assumed', []))} assumed.")
                else:
                    print(f"      -> Warning: could not parse structured JSON.")

            if any_success:
                enriched_record = {
                    **record,
                    "key_points_discussed_or_proposed": all_discussed,
                    "key_points_assumed": all_assumed
                }
                cleaned_entries.append(enriched_record)
                success_count += 1
            else:
                print(f"  -> Warning: no successful extraction for this turn.")
                cleaned_entries.append(record)

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(cleaned_entries, outfile, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete. Total turns processed: {total_count}, Successful: {success_count}")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
