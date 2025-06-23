import argparse
import random
import json
import logging
from sporc import SPORCDataset
import re
from typing import List, Optional, Dict

from vllm import LLM, SamplingParams

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VLLMGeneration:
    """Class for generating text using vLLM's Python API."""

    def __init__(self,
                 model_name: str = "qwen3:1.7b",
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1):
        # Initialize the vLLM backend
        self.client = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size
        )

    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 20000) -> str:
        # If you have a separate system prompt, prepend it
        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

        # vLLM expects a list of prompts, but we're doing one at a time here
        try:
            outputs = self.client.generate(
                [full_prompt],
                params=sampling_params
            )
            # outputs is a generator of BatchLLMOutput; grab the first
            for batch_output in outputs:
                # generated_text includes both prompt + completion;
                # if you only want the completion, you can subtract len(full_prompt)
                return batch_output.outputs[0].text[len(full_prompt):].strip()
        except Exception as e:
            logger.error(f"Exception during vLLM generate call: {e}")
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
    vllm_gen = VLLMGeneration()
    for rec in sampled:
        text = rec["turnText"].strip()
        prompt = (
            "Please analyze the following text and return a JSON with:\n"
            "- key_points_discussed_or_proposed\n"
            "- key_points_assumed\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
        raw = vllm_gen.generate(prompt)
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
