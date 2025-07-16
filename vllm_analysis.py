import os
import argparse
import random
import json
import logging
import re
from typing import List, Dict, Optional
from tqdm import tqdm
from sporc import SPORCDataset
from vllm import LLM, SamplingParams

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Normalization helper
def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    # without_think = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.DOTALL)
    start, end = raw_response.find('{'), raw_response.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(raw_response[start:end+1])
    except json.JSONDecodeError:
        return {}
    cleaned: Dict[str, List[str]] = {}
    if "key_points_discussed_or_proposed" in parsed:
        cleaned["key_points_discussed_or_proposed"] = parsed["key_points_discussed_or_proposed"]
    if "key_points_assumed" in parsed:
        cleaned["key_points_assumed"] = parsed["key_points_assumed"]
    return cleaned

# Word count utility
def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

# VLLM wrapper
class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        min_p: float = 0.1,
        temperature: float = 0.3,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.9,
        top_k: int = 0,
        max_toekns: int = 2048
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            download_dir="/shared/4/models",
            device=f"cuda:{gpu_id}"
        )
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, 
                                              min_p=min_p,
                                              repetition_penalty=repetition_penalty, 
                                              top_k=top_k,
                                              max_tokens=max_toekns)

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is not None:
            self.sampling_params.max_tokens = max_tokens
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text.strip()


def main():
    # Ensure CUDA ordering
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hosts", "-H",
        nargs='+',
        required=True,
        help="List of podcast hosts to analyze"
    )
    parser.add_argument(
        "--min_words", type=int, default=50,
        help="Minimum word count threshold for turns"
    )
    parser.add_argument(
        "--window_size", type=int, default=6,
        help="Number of turns per sliding window"
    )
    parser.add_argument(
        "--stride", type=int, default=3,
        help="Stride size for sliding window"
    )
    args = parser.parse_args()

    data_dir = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
    
    # Load podcasts for specified hosts
    sporc.load_podcast_subset(hosts=args.hosts)
    logger.info(f"Loaded podcasts for hosts: {', '.join(args.hosts)}")
    logger.info("Host names in dataset:")
    all_hosts = set()
    for podcast in sporc.get_all_podcasts():
        all_hosts.update(podcast.host_names)
    logger.info(f"Available hosts: {', '.join(sorted(all_hosts))}")

    # Get all episodes first
    all_episodes = sporc.search_episodes()
    logger.info(f"Total episodes found: {len(all_episodes)}")
    
    episodes_to_process = all_episodes
    logger.info(f"Processing all {len(episodes_to_process)} episodes")

    # Sample episodes
    sample_eps = random.sample(episodes_to_process, k=min(30, len(episodes_to_process)))
    logger.info(f"Selected {len(sample_eps)} episodes for analysis.")

    llm = LLMInterface(model_name="Qwen/Qwen3-8B", gpu_id=0)

    for host in args.hosts:
        # Filter episodes for current host
        host_episodes = [ep for ep in sample_eps if host in ep.host_names]
        logger.info(f"Found {len(host_episodes)} episodes for host '{host}'")
        if not host_episodes:
            logger.warning(f"No episodes found for host '{host}'. Check host name spelling.")
            continue
        
        # Log episode details
        for ep in host_episodes:
            logger.info(f"Episode: {ep.title}")
            logger.info(f"  Hosts: {', '.join(ep.host_names)}")
            logger.info(f"  Speakers: {', '.join(ep.main_ep_speakers)}")
        
        host_results: List[Dict] = []
        host_raw_results: List[Dict] = []

        for ep in tqdm(host_episodes, desc=f"Episodes for {host}", unit="episode"):
            turns = [t for t in ep.get_all_turns() if count_words(t.text.strip()) > args.min_words]
            windows = [turns[i:i+args.window_size]
                      for i in range(0, len(turns) - args.window_size + 1, args.stride)]
            
            for idx, win in enumerate(tqdm(windows, desc=f" Windows in {ep.title}", leave=False)):
                combined = "\n\n".join(f"{t.speaker}: {t.text.strip()}" for t in win)
                prompt = f"""
                You are a podcast conversation analyst. Your job is to read a block of six consecutive speaker turns, identify (1) the main points discussed or proposed, and (2) any implicit assumptions underlying the speakers' remarks. You must respond only with a single JSON object—nothing else.

                JSON schema:
                {{
                "key_points_discussed_or_proposed": [string, …],
                "key_points_assumed": [string, …]
                }}

                Now analyze Window #{idx}:
                {combined}

                Respond with a valid JSON that strictly follows the schema above.
                """
                raw = llm.generate_response(prompt)
                # Print raw response to terminal
                logger.info(f"\nWindow #{idx} Raw LLM Output:\n{raw}\n{'-'*80}")
                
                # Save raw response
                host_raw_results.append({
                    "Podcast": ep.title,
                    "WindowIndex": idx,
                    "RawOutput": raw
                })
                data = normalize_output(raw)
                spks = set(s for t in win for s in (t.speaker if isinstance(t.speaker, list) else [t.speaker]))
                host_results.append({
                    "Podcast": ep.title,
                    "Speakers": ','.join(spks),
                    "WindowIndex": idx,
                    "WindowText": combined,
                    "KeyPoints": data.get("key_points_discussed_or_proposed", []),
                    "Assumptions": data.get("key_points_assumed", [])
                })

        # Save results for current host
        host_filename = f"results/{host.replace(' ', '_').lower()}.json"
        with open(host_filename, 'w', encoding='utf-8') as f:
            json.dump(host_results, f, indent=2)
        logger.info(f"Processed results for {host} saved to {host_filename}")

        # Save raw results for current host
        raw_host_filename = f"results/{host.replace(' ', '_').lower()}_raw.json"
        with open(raw_host_filename, 'w', encoding='utf-8') as f:
            json.dump(host_raw_results, f, indent=2)
        logger.info(f"Raw LLM outputs for {host} saved to {raw_host_filename}")

if __name__ == "__main__":
    main()
