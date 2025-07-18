import os
import argparse
import random
import json
import logging
import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional
from tqdm import tqdm
from sporc import SPORCDataset
from vllm import LLM, SamplingParams

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    """Extract the JSON payload from the model’s raw text and return only the two lists."""
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

def count_words(text: str) -> int:
    """Count words by matching alphanumeric sequences."""
    return len(re.findall(r"\w+", text))

class LLMInterface:
    """Light wrapper around vLLM for generation with fixed sampling parameters."""
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
        max_tokens: int = 2048
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            download_dir="/shared/4/models",
            device=f"cuda:{gpu_id}"
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            max_tokens=max_tokens
        )

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is not None:
            self.sampling_params.max_tokens = max_tokens
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text.strip()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    parser = argparse.ArgumentParser(
        description="Analyze the top N podcast hosts via sliding-window LLM summarization."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/",
        help="Path to SPoRC processed data"
    )
    parser.add_argument(
        "--top_n_hosts", type=int, default=5,
        help="Number of hosts (by episode count) to analyze"
    )
    parser.add_argument(
        "--min_words", type=int, default=50,
        help="Minimum word count threshold for including a turn"
    )
    parser.add_argument(
        "--window_size", type=int, default=6,
        help="Number of turns per sliding window"
    )
    parser.add_argument(
        "--stride", type=int, default=3,
        help="Stride size for the sliding window"
    )
    parser.add_argument(
        "--gpu_id", "-g", type=int, default=0,
        help="CUDA GPU device ID for LLM inference"
    )
    args = parser.parse_args()

    # Initialize dataset (streaming mode)
    sporc = SPORCDataset(local_data_dir=args.data_dir, streaming=True)

    # -------------------------------------------------------------------------
    # STEP 1: Load all episodes (no host filter) and group by host
    # -------------------------------------------------------------------------
    logger.info("Loading all episodes (no host filter)…")
    sporc.load_podcast_subset()  # do NOT pass hosts=None
    all_eps = sporc.get_all_episodes()
    if not all_eps:
        logger.error("No episodes found in the dataset—exiting.")
        return

    episodes_by_host: Dict[str, List] = defaultdict(list)
    for ep in all_eps:
        # adjust if your Episode object stores host differently
        host = getattr(ep, "host", None) or ep.meta.get("host")
        episodes_by_host[host].append(ep)

    host_counts = Counter({h: len(eps) for h, eps in episodes_by_host.items()})
    top_hosts = [h for h, _ in host_counts.most_common(args.top_n_hosts)]
    logger.info(f"Top {args.top_n_hosts} hosts by episode count: {top_hosts}")

    # -------------------------------------------------------------------------
    # STEP 2: Set up LLM
    # -------------------------------------------------------------------------
    llm = LLMInterface(model_name="Qwen/Qwen3-8B", gpu_id=args.gpu_id)

    all_results, all_raw = [], []

    # -------------------------------------------------------------------------
    # STEP 3: Analyze each top host separately
    # -------------------------------------------------------------------------
    for host in top_hosts:
        logger.info(f"Reloading episodes for host {host!r}…")
        sporc.load_podcast_subset(hosts=[host])
        host_eps = sporc.get_all_episodes()
        logger.info(f"  → {len(host_eps)} episodes for {host!r}; sampling up to 30…")
        sample_eps = random.sample(host_eps, k=min(30, len(host_eps)))

        for ep in tqdm(sample_eps, desc=f"Host {host}", unit="episode"):
            turns = [
                t for t in ep.get_all_turns()
                if count_words(t.text.strip()) > args.min_words
            ]
            windows = [
                turns[i : i + args.window_size]
                for i in range(0, len(turns) - args.window_size + 1, args.stride)
            ]

            for idx, win in enumerate(windows):
                combined = "\n\n".join(f"{t.speaker}: {t.text.strip()}" for t in win)
                prompt = f"""
You are an expert podcast conversation analyst.
You must output *only* valid JSON matching exactly this schema:

{{
  "key_points_discussed_or_proposed": [ string, … ],
  "key_points_assumed":           [ string, … ]
}}

Analyze Window #{idx} of "{ep.title}" (host: {host}):
{combined}
"""
                raw = llm.generate_response(prompt)
                all_raw.append({
                    "Host": host,
                    "Podcast": ep.title,
                    "WindowIndex": idx,
                    "RawOutput": raw
                })

                data = normalize_output(raw)
                speakers = {
                    s
                    for t in win
                    for s in (t.speaker if isinstance(t.speaker, list) else [t.speaker])
                }
                all_results.append({
                    "Host": host,
                    "Podcast": ep.title,
                    "Speakers": ",".join(speakers),
                    "WindowIndex": idx,
                    "WindowText": combined,
                    "KeyPoints": data.get("key_points_discussed_or_proposed", []),
                    "Assumptions": data.get("key_points_assumed", [])
                })

    # -------------------------------------------------------------------------
    # STEP 4: Write outputs
    # -------------------------------------------------------------------------
    os.makedirs("results/hosts", exist_ok=True)
    summary_path = f"results/hosts/top_{args.top_n_hosts}_hosts_summary.json"
    raw_path     = f"results/hosts/top_{args.top_n_hosts}_hosts_raw.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Structured results → {summary_path}")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, indent=2)
    logger.info(f"Raw LLM outputs → {raw_path}")

if __name__ == "__main__":
    main()
