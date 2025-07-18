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

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    start, end = raw_response.find('{'), raw_response.rfind('}')
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(raw_response[start:end+1])
    except json.JSONDecodeError:
        return {}
    out: Dict[str, List[str]] = {}
    if "key_points_discussed_or_proposed" in parsed:
        out["key_points_discussed_or_proposed"] = parsed["key_points_discussed_or_proposed"]
    if "key_points_assumed" in parsed:
        out["key_points_assumed"] = parsed["key_points_assumed"]
    return out

def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

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
        self.params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            max_tokens=max_tokens
        )

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is not None:
            self.params.max_tokens = max_tokens
        outputs = self.llm.generate(prompt, self.params)
        return outputs[0].outputs[0].text.strip()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    parser = argparse.ArgumentParser(
        description="Analyze episodes for each of multiple podcast hosts"
    )
    parser.add_argument(
        "--hosts", "-H",
        nargs="+",
        required=True,
        help="One or more host names, e.g. -H 'Simon Shapiro' 'John Doe'"
    )
    parser.add_argument(
        "--gpu_id", "-g", type=int, default=0,
        help="CUDA GPU device ID for inference"
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
    args = parser.parse_args()

    data_dir = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    llm = LLMInterface(model_name="Qwen/Qwen3-8B", gpu_id=args.gpu_id)

    # Ensure output directory exists
    os.makedirs("results/hosts", exist_ok=True)

    for host in args.hosts:
        host_key = host.replace(" ", "_").lower()
        logger.info(f"Loading episodes for host={host!r}…")

        sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
        sporc.load_podcast_subset(hosts=[host])
        episodes = sporc.get_all_episodes()
        if not episodes:
            logger.warning(f"No episodes found for host {host!r}, skipping.")
            continue

        logger.info(f"Found {len(episodes)} episodes for {host!r}; sampling up to 30…")
        sample_eps = random.sample(episodes, k=min(30, len(episodes)))

        results: List[Dict] = []
        raw_results: List[Dict] = []

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
                raw_results.append({
                    "Host": host,
                    "Podcast": ep.title,
                    "WindowIndex": idx,
                    "RawOutput": raw
                })

                data = normalize_output(raw)
                speakers = {
                    s for t in win
                    for s in (t.speaker if isinstance(t.speaker, list) else [t.speaker])
                }
                results.append({
                    "Host": host,
                    "Podcast": ep.title,
                    "Speakers": ",".join(speakers),
                    "WindowIndex": idx,
                    "WindowText": combined,
                    "KeyPoints": data.get("key_points_discussed_or_proposed", []),
                    "Assumptions": data.get("key_points_assumed", [])
                })

        # Write out this host's results
        out_file = f"results/hosts/{host_key}.json"
        raw_file = f"results/hosts/{host_key}_raw.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Structured results saved → {out_file}")

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Raw results saved       → {raw_file}")

if __name__ == "__main__":
    main()
