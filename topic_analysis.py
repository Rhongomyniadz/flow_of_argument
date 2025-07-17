#!/usr/bin/env python3
"""
Analyze podcast episodes focused on George Floyd using SPoRC topic modeling and VLLM.
"""
import os
import argparse
import random
import json
import logging
import re
from typing import List, Dict, Optional
import pandas as pd
from sporc import SPORCDataset
from vllm import LLM, SamplingParams

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Normalization helper
def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    """
    Extracts the JSON object from the LLM's raw text and returns only the two target keys.
    """
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
        gpu_id: int = 0,
        min_p: float = 0.1,
        temperature: float = 0.3,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        top_k: int = 0,
        max_tokens: int = 2048
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
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


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    parser = argparse.ArgumentParser(
        description="Analyze podcast episodes about George Floyd via topic modeling"
    )
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA GPU device ID for inference")
    parser.add_argument("--min_words", type=int, default=50,
                        help="Minimum word count for speaker turns")
    parser.add_argument("--window_size", type=int, default=6,
                        help="Number of consecutive turns per analysis window")
    parser.add_argument("--stride", type=int, default=3,
                        help="Sliding window stride size")
    parser.add_argument("--sample_n", type=int, default=10,
                        help="Max number of episodes to sample")
    parser.add_argument("--topic_threshold", type=float, default=0.05,
                        help="Min topic proportion to select an episode")
    args = parser.parse_args()

    # ─── Load topic proportions from file ───────────────────────────────────────
    cols = ['row_id', 'url'] + [f'topic_{i}' for i in range(100)]
    doc_topics = pd.read_csv('doc_topics.txt', sep='\t', header=None, names=cols)

    # ─── Load topic keywords for interpretation ────────────────────────────────
    topic_keys = pd.read_csv(
        'topic_keys.txt', sep='\t', header=None,
        names=['topic_id', 'overall_prop', 'keywords']
    )

    # ─── Identify topics mentioning 'george' or 'floyd' ────────────────────────
    mask = (
        topic_keys['keywords'].str.contains('george', case=False) |
        topic_keys['keywords'].str.contains('floyd', case=False)
    )
    gf_topic_ids = topic_keys.loc[mask, 'topic_id'].tolist()
    if not gf_topic_ids:
        logger.error("No topics found matching 'george' or 'floyd'. Exiting.")
        return
    logger.info(f"Relevant topic IDs for George Floyd: {gf_topic_ids}")

    # ─── Filter episodes with high proportion on those topics ──────────────────
    topic_cols = [f'topic_{i}' for i in gf_topic_ids]
    gf_docs = doc_topics[doc_topics[topic_cols].max(axis=1) > args.topic_threshold]
    gf_urls = set(gf_docs['url'])
    logger.info(f"Selected {len(gf_urls)} episodes above threshold {args.topic_threshold}")

    # ─── Load SPoRC dataset in streaming/selective mode ────────────────────────
    data_dir = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
    sporc.load_podcast_subset()  # metadata only

    # ─── Match SPoRC episodes by URL ──────────────────────────────────────────
    all_eps = sporc.get_all_episodes()
    gf_eps = [ep for ep in all_eps if getattr(ep, 'url', None) in gf_urls]
    logger.info(f"Found {len(gf_eps)} matching SPoRC episodes")
    if not gf_eps:
        return

    # ─── Sample episodes ──────────────────────────────────────────────────────
    random.seed(42)
    sample_eps = random.sample(gf_eps, k=min(args.sample_n, len(gf_eps)))
    logger.info(f"Analyzing {len(sample_eps)} sampled episodes")

    # ─── Initialize LLM ───────────────────────────────────────────────────────
    llm = LLMInterface(model_name="Qwen/Qwen3-8B", gpu_id=args.gpu_id)
    results, raw_results = [], []

    # ─── Analyze each episode with sliding windows ────────────────────────────
    for ep in pd.util.testing.notnull(sample_eps) and sample_eps or []:
        turns = [t for t in ep.get_all_turns()
                 if count_words(t.text) > args.min_words]
        windows = [
            turns[i : i + args.window_size]
            for i in range(0, len(turns) - args.window_size + 1, args.stride)
        ]
        for idx, win in enumerate(windows):
            text_block = "\n\n".join(f"{t.speaker}: {t.text.strip()}" for t in win)
            prompt = f"""
SYSTEM:
You are an expert podcast conversation analyst.

TASK:
  • Given exactly six consecutive turns, extract two arrays:
    1) "key_points_discussed_or_proposed"
    2) "key_points_assumed"

OUTPUT only valid JSON matching this schema:
{{
  "key_points_discussed_or_proposed": [ string, … ],
  "key_points_assumed":           [ string, … ]
}}

Now analyze Window #{idx} from episode \"{ep.title}\":
{text_block}
"""
            raw = llm.generate_response(prompt)
            raw_results.append({
                "Podcast": ep.title,
                "WindowIndex": idx,
                "RawOutput": raw
            })
            data = normalize_output(raw)
            speakers = {s for t in win for s in (t.speaker if isinstance(t.speaker, list) else [t.speaker])}
            results.append({
                "Podcast": ep.title,
                "Speakers": ",".join(speakers),
                "WindowIndex": idx,
                "WindowText": text_block,
                "KeyPoints": data.get("key_points_discussed_or_proposed", []),
                "Assumptions": data.get("key_points_assumed", [])
            })

    # ─── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    base = "george_floyd_topic_analysis"
    with open(f"results/{base}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(f"results/{base}_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    logger.info("Analysis complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
