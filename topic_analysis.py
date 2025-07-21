import os
import argparse
import random
import json
import logging
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse

import pandas as pd
from sporc import SPORCDataset
from vllm import LLM, SamplingParams

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def canonical_to_raw(canonical_url: str) -> str:
    """
    Turn a Buzzsprout mp3_url like
      https://www.buzzsprout.com/783020/4252475-best-of-singout-speakout-no-3.mp3
    into the SPoRC merged‐slug form:
      /www.buzzsprout.com/o3/httpswww.buzzsprout.com7830204252475bestofsingoutspeakoutno3.mp3MERGED
    """
    p = urlparse(canonical_url)
    domain = p.netloc            # e.g. "www.buzzsprout.com"
    scheme = p.scheme            # "https"
    # path = "783020/4252475-best-of-singout-speakout-no-3.mp3"
    path = p.path.lstrip("/")
    # remove all slashes and hyphens
    collapsed = path.replace("/", "").replace("-", "")
    # assemble the host+domain without "://"
    host_noslash = f"{scheme}{domain}"  # "httpswww.buzzsprout.com"
    # build the raw slug + extension
    raw_body = f"{host_noslash}{collapsed}"
    return f"/{domain}/o3/{raw_body}MERGED"

def normalize_output(raw_response: str) -> Dict[str, List[str]]:
    start, end = raw_response.find('{'), raw_response.rfind('}')
    if start == -1 or end == -1 or end <= start:
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

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    parser = argparse.ArgumentParser(
        description="Analyze podcast episodes about George Floyd via topic modeling"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--min_words", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--sample_n", type=int, default=10)
    parser.add_argument("--topic_threshold", type=float, default=0.001)
    args = parser.parse_args()

    # ─── Load topic proportions ───────────────────────────────────────────────
    cols = ['row_id', 'url'] + [f'topic_{i}' for i in range(100)]
    doc_topics = pd.read_csv('doc_topics.txt', sep='\t', header=None, names=cols)

    # ─── Load topic keywords ────────────────────────────────────────────────
    topic_keys = pd.read_csv(
        'topic_keys.txt', sep='\t', header=None,
        names=['topic_id', 'overall_prop', 'keywords']
    )

    # ─── Find George/Floyd topics ───────────────────────────────────────────
    mask = (
        topic_keys['keywords'].str.contains('george', case=False) |
        topic_keys['keywords'].str.contains('floyd', case=False)
    )
    gf_ids = topic_keys.loc[mask, 'topic_id'].tolist()
    if not gf_ids:
        logger.error("No 'george' or 'floyd' topics found.")
        return
    topic_cols = [f'topic_{i}' for i in gf_ids]

    # ─── Pre-filter docs by threshold ───────────────────────────────────────
    filtered = doc_topics[doc_topics[topic_cols].max(axis=1) > args.topic_threshold]
    logger.info(f"{len(filtered)} docs above topic threshold")

    # ─── Load SPoRC episodes ────────────────────────────────────────────────
    data_dir = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/'
    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
    sporc.load_podcast_subset()
    all_eps = sporc.get_all_episodes()

    # build raw_url → episode map
    raw2ep = { canonical_to_raw(ep.mp3_url): ep for ep in all_eps }

    # ─── Keep only docs whose raw‐url matches an episode ─────────────────────
    matched = filtered[filtered['url'].isin(raw2ep)]
    logger.info(f"{len(matched)} docs match SPoRC raw URLs")

    eps = [ raw2ep[r] for r in matched['url'] ]
    if not eps:
        logger.warning("No episodes found after mapping. Exiting.")
        return

    # ─── Sample and analyze ─────────────────────────────────────────────────
    random.seed(42)
    sample_eps = random.sample(eps, k=min(args.sample_n, len(eps)))
    llm = LLMInterface(model_name="Qwen/Qwen3-8B", gpu_id=args.gpu_id)

    results, raw_results = [], []
    for ep in sample_eps:
        turns = [t for t in ep.get_all_turns() if count_words(t.text) > args.min_words]
        windows = [
            turns[i:i+args.window_size]
            for i in range(0, len(turns)-args.window_size+1, args.stride)
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

Now analyze Window #{idx} from episode "{ep.epTitle}":
{text_block}
"""
            raw = llm.generate_response(prompt)
            raw_results.append({
                "Podcast": ep.epTitle,
                "WindowIndex": idx,
                "RawOutput": raw
            })
            data = normalize_output(raw)
            speakers = {
                s
                for t in win
                for s in (t.speaker if isinstance(t.speaker, list) else [t.speaker])
            }
            results.append({
                "Podcast":     ep.epTitle,
                "Speakers":    ",".join(speakers),
                "WindowIndex": idx,
                "WindowText":  text_block,
                "KeyPoints":   data.get("key_points_discussed_or_proposed", []),
                "Assumptions": data.get("key_points_assumed", [])
            })

    os.makedirs("results", exist_ok=True)
    base = "george_floyd_topic_analysis"
    with open(f"results/{base}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(f"results/{base}_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
