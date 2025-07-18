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
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_output(raw: str) -> Dict[str, List[str]]:
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(raw[start:end+1])
    except json.JSONDecodeError:
        return {}
    out = {}
    if "key_points_discussed_or_proposed" in obj:
        out["key_points_discussed_or_proposed"] = obj["key_points_discussed_or_proposed"]
    if "key_points_assumed" in obj:
        out["key_points_assumed"] = obj["key_points_assumed"]
    return out

def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        gpu_id: int = 0,
        temperature: float = 0.3,
        top_p: float = 0.8,
        min_p: float = 0.1,
        repetition_penalty: float = 1.05,
        top_k: int = 0,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.9,
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            download_dir="/shared/4/models",
            device=f"cuda:{gpu_id}",
        )
        self.params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is not None:
            self.params.max_tokens = max_tokens
        out = self.llm.generate(prompt, self.params)
        return out[0].outputs[0].text.strip()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    p = argparse.ArgumentParser(
        description="Analyze top-N podcast hosts with a sliding-window LLM pipeline"
    )
    p.add_argument("--data_dir",    type=str, default="/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/")
    p.add_argument("--top_n_hosts", type=int, default=5)
    p.add_argument("--min_words",   type=int, default=50)
    p.add_argument("--window_size", type=int, default=6)
    p.add_argument("--stride",      type=int, default=3)
    p.add_argument("--gpu_id",      type=int, default=0)
    args = p.parse_args()

    # ─── 1) COUNT EPISODES PER HOST ─────────────────────────────────────────────
    logger.info("Loading *all* episodes for host counting…")
    sporc_all = SPORCDataset(local_data_dir=args.data_dir, streaming=True)
    sporc_all.load_podcast_subset()
    all_eps = sporc_all.get_all_episodes()
    if not all_eps:
        logger.error("No episodes found; check your data_dir!")
        return

    by_host: Dict[str, List] = defaultdict(list)
    for ep in all_eps:
        # Use the first predicted host name rather than metadata.host
        host_list = getattr(ep, "host_names", None) or getattr(ep, "hostPredictedNames", None)
        if isinstance(host_list, list) and host_list:
            host = host_list[0]
        else:
            host = "Unknown"
        by_host[host].append(ep)

    counts = Counter({h: len(eps) for h, eps in by_host.items()})
    top_hosts = [h for h, _ in counts.most_common(args.top_n_hosts)]
    logger.info(f"Top {args.top_n_hosts} hosts: {top_hosts}")

    # ─── 2) SET UP LLM ──────────────────────────────────────────────────────────
    llm = LLMInterface(gpu_id=args.gpu_id)

    all_results, all_raw = [], []

    # ─── 3) ANALYZE EACH TOP HOST ──────────────────────────────────────────────
    for host in top_hosts:
        logger.info(f"Loading episodes for host={host!r}…")
        sporc = SPORCDataset(local_data_dir=args.data_dir, streaming=True)
        sporc.load_podcast_subset(hosts=[host])   # ← always a non-empty list
        eps = sporc.get_all_episodes()
        logger.info(f"  → {len(eps)} episodes found; sampling up to 30…")
        sample = random.sample(eps, k=min(30, len(eps)))

        for ep in tqdm(sample, desc=f"Host {host}", unit="ep"):
            turns = [t for t in ep.get_all_turns() if count_words(t.text) > args.min_words]
            windows = [
                turns[i : i + args.window_size]
                for i in range(0, len(turns) - args.window_size + 1, args.stride)
            ]

            for idx, win in enumerate(windows):
                joined = "\n\n".join(f"{t.speaker}: {t.text.strip()}" for t in win)
                prompt = f"""
You are an expert podcast conversation analyst.
Output *only* valid JSON matching exactly this schema:

{{
  "key_points_discussed_or_proposed": [ string, … ],
  "key_points_assumed":           [ string, … ]
}}

Analyze Window #{idx} of "{ep.title}" (host: {host}):
{joined}
"""
                raw = llm.generate(prompt)
                all_raw.append({
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
                all_results.append({
                    "Host": host,
                    "Podcast": ep.title,
                    "Speakers": ",".join(speakers),
                    "WindowIndex": idx,
                    "WindowText": joined,
                    "KeyPoints": data.get("key_points_discussed_or_proposed", []),
                    "Assumptions": data.get("key_points_assumed", [])
                })

    # ─── 4) WRITE OUTPUTS ────────────────────────────────────────────────────────
    os.makedirs("results/hosts", exist_ok=True)
    summary = f"results/hosts/top_{args.top_n_hosts}_hosts_summary.json"
    raw_out = f"results/hosts/top_{args.top_n_hosts}_hosts_raw.json"

    with open(summary, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Wrote parsed results → {summary}")

    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, indent=2)
    logger.info(f"Wrote raw outputs      → {raw_out}")

if __name__ == "__main__":
    main()
