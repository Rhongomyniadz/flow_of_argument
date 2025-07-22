import os
import argparse
import random
import json
import logging
import re
from urllib.parse import urlparse
from typing import List, Dict, Optional

import pandas as pd
from sporc import SPORCDataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def canonical_to_raw(canonical_url: str) -> str:
    p = urlparse(canonical_url)
    domain = p.netloc
    scheme = p.scheme
    path = p.path.lstrip("/")
    collapsed = path.replace("/", "").replace("-", "")
    host_noslash = f"{scheme}{domain}"
    raw_body = f"{host_noslash}{collapsed}"
    return f"/{domain}/o3/{raw_body}MERGED"


def normalize_output(raw_response: str) -> List[str]:
    """Extract only the 'key_points_assumed' array, or return [] if missing."""
    start, end = raw_response.find("{"), raw_response.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw_response[start : end + 1])
        return parsed.get("key_points_assumed", [])
    except json.JSONDecodeError:
        return []


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7,
        top_p: float = 0.8,
        min_p: float = 0.1,
        repetition_penalty: float = 1.1,
        top_k: int = 30,
        max_tokens: int = 2048,
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


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(
        description="Analyze SPORC episodes for COVID and George Floyd topics"
    )
    parser.add_argument("--min_words", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sample_n", type=int, default=30)
    parser.add_argument("--topic_threshold", type=float, default=0.03)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # 1) Load doc-topic and topic-keys
    cols = ["row_id", "url"] + [f"topic_{i}" for i in range(100)]
    doc_topics = pd.read_csv("doc_topic.txt", sep="\t", header=None, names=cols)
    topic_keys = pd.read_csv(
        "topic_keys.txt", sep="\t", header=None, names=["topic_id", "overall_prop", "keywords"]
    )

    # 2) Define the two target topics and find their topic‐IDs
    topics_to_find = {
        "covid": ["covid"],
        "george_floyd": ["george", "floyd"],
    }

    # 3) Initialize SPORC
    data_dir = "/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/"
    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
    sporc.load_podcast_subset()
    all_episodes = sporc.get_all_episodes()

    # 4) Build URL→Episode map
    raw2ep = {canonical_to_raw(ep.mp3_url): ep for ep in all_episodes}

    # 5) Prepare LLM
    llm = LLMInterface(gpu_id=args.gpu_id)

    for topic_name, keywords in topics_to_find.items():
        # 5a) find matching topic IDs
        mask = topic_keys["keywords"].apply(
            lambda s: any(kw.lower() in s.lower() for kw in keywords)
        )
        topic_ids = topic_keys.loc[mask, "topic_id"].tolist()
        if not topic_ids:
            logger.warning(f"No topics found for '{topic_name}', skipping.")
            continue

        topic_cols = [f"topic_{i}" for i in topic_ids]

        # 5b) filter doc_topics by threshold
        filtered_docs = doc_topics[
            doc_topics[topic_cols].max(axis=1) > args.topic_threshold
        ]
        logger.info(f"[{topic_name}] {len(filtered_docs)} docs above threshold")

        # 5c) map to SPORC episodes, then keep only those with exactly 1 host + 1 guest
        matched = filtered_docs[filtered_docs["url"].isin(raw2ep)]
        eps = [raw2ep[u] for u in matched["url"]]
        eps_2spk = [
            e
            for e in eps
            if len(e.host_names) == 1 and len(e.guest_names) == 1
        ]
        logger.info(f"[{topic_name}] {len(eps_2spk)} episodes with 2 speakers")

        # 5d) sample if too many
        random.seed(42)
        sample_eps = random.sample(eps_2spk, k=min(args.sample_n, len(eps_2spk)))

        # 5e) analyze windows
        per_podcast_results: Dict[str, List[Dict]] = {}
        for ep in tqdm(sample_eps, desc=f"Analyzing {topic_name}"):
            turns = [t for t in ep.get_all_turns() if count_words(t.text) > args.min_words]
            windows = [
                turns[i : i + args.window_size]
                for i in range(0, len(turns) - args.window_size + 1, args.stride)
            ]
            for idx, win in enumerate(windows):
                # separate host & guest turns
                host_texts = [t.text.strip() for t in win if t.inferred_role == "host"]
                guest_texts = [t.text.strip() for t in win if t.inferred_role == "guest"]

                # build prompt on full window (you can also tailor to host/guest separately)
                text_block = "\n\n".join(
                    f"{t.inferred_role.upper()}: {t.text.strip()}" for t in win
                )
                prompt = f"""
SYSTEM:
You are an expert podcast conversation analyst.

TASK:
  • Given exactly {args.window_size} consecutive turns, extract only the array
    "key_points_assumed".

OUTPUT a JSON object with exactly one key "key_points_assumed" mapping to a list of strings.

Now analyze Window #{idx} from episode "{ep.title}":
{text_block}
"""
                raw = llm.generate(prompt)
                assumptions = normalize_output(raw)

                rec = {
                    "WindowIndex": idx,
                    "HostTurns": host_texts,
                    "GuestTurns": guest_texts,
                    "Assumptions": assumptions,
                }
                per_podcast_results.setdefault(ep.title, []).append(rec)

        # 5f) write out one JSON per podcast
        out_dir = os.path.join("results", topic_name)
        os.makedirs(out_dir, exist_ok=True)
        for podcast_name, records in per_podcast_results.items():
            fname = sanitize_filename(podcast_name) + ".json"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(f"[{topic_name}] done; results in {out_dir}/")

    logger.info("All topics processed.")


if __name__ == "__main__":
    main()
