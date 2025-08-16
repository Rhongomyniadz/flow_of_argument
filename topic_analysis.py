import os
import argparse
import random
import json
import logging
import re
from urllib.parse import urlparse
from typing import List
import gc

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
    return f"/{domain}/o3/{host_noslash}{collapsed}MERGED"


def normalize_output(raw: str) -> List[str]:
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return obj.get("key_points_assumed", [])
    except Exception:
        return []


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8b",
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.5,
        temperature: float = 0.7,
        top_p: float = 0.8,
        min_p: float = 0.1,
        repetition_penalty: float = 1.1,
        top_k: int = 30,
        max_tokens: int = 2048,
        tensor_parallel_size: int = 1,
        download_dir: str = "/shared/4/models",
        trust_remote_code: bool = True,
    ):
        # IMPORTANT: vLLM doesn't take a `device` kwarg. Pin GPU via environment var.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            download_dir=download_dir,
        )

        # Some vLLM versions don’t support `min_p`. Try with it, then fall back.
        try:
            self.params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,  # may not be supported in your vLLM version
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        except TypeError:
            logger.warning("vLLM SamplingParams has no `min_p`; proceeding without it.")
            self.params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                max_tokens=max_tokens,
            )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        out = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in out]


def main():
    parser = argparse.ArgumentParser(description="Analyze 2-speaker SPORC episodes (COVID topic filter)")
    parser.add_argument("--min_words",       type=int,   default=50)
    parser.add_argument("--sample_n",        type=int,   default=50)
    parser.add_argument("--topic_threshold", type=float, default=0.02)
    parser.add_argument("--gpu_id",          type=int,   default=0)
    args = parser.parse_args()

    random.seed(42)

    # 1) Load topic_keys and pick COVID-related topic columns (EXACT line you requested)
    topic_keys = pd.read_csv(
        "/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/topic_keys.txt",
        sep="\t", header=None,
        names=["topic_id", "overall_prop", "keywords"],
    )

    # DO NOT MODIFY THIS PER YOUR INSTRUCTION
    covid_ids = topic_keys[topic_keys.keywords.str.contains("covid", case=False)].topic_id
    topic_cols = [f"topic_{i}" for i in covid_ids]

    if len(topic_cols) == 0:
        logger.warning("No COVID-related topics found in topic_keys; nothing to do.")
        return

    # 2) Load only URL + selected topic columns from doc_topics and build matched_urls
    matched_urls = set()
    usecols = ["url"] + topic_cols
    dtype = {c: "float32" for c in topic_cols}

    logger.info("Reading doc_topics.txt…")
    doc_topics = pd.read_csv(
        "/shared/3/projects/podcasts/SPoRC/topicModelling/100/transcripts/doc_topics.txt",
        sep="\t",
        header=None,
        names=["row_id", "url"] + [f"topic_{i}" for i in range(100)],
        usecols=usecols,
        dtype=dtype,
    )

    # Keep any row where any COVID topic weight > threshold
    mask = doc_topics[topic_cols].max(axis=1) > args.topic_threshold
    matched_urls.update(doc_topics.loc[mask, "url"])

    logger.info(f"{len(matched_urls)} docs match threshold among COVID topics")

    del doc_topics, topic_keys
    gc.collect()

    # 3) Load SPORC in streaming mode and get exactly-2-speaker episodes
    data_dir = "/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/"

    sporc = SPORCDataset(local_data_dir=data_dir, streaming=True)
    sporc.load_podcast_subset()

    two_speaker_eps = sporc.search_episodes(min_speakers=2, max_speakers=2)
    logger.info(f"Found {len(two_speaker_eps)} two-speaker episodes")

    # 4) Reservoir-sample up to sample_n episodes that also match our URLs
    reservoir = []
    total_seen = 0
    for ep in tqdm(two_speaker_eps, desc="sampling eps"):
        raw = canonical_to_raw(ep.mp3_url)
        if raw not in matched_urls:
            continue
        total_seen += 1
        if len(reservoir) < args.sample_n:
            reservoir.append(ep)
        else:
            j = random.randint(0, total_seen - 1)
            if j < args.sample_n:
                reservoir[j] = ep
    logger.info(f"Reservoir sampled {len(reservoir)} episodes")

    if len(reservoir) == 0:
        logger.warning("No episodes found that match the criteria.")
        return

    llm = LLMInterface(gpu_id=args.gpu_id)

    # 5) Process sampled episodes and save results immediately
    out_dir = "results/covid"
    os.makedirs(out_dir, exist_ok=True)

    for ep in tqdm(reservoir, desc="processing eps"):
        label = re.sub(r"[^\w\-]", "_", ep.title)
        prompts, meta = [], []
        for idx, t in enumerate(tqdm(ep.get_all_turns(), desc=f"turns {label}", leave=False)):
            if count_words(t.text) < args.min_words:
                continue
            role = (t.inferred_speaker_role or "SPEAKER").upper()
            prompts.append(f"""
SYSTEM:
You are an expert podcast conversation analyst.

TASK:
Given a single speaker turn, extract the "key_points_assumed".

OUTPUT a JSON object with exactly one key "key_points_assumed" mapping to a list of strings.

Now analyze Turn #{idx}":
{role}: {t.text.strip()}
""")
            meta.append((t.text, t.speaker, t.inferred_speaker_name, t.inferred_speaker_role))

        if not prompts:
            continue

        outputs = llm.generate_batch(prompts)
        records = []
        for (text, speaker, name, role), raw in zip(meta, outputs):
            records.append({
                "turn_text": text,
                "speaker_id": speaker,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role,
                "assumptions": normalize_output(raw),
            })

        fname = f"{label}.json"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # free up memory
        del prompts, meta, outputs, records

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
