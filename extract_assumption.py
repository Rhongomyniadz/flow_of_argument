import os
import re
import gc
import json
import gzip
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------- Helpers --------------------

def normalize_output(raw: str) -> List[str]:
    """Extract list from model output that contains a JSON object with 'key_points_assumed'."""
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return []
        obj = json.loads(raw[start:end + 1])
        vals = obj.get("key_points_assumed", [])
        return [s for s in vals if isinstance(s, str)]
    except Exception:
        return []


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def get_text(turn: Dict) -> str:
    return (turn.get("text") or turn.get("turn_text") or "").strip()


def get_role(turn: Dict) -> str:
    role = turn.get("inferred_speaker_role") or turn.get("role") or ""
    role = role.strip()
    return role if role else "SPEAKER"


def get_name(turn: Dict) -> str:
    name = (turn.get("inferred_speaker_name") or turn.get("speaker_name") or "").strip()
    return name if name else "NO_INFERRED_SPEAKER"


def get_speaker_id(turn: Dict) -> Optional[str]:
    sid = turn.get("speaker_id", None)
    if sid is None:
        sid = turn.get("speaker", None)
    if isinstance(sid, list):
        return "-".join(str(x) for x in sid)
    if isinstance(sid, dict):
        return str(sid.get("id") or sid.get("speaker_id") or "")
    return str(sid) if sid is not None else None


def turns_from_episode(ep: Dict) -> List[Dict]:
    # Common keys across datasets
    for k in ("turns", "speaker_turns", "segments"):
        if isinstance(ep.get(k), list):
            return ep[k]
    # Sometimes nested under 'transcript' or 'content'
    trans = ep.get("transcript") or ep.get("content") or {}
    if isinstance(trans, dict):
        for k in ("turns", "speaker_turns", "segments"):
            v = trans.get(k)
            if isinstance(v, list):
                return v
    return []


def episode_title(ep: Dict) -> str:
    for k in ("title", "episode_title", "name"):
        v = ep.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return "untitled_episode"


def unique_speakers(turns: List[Dict]) -> List[str]:
    sids = []
    for t in turns:
        sid = get_speaker_id(t)
        if not sid:
            sid = get_name(t)  # fallback so we can still count distinct speakers
        sids.append(sid)
    uniq = []
    for s in sids:
        if s and s not in uniq:
            uniq.append(s)
    return uniq


def stream_episodes_from_jsonl_gz(path: Path) -> Iterable[Dict]:
    """Yield episode dicts from a jsonl or jsonl.gz file (one episode per line)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for o in obj:
                    if isinstance(o, dict):
                        yield o


# -------------------- LLM Wrapper --------------------

class LLMInterface:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
        # vLLM pins devices via CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            download_dir=download_dir,
        )
        try:
            self.params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
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


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Process episodes directly from /data/covid_episodes.jsonl.gz")
    ap.add_argument("--data_path", type=str, default="/data/covid_episodes.jsonl.gz",
                    help="Path to JSONL(.gz) file with one episode per line.")
    ap.add_argument("--min_words", type=int, default=50,
                    help="Minimum words in a turn to run the LLM on it.")
    ap.add_argument("--sample_n", type=int, default=30,
                    help="Max episodes to process via reservoir sampling.")
    ap.add_argument("--gpu_id", type=int, default=0, help="GPU id to use with vLLM.")
    ap.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="HuggingFace model name for vLLM.")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", str(data_path))
        return

    # 1) Stream episodes, keep those with exactly 2 speakers; reservoir sample up to sample_n
    reservoir: List[Dict] = []
    total_seen = 0
    logger.info("Streaming episodes from %s …", str(data_path))

    for ep in tqdm(stream_episodes_from_jsonl_gz(data_path), desc="scan jsonl"):
        turns = turns_from_episode(ep)
        if not turns:
            continue
        spk = unique_speakers(turns)
        if len(spk) != 2:
            continue

        total_seen += 1
        if len(reservoir) < args.sample_n:
            reservoir.append(ep)
        else:
            j = __import__("random").randint(0, total_seen - 1)
            if j < args.sample_n:
                reservoir[j] = ep

    logger.info("Reservoir sampled %d episodes (out of %d 2-speaker matches)", len(reservoir), total_seen)
    if not reservoir:
        logger.warning("No qualifying episodes found; exiting.")
        return

    # 2) Init LLM
    llm = LLMInterface(model_name=args.model_name, gpu_id=args.gpu_id)

    # 3) Process sampled episodes
    out_dir = Path("results/covid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in tqdm(reservoir, desc="processing eps"):
        title = episode_title(ep)
        label = re.sub(r"[^\w\-]", "_", title)

        turns = turns_from_episode(ep)
        prompts, meta = [], []
        for idx, t in enumerate(tqdm(turns, desc=f"turns {label}", leave=False)):
            text = get_text(t)
            if count_words(text) < args.min_words:
                continue
            role = get_role(t).upper()
            prompts.append(f"""
SYSTEM:
You are an expert podcast conversation analyst.

TASK:
Given a single speaker turn, extract the "key_points_assumed".

OUTPUT a JSON object with exactly one key "key_points_assumed" mapping to a list of strings.

Now analyze Turn #{idx}":
{role}: {text.strip()}
""")
            meta.append((
                text,
                get_speaker_id(t),
                get_name(t),
                t.get("inferred_speaker_role"),
            ))

        if not prompts:
            continue

        outputs = llm.generate_batch(prompts)
        records = []
        for (text, speaker, name, role), raw_out in zip(meta, outputs):
            records.append({
                "turn_text": text,
                "speaker_id": speaker,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role,
                "assumptions": normalize_output(raw_out),
            })

        with open(out_dir / f"{label}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # free memory between episodes
        del prompts, meta, outputs, records
        gc.collect()

    logger.info("Done. Wrote episode JSONs to %s", str(out_dir))


if __name__ == "__main__":
    main()
