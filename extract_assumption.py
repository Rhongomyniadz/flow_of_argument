import os
import re
import gc
import json
import gzip
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple, Set
from urllib.parse import urlparse

from tqdm import tqdm
from vllm import LLM, SamplingParams
from text_regularizer import TextRegularizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-extract")


# -------------------- URL helpers --------------------

def canonical_to_raw(canonical_url: str) -> str:
    """Map an mp3 URL to the 'raw' url format used in indices."""
    if not canonical_url:
        return ""
    p = urlparse(canonical_url)
    domain = p.netloc
    scheme = p.scheme
    path = (p.path or "").lstrip("/")
    collapsed = path.replace("/", "").replace("-", "")
    host_noslash = f"{scheme}{domain}"
    return f"/{domain}/o3/{host_noslash}{collapsed}MERGED" if domain and scheme else ""


# -------------------- Episode & turn parsing --------------------

def get_text(turn: Dict) -> str:
    return (turn.get("text") or turn.get("turn_text") or "").strip()

def get_role(turn: Dict) -> str:
    role = (turn.get("inferred_speaker_role") or turn.get("role") or "").strip()
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
    # Common keys if an episode embeds turns (often it won't)
    for k in ("turns", "speaker_turns", "segments"):
        v = ep.get(k)
        if isinstance(v, list):
            return v
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

def episode_mp3(ep: Dict) -> str:
    # Prefer explicit mp3_url if present
    for k in ("mp3_url", "audio_url", "audio", "url", "audio_url_http"):
        v = ep.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    return ""

def has_two_speakers_meta(ep: Dict) -> bool:
    # Strong signals
    n_main = ep.get("num_main_speakers")
    if isinstance(n_main, (int, float)) and int(n_main) == 2:
        return True

    q = ep.get("quality_indicators", {})
    if isinstance(q, dict):
        total_labels = q.get("total_speaker_labels")
        if isinstance(total_labels, (int, float)) and int(total_labels) == 2:
            return True

    # Hosts + guests
    nh = ep.get("num_hosts")
    ng = ep.get("num_guests")
    if isinstance(nh, (int, float)) and isinstance(ng, (int, float)):
        if int(nh) + int(ng) == 2:
            return True

    # Fallback list fields
    for k in ("speakers", "speaker_ids"):
        v = ep.get(k)
        if isinstance(v, list) and len(v) == 2:
            return True

    return False

def normalize_output(raw: str) -> List[str]:
    """Extract list from model output that contains a JSON object with 'key_points_assumed'."""
    try:
        start = raw.find("{"); end = raw.rfind("}")
        if start == -1 or end == -1:
            return []
        obj = json.loads(raw[start:end + 1])
        vals = obj.get("key_points_assumed", [])
        return [s for s in vals if isinstance(s, str)]
    except Exception:
        return []

def count_words(text: str) -> int:
    import re
    return len(re.findall(r"\w+", text or ""))


# -------------------- Local streamers & path discovery --------------------

def stream_jsonl(path: Path) -> Iterable[Dict]:
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

def find_episodes_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Data file not found: {p}")
    # auto-discover under CWD
    candidates = [
        Path("data/covid_episodes.jsonl.gz"),
        Path("results/covid_episodes.jsonl.gz"),
        Path("data/covid_episodes.jsonl"),
        Path("results/covid_episodes.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Could not locate covid_episodes.jsonl[.gz] under ./data or ./results; "
                            "pass --data_path explicitly.")

def find_turns_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Turns file not found: {p}")
    candidates = [
        Path("data/covid_episodes_turn.jsonl.gz"),
        Path("results/covid_episodes_turn.jsonl.gz"),
        Path("data/covid_episodes_turn.jsonl"),
        Path("results/covid_episodes_turn.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Could not locate covid_episodes_turn.jsonl[.gz] under ./data or ./results; "
                            "pass --turns_path explicitly.")


def collect_needed_turns_from_local(mp3_urls: Set[str],
                                    turns_path: Path,
                                    max_per_episode: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Stream the local turns JSONL(.gz) and collect turns only for the target episodes.
    Matching is by mp3_url OR canonical raw url found in each turn record.
    Returns: dict mp3_url -> list of turn dicts (min fields text/speaker/name/role).
    """
    opener = gzip.open if turns_path.suffix == ".gz" else open

    # Precompute acceptable keys for quick matching
    keys_by_mp3 = {}
    for mp3 in mp3_urls:
        keys_by_mp3[mp3] = {mp3, canonical_to_raw(mp3)}

    out: Dict[str, List[Dict]] = {mp3: [] for mp3 in mp3_urls}
    hits_left = set(mp3_urls)

    candidate_keys = (
        "mp3_url", "audio_url", "audio", "episode_mp3", "episode_url", "url",
        "canonical_url", "raw_url",
    )

    with opener(turns_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="scan local turns", unit="lines"):
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # candidate episode identifiers inside a turn record
            cand = ""
            for k in candidate_keys:
                v = rec.get(k)
                if isinstance(v, str) and v:
                    cand = v
                    break

            if not cand:
                continue

            matched_mp3 = None
            for mp3, keys in keys_by_mp3.items():
                if cand in keys:
                    matched_mp3 = mp3
                    break
                # Try canonicalizing if cand looks like an mp3 url
                if cand.startswith("http") and canonical_to_raw(cand) in keys:
                    matched_mp3 = mp3
                    break

            if not matched_mp3:
                continue

            # Extract minimal turn fields
            text = (rec.get("text") or rec.get("turn_text") or "").strip()
            if not text:
                continue
            turn_obj = {
                "text": text,
                "speaker_id": rec.get("speaker_id", rec.get("speaker")),
                "inferred_speaker_name": rec.get("inferred_speaker_name", "NO_INFERRED_SPEAKER"),
                "inferred_speaker_role": rec.get("inferred_speaker_role", "NO_INFERRED_ROLE"),
            }
            out[matched_mp3].append(turn_obj)

            if max_per_episode and len(out[matched_mp3]) >= max_per_episode:
                if matched_mp3 in hits_left:
                    hits_left.remove(matched_mp3)

            # Optional early stop if all found and capped
            if max_per_episode and not hits_left:
                break

    return out


# -------------------- LLM wrapper --------------------

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
        download_dir: str = "/shared/4/models",
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.llm = LLM(
            model=model_name,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
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
            log.warning("vLLM SamplingParams has no `min_p`; proceeding without it.")
            self.params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        self.regularizer = TextRegularizer()

    def generate_batch(self, prompts: List[str], original_texts: List[str]) -> List[str]:
        prompts = []
        for text in original_texts:
            prompt = f"""
SYSTEM:
You are an expert in analyzing conversations and identifying implicit assumptions in speech. Your task is to uncover the underlying assumptions that speakers make in their statements.

TASK:
Analyze the following conversation turn and identify the key underlying assumptions. Focus on:
1. Unstated beliefs the speaker holds
2. Implicit knowledge they assume their audience has
3. Hidden premises that support their arguments
4. Contextual assumptions about their environment or situation

FORMAT:
Return a JSON object with one key "key_points_assumed" containing a list of clear, specific assumptions.
Each assumption should be:
- A complete, well-formed sentence
- Different from but related to the original text
- Focused on one specific point
- Not a mere restatement of what was explicitly said

EXAMPLE:
Turn: "We need to move the meeting online since everyone's working remotely now."
{
  "key_points_assumed": [
    "In-person meetings are no longer feasible or safe.",
    "All team members have access to reliable internet connectivity.",
    "The team is familiar with virtual meeting technology.",
    "Remote work will continue for the foreseeable future."
  ]
}

Now analyze this turn:
{text}
"""
            prompts.append(prompt)

        out = self.llm.generate(prompts, self.params)
        results = []
        
        for output, orig_text in zip(out, original_texts):
            raw_assumptions = normalize_output(output.outputs[0].text.strip())
            regularized = self.regularizer.regularize_assumptions(orig_text, raw_assumptions)
            results.append(regularized)
            
        return results


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="", help="Path to local episodes jsonl(.gz). If empty, auto-discover.")
    ap.add_argument("--turns_path", type=str, default="", help="Path to local turns jsonl(.gz). If empty, auto-discover.")
    ap.add_argument("--min_words", type=int, default=50, help="Min words in a turn to run LLM.")
    ap.add_argument("--sample_n", type=int, default=30, help="Max episodes to process (reservoir).")
    args = ap.parse_args()

    # 1) Locate the local episodes & turns files
    try:
        episodes_path = find_episodes_path(args.data_path)
        turns_path = find_turns_path(args.turns_path)
    except FileNotFoundError as e:
        log.error(str(e))
        return

    log.info("Streaming episodes from %s …", str(episodes_path))
    log.info("Using local turns from %s …", str(turns_path))

    # 2) Build reservoir of candidate episodes (2-speaker via embedded turns OR metadata)
    reservoir: List[Dict] = []
    total_seen = 0
    all_sampled_mp3: List[str] = []
    episodes_buffer: List[Tuple[Dict, Optional[List[Dict]], str]] = []
    # tuple: (episode_json, turns_if_present, mp3_url)

    for ep in tqdm(stream_jsonl(episodes_path), desc="scan episodes jsonl"):
        mp3 = episode_mp3(ep)
        if not mp3:
            continue

        turns = turns_from_episode(ep)
        if turns:
            # deduce speakers directly
            sids = []
            for t in turns:
                sid = get_speaker_id(t) or get_name(t)
                if sid and sid not in sids:
                    sids.append(sid)
            is_two = len(sids) == 2
        else:
            # fall back to metadata-based check
            is_two = has_two_speakers_meta(ep)

        if not is_two:
            continue

        total_seen += 1
        if len(reservoir) < args.sample_n:
            reservoir.append(ep)
            episodes_buffer.append((ep, turns if turns else None, mp3))
            all_sampled_mp3.append(mp3)
        else:
            import random
            j = random.randint(0, total_seen - 1)
            if j < args.sample_n:
                reservoir[j] = ep
                episodes_buffer[j] = (ep, turns if turns else None, mp3)
                all_sampled_mp3[j] = mp3

    log.info("Reservoir sampled %d episodes (out of %d 2-speaker matches)", len(reservoir), total_seen)
    if not reservoir:
        log.warning("No qualifying episodes found; exiting.")
        return

    # 3) Back-fill turns for sampled episodes that lack embedded turns (from local turns file)
    need_turns_for = {mp3 for (_, tlist, mp3) in episodes_buffer if tlist is None}
    turns_by_mp3: Dict[str, List[Dict]] = {}
    if need_turns_for:
        log.info("Back-filling turns for %d sampled episodes from local turns file", len(need_turns_for))
        turns_by_mp3 = collect_needed_turns_from_local(need_turns_for, turns_path)

    # 4) Init LLM
    llm = LLMInterface()

    # 5) Process episodes and write per-episode outputs
    out_dir = Path("results/covid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep, tlist, mp3 in tqdm(episodes_buffer, desc="processing eps"):
        title = episode_title(ep)
        label = re.sub(r"[^\w\-]", "_", title)

        if tlist is None:
            tlist = turns_by_mp3.get(mp3, [])
        if not tlist:
            log.warning("No turns found for sampled episode: %s", title)
            continue

        prompts, meta = [], []
        for idx, t in enumerate(tqdm(tlist, desc=f"turns {label}", leave=False)):
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
            log.warning("No qualifying turns (min_words=%d) for: %s", args.min_words, title)
            continue

        outputs = llm.generate_batch(prompts, [text for text, _, _, _ in meta])
        records = []
        for (text, speaker, name, role), raw_out in zip(meta, outputs):
            records.append({
                "turn_text": text,
                "speaker_id": speaker,
                "inferred_speaker_name": name,
                "inferred_speaker_role": role if role else "NO_INFERRED_ROLE",
                "assumptions": normalize_output(raw_out),
            })

        with open(out_dir / f"{label}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        del prompts, meta, outputs, records
        gc.collect()

    log.info("Done. Wrote episode JSONs to %s", str(out_dir))


if __name__ == "__main__":
    main()
