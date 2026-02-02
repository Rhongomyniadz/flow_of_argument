import os, json, glob, argparse, re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from vllm import LLM, SamplingParams


class LLMInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        download_dir: str = "/shared/4/models",
        max_tokens: int = 1000,
    ):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        out = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in out]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_turn_text(turn: Dict[str, Any]) -> str:
    key = turn.get("chosen_text_key")
    if isinstance(key, str) and key in turn and isinstance(turn[key], str):
        return turn[key]
    for cand in ("transcript", "turn_text", "text"):
        if cand in turn and isinstance(turn[cand], str):
            return turn[cand]
    return ""


def turn_start_time(turn: Dict[str, Any]) -> Optional[float]:
    st = turn.get("startTime")
    if isinstance(st, (int, float)):
        return float(st)
    return None


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


STOP = set("""
a an the and or but if then else to of in on for from with without by as is are was were be been being
this that these those it its i you he she they we them him her my your our their
""".split())


def keywords(s: str) -> set:
    toks = normalize_text(s).split()
    return set(t for t in toks if len(t) >= 3 and t not in STOP)


def overlap_score(a_kw: set, c_kw: set) -> int:
    return len(a_kw & c_kw)


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first balanced JSON object found in `text`.
    More robust than a greedy regex when the model emits extra braces or prose.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        return json.loads(blob)
                    except Exception:
                        return None

    return None


def context_window(turns: List[Dict[str, Any]], idx: int, w: int) -> str:
    lo = max(0, idx - w)
    hi = min(len(turns), idx + w + 1)
    lines = []
    for k in range(lo, hi):
        spk = turns[k].get("speaker_id") or turns[k].get("speaker") or "SPEAKER"
        txt = extract_turn_text(turns[k]).replace("\n", " ").strip()
        if not txt:
            continue
        lines.append(f"[{k}] {spk}: {txt}")
    return "\n".join(lines) if lines else "[NO CONTEXT]"


def build_entailment_prompt(
    assumption_text: str,
    claim_text: str,
    a_turn_idx: int,
    c_turn_idx: int,
    a_context: str,
    c_context: str,
) -> str:
    return f"""You are evaluating conversational accommodation.

Goal:
Rate how strongly the FUTURE explicit claim C (turn {c_turn_idx}) makes the earlier assumption A (turn {a_turn_idx}) explicit.

Important:
- Resolve coreference using the provided context (e.g., he/they/it).
- Judge whether C states or clearly verifies the substance of A.
- Use a 1-10 scale (see below).
- Output MUST be strict JSON only.
- Output MUST contain ONLY the keys listed below. No extra keys, no prose.

Entailment Scale (1-10):
1 = clearly unrelated or contradicts A
3 = weak topical relation only
5 = partially supports A but missing key content
7 = strongly entails A (core meaning stated)
9 = directly and explicitly verifies A
10 = explicit verification with no ambiguity

Earlier assumption A (turn {a_turn_idx}):
{assumption_text}

Context around A:
{a_context}

Future explicit claim C (turn {c_turn_idx}):
{claim_text}

Context around C:
{c_context}

Return JSON with EXACTLY these keys:
{{
  "entailment_score": 1-10,
  "confidence": 0.0-1.0
}}
"""


def run_episode_labeling(
    episode_path: str,
    llm: LLMInterface,
    out_path: str,
    max_future_turns: int,
    max_claims_per_assumption: int,
    min_overlap: int,
    context_w: int,
    batch_size: int,
) -> Dict[str, Any]:
    turns = load_json(episode_path)
    if not isinstance(turns, list):
        raise ValueError(f"Expected list in {episode_path}")

    # Pre-index explicit claims by turn
    claims_by_turn: List[List[str]] = []
    for t in turns:
        cps = t.get("explicit_propositions", []) or []
        out = []
        for c in cps:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                txt = c["text"].strip()
                if txt:
                    out.append(txt)
        claims_by_turn.append(out)

    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []

    # Candidate pairs (A, future C)
    for i, t in enumerate(turns):
        assumptions = t.get("assumptions", []) or []
        if not assumptions:
            continue

        a_time = turn_start_time(t)
        a_ctx = context_window(turns, i, context_w)

        for a_idx_in_turn, a in enumerate(assumptions):
            if not (isinstance(a, dict) and isinstance(a.get("text"), str)):
                continue
            a_text = a["text"].strip()
            if not a_text:
                continue

            a_kw = keywords(a_text)
            cand: List[Tuple[int, str, int]] = []

            j_end = min(len(turns), i + 1 + max_future_turns) if max_future_turns > 0 else len(turns)
            for j in range(i + 1, j_end):
                for c_text in claims_by_turn[j]:
                    score = overlap_score(a_kw, keywords(c_text))
                    if score >= min_overlap:
                        cand.append((j, c_text, score))

            cand.sort(key=lambda x: x[2], reverse=True)
            cand = cand[:max_claims_per_assumption]

            for (j, c_text, score) in cand:
                c_ctx = context_window(turns, j, context_w)
                prompts.append(build_entailment_prompt(
                    assumption_text=a_text,
                    claim_text=c_text,
                    a_turn_idx=i,
                    c_turn_idx=j,
                    a_context=a_ctx,
                    c_context=c_ctx,
                ))
                meta.append({
                    "a_turn_idx": i,
                    "a_time": a_time,
                    "a_idx_in_turn": a_idx_in_turn,
                    "assumption_text": a_text,
                    "c_turn_idx": j,
                    "claim_text": c_text,
                    "overlap_score": score,
                    "c_time": turn_start_time(turns[j]),
                })

    # Run model in batches
    judged: List[str] = []
    for s in range(0, len(prompts), batch_size):
        outs = llm.generate_batch(prompts[s:s + batch_size])
        judged.extend(outs)

    # Record scored candidate pairs
    pairs: List[Dict[str, Any]] = []
    for m, out in zip(meta, judged):
        parsed = safe_json_extract(out)
        if parsed is None or "entailment_score" not in parsed:
            parsed = {
                "entailment_score": 0,
                "confidence": 0.0,
            }

        try:
            score = int(parsed.get("entailment_score", 0))
        except Exception:
            score = 0
        score = max(0, min(10, score))

        try:
            conf = float(parsed.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        pairs.append({
            **m,
            "entailment_score": score,
            "confidence": conf,
            "raw": out,
        })

    episode_id = os.path.splitext(os.path.basename(episode_path))[0]
    out_obj = {
        "episode_id": episode_id,
        "params": {
            "max_future_turns": max_future_turns,
            "max_claims_per_assumption": max_claims_per_assumption,
            "min_overlap": min_overlap,
            "context_w": context_w,
            "batch_size": batch_size,
        },
        "pairs": pairs,
    }
    save_json(out_path, out_obj)

    # small metadata return
    return {
        "episode_id": episode_id,
        "n_pairs": len(pairs),
        "out_path": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/stance_labeled")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--glob", type=str, default="*.json")
    ap.add_argument("--out_dir", type=str, default="data/implicature_flow/entailment_pairs_1to10")

    ap.add_argument("--max_future_turns", type=int, default=30, help="0 = all future")
    ap.add_argument("--max_claims_per_assumption", type=int, default=15)
    ap.add_argument("--min_overlap", type=int, default=2)
    ap.add_argument("--context_w", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=128)

    args = ap.parse_args()

    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    in_k_dir = os.path.join(args.in_dir, str(args.k))
    paths = sorted(glob.glob(os.path.join(in_k_dir, args.glob)))
    ensure_dir(args.out_dir)

    meta = []
    for ep_path in tqdm(paths, desc="Label episodes", unit="ep"):
        episode_id = os.path.splitext(os.path.basename(ep_path))[0]
        out_path = os.path.join(args.out_dir, f"{episode_id}.json")
        meta.append(run_episode_labeling(
            episode_path=ep_path,
            llm=llm,
            out_path=out_path,
            max_future_turns=args.max_future_turns,
            max_claims_per_assumption=args.max_claims_per_assumption,
            min_overlap=args.min_overlap,
            context_w=args.context_w,
            batch_size=args.batch_size,
        ))

    save_json(os.path.join(args.out_dir, f"_LABELING_META_k{args.k}.json"), {
        "k": args.k,
        "n_episodes": len(meta),
        "per_episode": meta
    })


if __name__ == "__main__":
    main()
