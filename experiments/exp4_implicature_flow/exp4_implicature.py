import os, json, glob, argparse, re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# ---- vLLM wrapper (your interface) ----
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
        max_tokens: int = 2000,
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


# ----------------- utilities -----------------

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
    Try to parse the first JSON object in the model output.
    """
    if not text:
        return None
    # find a {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
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


# ----------------- LLM judge prompt -----------------

def build_entailment_prompt(
    assumption_text: str,
    claim_text: str,
    a_turn_idx: int,
    c_turn_idx: int,
    a_context: str,
    c_context: str,
) -> str:
    return f"""You are evaluating "Accommodation" in conversation.

Goal:
Decide whether the FUTURE explicit claim C (turn {c_turn_idx}) makes the earlier assumption A (turn {a_turn_idx}) explicit.

Important:
- Resolve coreference using the provided context (e.g., he/they/it).
- We care about whether C explicitly states the substance of A (or a logically equivalent statement).
- Output must be strict JSON only.

Labels:
- "entailed" = C explicitly verifies/states A (or logically entails A).
- "not_entailed" = C does not make A explicit.
- "uncertain" = ambiguous due to missing context.

Earlier assumption A (turn {a_turn_idx}):
{assumption_text}

Context around A:
{a_context}

Future explicit claim C (turn {c_turn_idx}):
{claim_text}

Context around C:
{c_context}

Return JSON with exactly these keys:
{{
  "label": "entailed" | "not_entailed" | "uncertain",
  "confidence": 0.0-1.0,
  "coref_notes": "briefly explain key coreference links, if any",
  "reason": "brief justification focusing on semantics"
}}
"""


# ----------------- main experiment -----------------

def run_episode(
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

    # Pre-index all explicit claims by turn
    claims_by_turn: List[List[Dict[str, Any]]] = []
    for t in turns:
        cps = t.get("explicit_propositions", []) or []
        # keep only text
        out = []
        for c in cps:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                out.append(c)
        claims_by_turn.append(out)

    results = []
    prompts = []
    meta = []  # aligns with prompts

    # Build candidate pairs (A, C) with pruning
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

            # Gather future claim candidates
            cand: List[Tuple[int, str, int]] = []  # (turn_j, claim_text, score)
            j_end = min(len(turns), i + 1 + max_future_turns) if max_future_turns > 0 else len(turns)

            for j in range(i + 1, j_end):
                for c in claims_by_turn[j]:
                    c_text = c.get("text", "").strip()
                    if not c_text:
                        continue
                    score = overlap_score(a_kw, keywords(c_text))
                    if score >= min_overlap:
                        cand.append((j, c_text, score))

            # Keep top-M by overlap score
            cand.sort(key=lambda x: x[2], reverse=True)
            cand = cand[:max_claims_per_assumption]

            # Create prompts for each candidate
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

    # Batch LLM judging
    judged = []
    for s in range(0, len(prompts), batch_size):
        batch_prompts = prompts[s:s+batch_size]
        outs = llm.generate_batch(batch_prompts)
        judged.extend(outs)

    # Aggregate: for each assumption, find earliest entailed claim
    # Key assumptions by (a_turn_idx, a_idx_in_turn, assumption_text)
    by_assumption: Dict[Tuple[int,int,str], List[Dict[str, Any]]] = {}

    for m, out in zip(meta, judged):
        parsed = safe_json_extract(out)
        if parsed is None:
            parsed = {"label": "uncertain", "confidence": 0.0, "coref_notes": "", "reason": "parse_failed"}

        rec = {**m, **parsed, "raw": out}
        key = (m["a_turn_idx"], m["a_idx_in_turn"], m["assumption_text"])
        by_assumption.setdefault(key, []).append(rec)

    accommodated = 0
    lags = []
    detailed = []

    for key, recs in by_assumption.items():
        # sort by claim turn index (earliest future first)
        recs.sort(key=lambda r: r["c_turn_idx"])
        first_entail = next((r for r in recs if r.get("label") == "entailed"), None)

        a_turn_idx, a_idx_in_turn, a_text = key
        a_time = recs[0].get("a_time")

        if first_entail is None:
            detailed.append({
                "assumption_key": {"a_turn_idx": a_turn_idx, "a_idx_in_turn": a_idx_in_turn},
                "assumption_text": a_text,
                "status": "dark_matter",
                "accommodated_by": None,
                "lag_seconds": None,
                "candidates_judged": recs,
            })
            continue

        accommodated += 1
        c_time = first_entail.get("c_time")
        lag = None
        if isinstance(a_time, (int, float)) and isinstance(c_time, (int, float)):
            lag = float(c_time) - float(a_time)
            if lag < 0:
                lag = None

        if lag is not None:
            lags.append(lag)

        detailed.append({
            "assumption_key": {"a_turn_idx": a_turn_idx, "a_idx_in_turn": a_idx_in_turn},
            "assumption_text": a_text,
            "status": "accommodated",
            "accommodated_by": {
                "c_turn_idx": first_entail["c_turn_idx"],
                "claim_text": first_entail["claim_text"],
                "confidence": first_entail.get("confidence", None),
                "coref_notes": first_entail.get("coref_notes", ""),
                "reason": first_entail.get("reason", ""),
            },
            "lag_seconds": lag,
            "candidates_judged": recs,
        })

    total_assumptions = len(by_assumption)
    conversion_rate = (accommodated / total_assumptions) if total_assumptions > 0 else None
    mean_lag = float(sum(lags) / len(lags)) if lags else None

    episode_summary = {
        "episode_id": os.path.splitext(os.path.basename(episode_path))[0],
        "total_assumptions": total_assumptions,
        "accommodated": accommodated,
        "conversion_rate": conversion_rate,
        "mean_lag_seconds": mean_lag,
        "n_lags_measured": len(lags),
        "params": {
            "max_future_turns": max_future_turns,
            "max_claims_per_assumption": max_claims_per_assumption,
            "min_overlap": min_overlap,
            "context_w": context_w,
        },
    }

    save_json(out_path, {"summary": episode_summary, "assumptions": detailed})
    return episode_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/stance_labeled")
    ap.add_argument("--k", type=int, required=True, help="which stance_labeled/{k} folder to read")
    ap.add_argument("--out_dir", type=str, default="data/implicature_flow/entailment_judged")
    ap.add_argument("--glob", type=str, default="*.json")

    # Candidate control
    ap.add_argument("--max_future_turns", type=int, default=200,
                    help="limit search horizon; 0 means all future turns")
    ap.add_argument("--max_claims_per_assumption", type=int, default=16,
                    help="top-M future claims (after pruning) to judge per assumption")
    ap.add_argument("--min_overlap", type=int, default=2,
                    help="minimum keyword overlap between A and C to be considered a candidate")
    ap.add_argument("--context_w", type=int, default=2,
                    help="context window size (turns before/after) to help coref")

    # vLLM / batching
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

    all_summaries = []
    for ep_path in tqdm(paths, desc="Episodes", unit="ep"):
        episode_id = os.path.splitext(os.path.basename(ep_path))[0]
        out_path = os.path.join(args.out_dir, f"{episode_id}.json")

        summary = run_episode(
            episode_path=ep_path,
            llm=llm,
            out_path=out_path,
            max_future_turns=args.max_future_turns,
            max_claims_per_assumption=args.max_claims_per_assumption,
            min_overlap=args.min_overlap,
            context_w=args.context_w,
            batch_size=args.batch_size,
        )
        all_summaries.append(summary)

    # pooled summary
    pooled = {
        "k": args.k,
        "n_episodes": len(all_summaries),
        "total_assumptions": sum(s.get("total_assumptions", 0) for s in all_summaries),
        "accommodated": sum(s.get("accommodated", 0) for s in all_summaries),
    }
    if pooled["total_assumptions"] > 0:
        pooled["conversion_rate"] = pooled["accommodated"] / pooled["total_assumptions"]
    else:
        pooled["conversion_rate"] = None

    save_json(os.path.join(args.out_dir, f"_POOLED_summary_k{args.k}.json"), {
        "pooled": pooled,
        "per_episode": all_summaries
    })


if __name__ == "__main__":
    main()