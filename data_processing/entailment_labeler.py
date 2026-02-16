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
        default_max_tokens: int = 256,  # 默认设小一点，提高速度
    ):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir,
            tensor_parallel_size=tensor_parallel_size,
        )
        # 保存基础配置，但max_tokens现在可以在generate时动态调整
        self.base_params_config = {
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        self.default_max_tokens = default_max_tokens

    def generate_batch(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[str]:
        # 如果未指定，使用默认值
        tokens_to_gen = max_tokens if max_tokens is not None else self.default_max_tokens

        # 每次生成创建一个新的 SamplingParams 对象
        params = SamplingParams(
            max_tokens=tokens_to_gen,
            **self.base_params_config
        )

        out = self.llm.generate(prompts, params)
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
    """
    if not text:
        return None

    # 尝试找到第一个 {
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


def build_entailment_prompt(
    assumption_text: str,
    claim_text: str,
    a_turn_idx: int,
    c_turn_idx: int,
    a_context: str,
    c_context: str,
) -> str:
    return f"""You are an expert annotator evaluating conversational accommodation.

Goal:
Rate how strongly the FUTURE explicit claim C (turn {c_turn_idx}) makes the earlier assumption A (turn {a_turn_idx}) explicit.

Earlier assumption A (turn {a_turn_idx}):
{assumption_text}

Context around A:
{a_context}

Future explicit claim C (turn {c_turn_idx}):
{claim_text}

Context around C:
{c_context}

Instructions:
1. Resolve coreference using context.
2. Determine if C verifies the substance of A.
3. Use the Entailment Scale (1-10).

Entailment Scale:
1 = Clearly unrelated or contradicts A
3 = Weak topical relation only
5 = Partially supports A but missing key content
7 = Strongly entails A (core meaning stated)
9 = Directly and explicitly verifies A
10 = Explicit verification with no ambiguity

OUTPUT FORMAT RULES (STRICT):
- Return ONLY a raw JSON object. Do not use Markdown code blocks (no ```json).
- The JSON must contain EXACTLY two keys: "entailment_score" and "confidence".
- **DO NOT** include "reasoning", "explanation", "thoughts", or any other keys.
- **DO NOT** output any text before or after the JSON.

Example Output:
{{
  "entailment_score": 7,
  "confidence": 0.9
}}

Your Output:
"""


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


def run_episode_labeling(
    episode_path: str,
    llm: LLMInterface,
    out_path: str,
    max_future_turns: int,
    max_claims_per_assumption: int,
    min_overlap: int,
    context_w: int,
    batch_size: int,
    retry_max_tokens: int = 2000,  # 重试时使用的更大 token 限制
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

                # ============================
                # ✅ 关键更新：在每个 pair 中保留 a_turn / c_turn 的 speaker_id
                # ============================
                a_turn_speaker_id = t.get("speaker_id", None)
                c_turn_speaker_id = turns[j].get("speaker_id", None)

                # 仍然保留 fallback（方便 downstream 使用/对齐旧字段）
                a_speaker_id = a_turn_speaker_id or t.get("speaker") or "UNKNOWN"
                c_speaker_id = c_turn_speaker_id or turns[j].get("speaker") or "UNKNOWN"

                # same_speaker：优先用 speaker_id 比较；缺失时退回 fallback
                if a_turn_speaker_id is not None and c_turn_speaker_id is not None:
                    same_speaker = (a_turn_speaker_id == c_turn_speaker_id)
                else:
                    same_speaker = (a_speaker_id == c_speaker_id)

                meta.append({
                    "a_turn_idx": i,
                    "a_time": a_time,
                    "a_idx_in_turn": a_idx_in_turn,

                    # ✅ 新增：严格保留 turn 内 speaker_id
                    "a_turn_speaker_id": a_turn_speaker_id,
                    "c_turn_speaker_id": c_turn_speaker_id,

                    # （保留旧字段，兼容原有逻辑/下游代码）
                    "a_speaker_id": a_speaker_id,
                    "assumption_text": a_text,

                    "c_turn_idx": j,
                    "c_time": turn_start_time(turns[j]),
                    "c_speaker_id": c_speaker_id,
                    "claim_text": c_text,

                    "same_speaker": same_speaker,
                    "overlap_score": score,
                })

    # =========================================================================
    # REVISED BATCH GENERATION WITH RETRY LOGIC
    # =========================================================================

    # 1. Initialize result storage
    final_outputs = [""] * len(prompts)

    # 2. First Pass: Standard Generation
    for s in range(0, len(prompts), batch_size):
        end = min(s + batch_size, len(prompts))
        batch_prompts = prompts[s:end]
        # 使用默认 max_tokens (例如 128 或 256)
        batch_outs = llm.generate_batch(batch_prompts)
        for idx, out_txt in enumerate(batch_outs):
            final_outputs[s + idx] = out_txt

    # 3. Validation & Identification of Failures
    failed_indices = []
    parsed_results = []

    for idx, out_txt in enumerate(final_outputs):
        parsed = safe_json_extract(out_txt)
        parsed_results.append(parsed)
        # 如果解析失败（None）或者解析出的JSON缺少关键字段
        if parsed is None or "entailment_score" not in parsed:
            failed_indices.append(idx)

    # 4. Retry Pass: Process failures with higher max_tokens
    if failed_indices:
        print(f"Episode {os.path.basename(episode_path)}: Retrying {len(failed_indices)}/{len(prompts)} items...")

        # 收集需要重试的 Prompt
        retry_prompts = [prompts[i] for i in failed_indices]

        # 分批重试，使用更大的 max_tokens
        retry_outputs_list = []
        for s in range(0, len(retry_prompts), batch_size):
            end = min(s + batch_size, len(retry_prompts))
            batch_p = retry_prompts[s:end]
            # 关键：传入更大的 max_tokens
            batch_o = llm.generate_batch(batch_p, max_tokens=retry_max_tokens)
            retry_outputs_list.extend(batch_o)

        # 5. Merge Retry Results
        for i, original_idx in enumerate(failed_indices):
            new_text = retry_outputs_list[i]
            final_outputs[original_idx] = new_text  # Update raw text

            # 再次尝试解析
            new_parsed = safe_json_extract(new_text)
            parsed_results[original_idx] = new_parsed  # Update parsed object

    # =========================================================================

    # Record scored candidate pairs
    pairs: List[Dict[str, Any]] = []
    for m, parsed, raw_txt in zip(meta, parsed_results, final_outputs):

        if parsed is None:
            parsed = {
                "entailment_score": 0,
                "confidence": 0.0,
            }

        clean_parsed = {
            "entailment_score": parsed.get("entailment_score", 0),
            "confidence": parsed.get("confidence", 0.0)
        }

        # Normalize types
        try:
            score = int(clean_parsed["entailment_score"])
        except:
            score = 0
        score = max(0, min(10, score))

        try:
            conf = float(clean_parsed["confidence"])
        except:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        pairs.append({
            **m,
            "entailment_score": score,
            "confidence": conf,
            "raw": raw_txt,  # 保留 raw text 用于 debug，即使是废话
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

    return {
        "episode_id": episode_id,
        "n_pairs": len(pairs),
        "n_retries": len(failed_indices),
        "out_path": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/stance_labeled")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--glob", type=str, default="*.json")
    ap.add_argument("--out_dir", type=str, default="data/implicature_flow/entailment_pairs_1to10")

    ap.add_argument("--max_future_turns", type=int, default=30)
    ap.add_argument("--max_claims_per_assumption", type=int, default=15)
    ap.add_argument("--min_overlap", type=int, default=2)
    ap.add_argument("--context_w", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--retry_max_tokens", type=int, default=4000)

    args = ap.parse_args()

    llm = LLMInterface(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        default_max_tokens=args.max_tokens,
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
            retry_max_tokens=args.retry_max_tokens,  # Pass retry limit
        ))

    save_json(os.path.join(args.out_dir, f"_LABELING_META_k{args.k}.json"), {
        "k": args.k,
        "n_episodes": len(meta),
        "per_episode": meta
    })


if __name__ == "__main__":
    main()
