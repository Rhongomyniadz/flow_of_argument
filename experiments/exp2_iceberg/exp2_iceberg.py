import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vllm import LLM, SamplingParams

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:
    grangercausalitytests = None


# =========================================================
# LLM Interface (defaults only, tensor_parallel_size=2)
# =========================================================
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
        max_tokens: int = 16,
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


# =========================================================
# Prompt (stance 1..5)
# =========================================================
STANCE_PROMPT = """\
You are a stance judge.

Task:
Given a short dialogue context and the CURRENT TURN, rate the CURRENT TURN's stance toward the immediately preceding speaker's content.

Use a 5-point scale:
5 = clearly agrees / aligns / endorses the prior turn
4 = mostly agrees (minor hedges)
3 = neutral / unclear / independent (no clear agree/disagree)
2 = mostly disagrees / pushes back (mild challenge)
1 = clearly disagrees / rejects / challenges the prior turn

Rules:
- Judge stance relative to the most recent prior turn in the CONTEXT.
- Clarification questions are usually 3 unless they clearly imply rejection.
- Answering a question without evaluation is usually 3.
- Output ONLY the number 1,2,3,4,or 5. No words, no punctuation.

CONTEXT (immediately preceding turns, truncated to a token budget):
{context}

CURRENT TURN:
{cur}

Output: one digit in [1..5]
"""


# =========================================================
# Core helpers
# =========================================================
def build_context_up_to_k_tokens(previous_turn_texts: List[str], max_tokens: int) -> str:
    """
    Uses as many immediate previous turns as fit into `max_tokens` (approx words).
    Keeps most recent turns; truncates the oldest included turn if needed.
    """
    if max_tokens <= 0 or not previous_turn_texts:
        return "(none)"

    token_budget = max_tokens
    kept: List[str] = []

    # walk from most recent backward
    for t in reversed(previous_turn_texts):
        toks = t.split()
        if not toks:
            continue

        if len(toks) <= token_budget:
            kept.append(t)
            token_budget -= len(toks)
        else:
            # take tail of this turn to fill remaining budget
            tail = " ".join(toks[-token_budget:])
            kept.append(tail)
            token_budget = 0

        if token_budget <= 0:
            break

    kept.reverse()

    # label them relative to current
    lines = []
    for i, txt in enumerate(kept):
        # i=0 is oldest in kept, but still preceding
        lines.append(f"[prev-{len(kept)-i}] {txt}")
    return "\n\n".join(lines) if lines else "(none)"


def parse_stance_1to5(s: str) -> int:
    s = (s or "").strip()
    for ch in s:
        if ch in "12345":
            return int(ch)
    return 3


def compute_duration(turn: Dict[str, Any]) -> float:
    if isinstance(turn.get("duration"), (int, float)) and float(turn["duration"]) > 0:
        return float(turn["duration"])
    st = turn.get("startTime")
    et = turn.get("endTime")
    if isinstance(st, (int, float)) and isinstance(et, (int, float)) and float(et) > float(st):
        return float(et) - float(st)
    return 1.0


def compute_time_x(turn: Dict[str, Any], fallback_order: int) -> float:
    st = turn.get("startTime")
    if isinstance(st, (int, float)):
        return float(st)
    return float(fallback_order)


def compute_iceberg(turn: Dict[str, Any]) -> Dict[str, Any]:
    explicit_count = len(turn.get("explicit_propositions", []) or [])
    assumption_count = len(turn.get("assumptions", []) or [])
    duration = compute_duration(turn)
    D = float(explicit_count) / float(max(assumption_count, 1))
    Dnorm = D / float(duration if duration > 0 else 1.0)
    return {
        "explicit_count": explicit_count,
        "assumption_count": assumption_count,
        "duration": duration,
        "D_iceberg": D,
        "D_iceberg_norm": Dnorm,
    }


def lagged_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            x1, y1 = x, y
        else:
            x1, y1 = x[:-lag], y[lag:]
        mask = np.isfinite(x1) & np.isfinite(y1)
        if mask.sum() < 3:
            out[lag] = float("nan")
            continue
        if np.std(x1[mask]) < 1e-12 or np.std(y1[mask]) < 1e-12:
            out[lag] = float("nan")
            continue
        out[lag] = float(np.corrcoef(x1[mask], y1[mask])[0, 1])
    return out


def granger_two_way(iceberg: np.ndarray, stance: np.ndarray, maxlag: int) -> Dict[str, Any]:
    if grangercausalitytests is None:
        return {"available": False, "reason": "statsmodels not available"}

    mask = np.isfinite(iceberg) & np.isfinite(stance)
    iceberg = iceberg[mask]
    stance = stance[mask]

    if len(iceberg) < maxlag + 6:
        return {"available": True, "skipped": True, "reason": "too few points", "n": int(len(iceberg))}

    out: Dict[str, Any] = {"available": True, "skipped": False, "maxlag": int(maxlag), "n": int(len(iceberg))}

    # iceberg -> stance  (y=stance, x=iceberg)
    data_is = np.column_stack([stance, iceberg])
    res_is = grangercausalitytests(data_is, maxlag=maxlag, verbose=False)
    p_is = {int(lag): float(res_is[lag][0]["ssr_ftest"][1]) for lag in res_is}
    out["iceberg_causes_stance_pvals"] = p_is
    out["iceberg_causes_stance_min_p"] = min(p_is.values()) if p_is else None

    # stance -> iceberg  (y=iceberg, x=stance)
    data_si = np.column_stack([iceberg, stance])
    res_si = grangercausalitytests(data_si, maxlag=maxlag, verbose=False)
    p_si = {int(lag): float(res_si[lag][0]["ssr_ftest"][1]) for lag in res_si}
    out["stance_causes_iceberg_pvals"] = p_si
    out["stance_causes_iceberg_min_p"] = min(p_si.values()) if p_si else None

    return out


def roll(v: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(v, np.nan, dtype=float)
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out[i] = float(np.nanmean(v[lo:i + 1]))
    return out


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=Path("data/conversation_moves_labeled"))
    ap.add_argument("--output_dir", type=Path, default=Path("experiments/exp2_iceberg"))
    ap.add_argument("--max_context_tokens", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--maxlag", type=int, default=3)
    ap.add_argument("--rolling_window", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_episode").mkdir(exist_ok=True)
    (args.output_dir / "plots").mkdir(exist_ok=True)

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files in {args.input_dir}")

    llm = LLMInterface()
    summaries = []

    for fp in tqdm(files, desc="Episodes"):
        episode_id = fp.stem
        turns = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(turns, list) or not turns:
            continue

        # sort by time if present, else turn_idx
        if any(t.get("startTime") is not None for t in turns):
            turns.sort(key=lambda t: (t.get("startTime", float("inf")), t.get("turn_idx", 10**9)))
        else:
            turns.sort(key=lambda t: t.get("turn_idx", 10**9))

        history_texts: List[str] = []
        prompts: List[str] = []
        meta: List[Tuple[int, Dict[str, Any]]] = []

        # only evaluate stance for Substantive turns (since iceberg is defined there)
        for idx, t in enumerate(turns):
            cur_text = (t.get("turn_text") or "").strip()
            history_texts.append(cur_text)

            if t.get("turn_type_label") != "Substantive":
                continue

            context = build_context_up_to_k_tokens(history_texts[:-1], max_tokens=args.max_context_tokens)
            prompts.append(STANCE_PROMPT.format(context=context, cur=cur_text))
            meta.append((idx, t))

        # LLM stance
        stance_scores: List[int] = []
        for i in range(0, len(prompts), args.batch_size):
            outs = llm.generate_batch(prompts[i:i + args.batch_size])
            stance_scores.extend([parse_stance_1to5(o) for o in outs])

        rows = []
        x_series = []
        stance_series = []
        iceberg_series = []

        for (idx, t), s in zip(meta, stance_scores):
            ice = compute_iceberg(t)
            x = compute_time_x(t, fallback_order=idx)

            rows.append({
                "episode_id": episode_id,
                "turn_idx": t.get("turn_idx", idx),
                "startTime": t.get("startTime"),
                "endTime": t.get("endTime"),
                **ice,
                "stance_1to5": int(s),
            })

            x_series.append(x)
            stance_series.append(float(s))
            iceberg_series.append(float(ice["D_iceberg_norm"]))

        # write per-episode json
        with (args.output_dir / "per_episode" / f"{episode_id}.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        if len(rows) < 6:
            summaries.append({"episode_id": episode_id, "n_substantive": len(rows), "note": "too few points"})
            continue

        stance_arr = np.array(stance_series, dtype=float)
        iceberg_arr = np.array(iceberg_series, dtype=float)

        mask = np.isfinite(stance_arr) & np.isfinite(iceberg_arr)
        corr = float(np.corrcoef(stance_arr[mask], iceberg_arr[mask])[0, 1]) if mask.sum() >= 3 else float("nan")

        lagcorr_stance_to_iceberg = lagged_corr(stance_arr, iceberg_arr, max_lag=args.maxlag)
        lagcorr_iceberg_to_stance = lagged_corr(iceberg_arr, stance_arr, max_lag=args.maxlag)

        gr = granger_two_way(iceberg_arr, stance_arr, maxlag=args.maxlag)

        summaries.append({
            "episode_id": episode_id,
            "n_substantive": len(rows),
            "pearson_corr(stance_1to5, iceberg_norm)": corr,
            "lagcorr_stance_to_iceberg": lagcorr_stance_to_iceberg,
            "lagcorr_iceberg_to_stance": lagcorr_iceberg_to_stance,
            "granger": gr,
        })

        # Plot two lines on one axis (iceberg z-scored for visibility)
        xs = np.array(x_series, dtype=float)
        stance_sm = roll(stance_arr, args.rolling_window)

        if np.isfinite(iceberg_arr).sum() >= 3 and np.nanstd(iceberg_arr) > 1e-12:
            iceberg_z = (iceberg_arr - np.nanmean(iceberg_arr)) / np.nanstd(iceberg_arr)
        else:
            iceberg_z = iceberg_arr
        iceberg_sm = roll(iceberg_z, args.rolling_window)

        plt.figure(figsize=(11, 4.8))
        plt.plot(xs, stance_sm, label="Stance (1=disagree, 5=agree)")
        plt.plot(xs, iceberg_sm, label="Iceberg (D_norm, z-scored for plot)")
        plt.xlabel("Time (seconds)" if rows[0].get("startTime") is not None else "Turn index")
        plt.ylabel("Value")
        plt.title(f"Episode {episode_id}: Stance & Iceberg over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / f"{episode_id}.png", dpi=200)
        plt.close()

    # Save summary
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "max_context_tokens": args.max_context_tokens,
            "maxlag": args.maxlag,
            "rolling_window": args.rolling_window,
            "episodes_processed": len(summaries),
            "per_episode": summaries,
        }, f, ensure_ascii=False, indent=2)

    logging.info("Done. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
