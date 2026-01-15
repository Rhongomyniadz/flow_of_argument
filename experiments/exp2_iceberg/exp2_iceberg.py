import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:
    grangercausalitytests = None


def load_episode(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} is not a JSON list")
    return obj


def sort_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if any("startTime" in t and t.get("startTime") is not None for t in turns):
        return sorted(
            turns,
            key=lambda t: (
                t.get("startTime", float("inf")),
                t.get("turn_idx", 10**9),
            ),
        )
    return sorted(turns, key=lambda t: t.get("turn_idx", 10**9))


def rolling_mean(arr: List[float], w: int) -> List[float]:
    out = []
    for i in range(len(arr)):
        window = arr[max(0, i - w + 1) : i + 1]
        out.append(float(np.nanmean(window)))
    return out


def compute_granger(drop: np.ndarray, arg: np.ndarray, maxlag: int) -> Dict[str, Any]:
    if grangercausalitytests is None:
        return {"available": False, "reason": "statsmodels not available"}

    if len(drop) < (maxlag + 5):
        return {"available": True, "skipped": True, "reason": "too few points", "n": int(len(drop))}

    data = np.column_stack([arg, drop])  # [y, x]
    try:
        res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        pvals = {int(lag): float(res[lag][0]["ssr_ftest"][1]) for lag in res}
        return {"available": True, "skipped": False, "pvals": pvals, "min_p": min(pvals.values())}
    except Exception as e:
        return {"available": True, "skipped": False, "error": str(e)}


def is_substantive(turn: Dict[str, Any]) -> bool:
    return (turn.get("turn_type_label", "") or "").strip() == "Substantive"


def stance_from_move(move_label: str) -> Tuple[str, float]:
    """
    Option A: richer mapping from conversation_move_label -> stance_prob (P(Agreement)).
    This is a pragmatic proxy for "agreement vs disagreement" without an LLM.
    """
    m = (move_label or "").strip()

    # Strong alignment
    if m == "Agree / Align":
        return "Agreement", 0.90

    # Strong conflict
    if m == "Correction / Challenge":
        return "Disagreement", 0.10

    # Answers are usually cooperative/constructive
    if m == "Answer":
        return "Agreement", 0.65

    # Asserting/elaborating often continues the world; mildly cooperative by default
    if m == "Assert / Elaborate":
        return "Agreement", 0.55

    # Clarification requests often reflect a mismatch / missing assumptions -> mild disagreement signal
    if m in ("Clarification Request (Generic)", "Clarification Request (Specific)"):
        return "Disagreement", 0.45

    # Self-correction is not stance; keep it neutral
    if m == "Self-Correction":
        return "Neutral", 0.50

    # Topic shift often breaks coherence; mildly disagreement-ish (not conflict, but not alignment)
    if m == "Topic Shift":
        return "Disagreement", 0.40

    # Stonewalling is disengagement / low cooperation
    if m == "Stonewalling / Non-Response":
        return "Disagreement", 0.35

    # Unknown / missing
    return "Neutral", 0.50


def compute_episode_metrics(turns: List[Dict[str, Any]], episode_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    order = 0

    for t in turns:
        if not is_substantive(t):
            continue

        explicit_count = len(t.get("explicit_propositions", []) or [])
        assumption_count = len(t.get("assumptions", []) or [])

        # duration: prefer duration; else end-start; else 1.0
        if isinstance(t.get("duration"), (int, float)) and float(t["duration"]) > 0:
            duration = float(t["duration"])
        else:
            st = t.get("startTime")
            et = t.get("endTime")
            if isinstance(st, (int, float)) and isinstance(et, (int, float)) and float(et) > float(st):
                duration = float(et) - float(st)
            else:
                duration = 1.0

        D_iceberg = float(explicit_count) / float(max(assumption_count, 1))
        D_norm = D_iceberg / float(duration if duration > 0 else 1.0)

        move = (t.get("conversation_move_label") or "").strip()
        stance_label, stance_prob = stance_from_move(move)

        rows.append(
            {
                "episode_id": episode_id,
                "turn_idx": t.get("turn_idx", None),
                "speaker_id": t.get("speaker_id", None),
                "startTime": t.get("startTime", None),
                "endTime": t.get("endTime", None),
                "duration": duration,
                "explicit_count": explicit_count,
                "assumption_count": assumption_count,
                "D_iceberg": D_iceberg,
                "D_iceberg_norm": D_norm,
                "stance_label": stance_label,
                "stance_prob": stance_prob,
                "conversation_move_label": move,
                "order": order,
            }
        )
        order += 1

    return rows


def plot_episode(rows: List[Dict[str, Any]], episode_id: str, out_path: Path, rolling_window: int) -> None:
    if not rows:
        return

    x_axis = [
        float(r["startTime"]) if r.get("startTime") is not None else float(r["order"])
        for r in rows
    ]

    # Use stance_label buckets
    y_agree = [
        r["D_iceberg_norm"] if r["stance_label"] == "Agreement" else np.nan
        for r in rows
    ]
    y_disagree = [
        r["D_iceberg_norm"] if r["stance_label"] == "Disagreement" else np.nan
        for r in rows
    ]

    # Skip plot if both are empty
    if not (np.isfinite(np.array(y_agree)).any() or np.isfinite(np.array(y_disagree)).any()):
        return

    agree_line = rolling_mean(y_agree, rolling_window)
    disagree_line = rolling_mean(y_disagree, rolling_window)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(x_axis, agree_line, label="Agreement")
    plt.plot(x_axis, disagree_line, label="Disagreement")
    plt.xlabel("Time (seconds)" if rows[0].get("startTime") is not None else "Turn order")
    plt.ylabel("D_iceberg / duration")
    plt.title(f"Episode {episode_id}: Iceberg Ratio (Normalized) by Stance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def episode_summary(rows: List[Dict[str, Any]], maxlag: int) -> Dict[str, Any]:
    x = np.array([r["stance_prob"] for r in rows], dtype=float)
    y = np.array([r["D_iceberg_norm"] for r in rows], dtype=float)

    if len(x) >= 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float("nan")

    # "Argument" proxy: Correction/Challenge on substantive turns
    arg = np.array([1.0 if r["conversation_move_label"] == "Correction / Challenge" else 0.0 for r in rows], dtype=float)

    # Sudden drop in iceberg ratio (positive when D_norm decreases)
    drop = np.zeros_like(y)
    if len(y) > 1:
        drop[1:] = np.maximum(0.0, y[:-1] - y[1:])

    gr = compute_granger(drop, arg, maxlag=maxlag)
    return {
        "n_substantive": int(len(rows)),
        "pearson_corr(stance_prob, D_norm)": corr,
        "granger_drop_causes_argument": gr,
    }


def run_exp2(input_dir: Path, output_dir: Path, maxlag: int, rolling_window: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_episode_dir = output_dir / "per_episode"
    plots_dir = output_dir / "plots"
    per_episode_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No .json files found in {input_dir}")

    summaries = []
    all_stance = []
    all_dnorm = []

    for path in tqdm(files, desc="Episodes"):
        episode_id = path.stem
        turns = sort_turns(load_episode(path))
        rows = compute_episode_metrics(turns, episode_id)

        # Save per-episode
        with (per_episode_dir / f"{episode_id}.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        # Plot
        plot_episode(rows, episode_id, plots_dir / f"{episode_id}.png", rolling_window)

        # Summary
        s = episode_summary(rows, maxlag=maxlag)
        s["episode_id"] = episode_id
        summaries.append(s)

        for r in rows:
            all_stance.append(r["stance_prob"])
            all_dnorm.append(r["D_iceberg_norm"])

    # Global correlation across all episodes
    all_stance = np.array(all_stance, dtype=float)
    all_dnorm = np.array(all_dnorm, dtype=float)
    if len(all_stance) >= 2 and np.std(all_stance) > 1e-12 and np.std(all_dnorm) > 1e-12:
        global_corr = float(np.corrcoef(all_stance, all_dnorm)[0, 1])
    else:
        global_corr = float("nan")

    summary_obj = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "episodes_processed": len(summaries),
        "global_pearson_corr(stance_prob, D_norm)": global_corr,
        "per_episode": summaries,
        "notes": {
            "D_iceberg": "explicit_count / max(assumption_count, 1)",
            "D_norm": "D_iceberg / duration_seconds",
            "stance_prob": "Move-based mapping (Option A)",
            "argument_proxy": "Correction / Challenge on Substantive turns",
            "granger_test": "Does iceberg drop predict upcoming argument proxy?",
        },
        "stance_mapping": {
            "Agree / Align": 0.90,
            "Answer": 0.65,
            "Assert / Elaborate": 0.55,
            "Clarification Request (Generic)": 0.45,
            "Clarification Request (Specific)": 0.45,
            "Self-Correction": 0.50,
            "Topic Shift": 0.40,
            "Stonewalling / Non-Response": 0.35,
            "Correction / Challenge": 0.10,
            "default": 0.50
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    with (output_dir / "summary.csv").open("w", encoding="utf-8") as f:
        f.write("episode_id,n_substantive,pearson_corr,granger_min_p\n")
        for s in summaries:
            gr = s.get("granger_drop_causes_argument", {})
            min_p = gr.get("min_p", "")
            f.write(f"{s['episode_id']},{s['n_substantive']},{s['pearson_corr(stance_prob, D_norm)']},{min_p}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=Path("data/conversation_moves_labeled"))
    ap.add_argument("--output_dir", type=Path, default=Path("experiments/exp2_iceberg"))
    ap.add_argument("--maxlag", type=int, default=3)
    ap.add_argument("--rolling_window", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_exp2(args.input_dir, args.output_dir, args.maxlag, args.rolling_window)


if __name__ == "__main__":
    main()
