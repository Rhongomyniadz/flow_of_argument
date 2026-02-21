import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_optional_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = x.strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_offset(x: str) -> Any:
    if x is None:
        return ""
    s = x.strip()
    if not s:
        return ""
    try:
        return int(s)
    except Exception:
        return s


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR adjustment.
    Returns q-values aligned with input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    q_values = [1.0] * m
    prev_q = 1.0

    for i in range(m - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        q = p_values[idx] * m / rank
        q = min(1.0, q, prev_q)
        q_values[idx] = q
        prev_q = q

    return q_values


def safe_mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.fmean(xs))


def safe_median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def sort_episode_key(ep: str) -> Tuple[int, Any]:
    try:
        return (0, int(ep))
    except Exception:
        return (1, ep)


def sort_offset_key(off: Any) -> Tuple[int, Any]:
    if isinstance(off, int):
        return (0, off)
    return (1, str(off))


def rows_to_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pvals = [r["granger_p_value"] for r in rows if r["granger_p_value"] is not None]
    n_rows = len(rows)
    n_testable = len(pvals)
    n_missing = n_rows - n_testable
    n_sig_raw = sum(1 for r in rows if r["sig_raw"])
    n_sig_fdr_global = sum(1 for r in rows if r["sig_fdr_global"])
    n_sig_fdr_episode = sum(1 for r in rows if r["sig_fdr_episode"])
    min_p = min(pvals) if pvals else None
    min_q_global = min((r["q_global"] for r in rows if r["q_global"] is not None), default=None)
    min_q_episode = min((r["q_episode"] for r in rows if r["q_episode"] is not None), default=None)

    return {
        "n_rows": n_rows,
        "n_testable": n_testable,
        "n_missing": n_missing,
        "min_p": min_p,
        "min_q_global": min_q_global,
        "min_q_episode": min_q_episode,
        "mean_p": safe_mean(pvals),
        "median_p": safe_median(pvals),
        "n_sig_raw": n_sig_raw,
        "n_sig_fdr_global": n_sig_fdr_global,
        "n_sig_fdr_episode": n_sig_fdr_episode,
        "sig_raw_rate": ratio(n_sig_raw, n_testable),
        "sig_fdr_global_rate": ratio(n_sig_fdr_global, n_testable),
        "sig_fdr_episode_rate": ratio(n_sig_fdr_episode, n_testable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize long-form Granger CSV with FDR and paper-ready tables.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="experiments/exp2_iceberg/results/granger_longform_prop.csv",
        help="Input long-form CSV with columns: episode,speaker_id,offset,granger_p_value",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/exp2_iceberg/results/summary",
        help="Directory to write summary outputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for raw/FDR flags.")
    parser.add_argument(
        "--min_tests_for_robust",
        type=int,
        default=8,
        help="Minimum non-missing tests required before marking an episode robust.",
    )
    parser.add_argument(
        "--min_sig_for_robust",
        type=int,
        default=3,
        help="Minimum significant tests required before marking an episode robust.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "episode": str(raw.get("episode", "")).strip(),
                "speaker_id": str(raw.get("speaker_id", "")).strip(),
                "offset": parse_offset(raw.get("offset", "")),
                "granger_p_value": parse_optional_float(raw.get("granger_p_value", "")),
                "q_global": None,
                "q_episode": None,
                "sig_raw": False,
                "sig_fdr_global": False,
                "sig_fdr_episode": False,
            }
            rows.append(row)

    # Global BH-FDR over all valid tests.
    global_indices = [i for i, r in enumerate(rows) if r["granger_p_value"] is not None]
    global_p = [rows[i]["granger_p_value"] for i in global_indices]
    global_q = bh_adjust(global_p)
    for i, q in zip(global_indices, global_q):
        rows[i]["q_global"] = q

    # Episode-level BH-FDR.
    per_episode_indices: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["granger_p_value"] is not None:
            per_episode_indices[r["episode"]].append(i)
    for ep, idxs in per_episode_indices.items():
        pvals = [rows[i]["granger_p_value"] for i in idxs]
        qvals = bh_adjust(pvals)
        for i, q in zip(idxs, qvals):
            rows[i]["q_episode"] = q

    alpha = float(args.alpha)
    for r in rows:
        p = r["granger_p_value"]
        qg = r["q_global"]
        qe = r["q_episode"]
        r["sig_raw"] = bool(p is not None and p < alpha)
        r["sig_fdr_global"] = bool(qg is not None and qg < alpha)
        r["sig_fdr_episode"] = bool(qe is not None and qe < alpha)

    # Row-level enriched output.
    enriched_rows: List[Dict[str, Any]] = []
    for r in rows:
        enriched_rows.append(
            {
                "episode": r["episode"],
                "speaker_id": r["speaker_id"],
                "offset": r["offset"],
                "granger_p_value": r["granger_p_value"],
                "q_global": r["q_global"],
                "q_episode": r["q_episode"],
                "sig_raw": int(r["sig_raw"]),
                "sig_fdr_global": int(r["sig_fdr_global"]),
                "sig_fdr_episode": int(r["sig_fdr_episode"]),
            }
        )
    rows_to_csv(
        output_dir / "granger_longform_enriched.csv",
        enriched_rows,
        [
            "episode",
            "speaker_id",
            "offset",
            "granger_p_value",
            "q_global",
            "q_episode",
            "sig_raw",
            "sig_fdr_global",
            "sig_fdr_episode",
        ],
    )

    # Episode-level summary.
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_episode[r["episode"]].append(r)

    episode_rows: List[Dict[str, Any]] = []
    for ep in sorted(by_episode.keys(), key=sort_episode_key):
        grp = by_episode[ep]
        s = summarize_block(grp)
        sig_speakers_raw = {x["speaker_id"] for x in grp if x["sig_raw"]}
        sig_speakers_fdr_global = {x["speaker_id"] for x in grp if x["sig_fdr_global"]}
        sig_speakers_fdr_episode = {x["speaker_id"] for x in grp if x["sig_fdr_episode"]}
        n_testable = s["n_testable"]
        n_sig_raw = s["n_sig_raw"]
        n_sig_fdr_global = s["n_sig_fdr_global"]
        n_sig_fdr_episode = s["n_sig_fdr_episode"]
        episode_rows.append(
            {
                "episode": ep,
                **s,
                "n_speakers": len({x["speaker_id"] for x in grp}),
                "n_sig_speakers_raw": len(sig_speakers_raw),
                "n_sig_speakers_fdr_global": len(sig_speakers_fdr_global),
                "n_sig_speakers_fdr_episode": len(sig_speakers_fdr_episode),
                "robust_raw": int(
                    n_testable >= args.min_tests_for_robust and n_sig_raw >= args.min_sig_for_robust
                ),
                "robust_fdr_global": int(
                    n_testable >= args.min_tests_for_robust and n_sig_fdr_global >= args.min_sig_for_robust
                ),
                "robust_fdr_episode": int(
                    n_testable >= args.min_tests_for_robust and n_sig_fdr_episode >= args.min_sig_for_robust
                ),
            }
        )
    episode_rows.sort(
        key=lambda x: (-x["n_sig_fdr_global"], -x["n_sig_fdr_episode"], -x["n_sig_raw"], sort_episode_key(x["episode"])),
    )
    rows_to_csv(
        output_dir / "summary_by_episode.csv",
        episode_rows,
        [
            "episode",
            "n_rows",
            "n_testable",
            "n_missing",
            "min_p",
            "min_q_global",
            "min_q_episode",
            "mean_p",
            "median_p",
            "n_sig_raw",
            "n_sig_fdr_global",
            "n_sig_fdr_episode",
            "sig_raw_rate",
            "sig_fdr_global_rate",
            "sig_fdr_episode_rate",
            "n_speakers",
            "n_sig_speakers_raw",
            "n_sig_speakers_fdr_global",
            "n_sig_speakers_fdr_episode",
            "robust_raw",
            "robust_fdr_global",
            "robust_fdr_episode",
        ],
    )

    # Direction-level summary (episode + speaker direction).
    by_direction: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_direction[(r["episode"], r["speaker_id"])].append(r)

    direction_rows: List[Dict[str, Any]] = []
    for (ep, spk), grp in by_direction.items():
        s = summarize_block(grp)
        direction_rows.append({"episode": ep, "speaker_id": spk, **s})
    direction_rows.sort(key=lambda x: (x["n_sig_fdr_global"], x["n_sig_raw"]), reverse=True)
    rows_to_csv(
        output_dir / "summary_by_direction.csv",
        direction_rows,
        [
            "episode",
            "speaker_id",
            "n_rows",
            "n_testable",
            "n_missing",
            "min_p",
            "min_q_global",
            "min_q_episode",
            "mean_p",
            "median_p",
            "n_sig_raw",
            "n_sig_fdr_global",
            "n_sig_fdr_episode",
            "sig_raw_rate",
            "sig_fdr_global_rate",
            "sig_fdr_episode_rate",
        ],
    )

    # Offset-level summary.
    by_offset: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_offset[r["offset"]].append(r)

    offset_rows: List[Dict[str, Any]] = []
    for off in sorted(by_offset.keys(), key=sort_offset_key):
        s = summarize_block(by_offset[off])
        offset_rows.append({"offset": off, **s})
    rows_to_csv(
        output_dir / "summary_by_offset.csv",
        offset_rows,
        [
            "offset",
            "n_rows",
            "n_testable",
            "n_missing",
            "min_p",
            "min_q_global",
            "min_q_episode",
            "mean_p",
            "median_p",
            "n_sig_raw",
            "n_sig_fdr_global",
            "n_sig_fdr_episode",
            "sig_raw_rate",
            "sig_fdr_global_rate",
            "sig_fdr_episode_rate",
        ],
    )

    # Overall summary JSON.
    overall = summarize_block(rows)
    episodes = sorted(by_episode.keys(), key=sort_episode_key)
    episode_summary_lookup = {r["episode"]: r for r in episode_rows}

    overall_json = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "alpha": alpha,
        "min_tests_for_robust": args.min_tests_for_robust,
        "min_sig_for_robust": args.min_sig_for_robust,
        **overall,
        "n_episodes_total": len(episodes),
        "n_episodes_with_any_test": sum(1 for ep in episodes if episode_summary_lookup[ep]["n_testable"] > 0),
        "n_episodes_with_any_sig_raw": sum(1 for ep in episodes if episode_summary_lookup[ep]["n_sig_raw"] > 0),
        "n_episodes_with_any_sig_fdr_global": sum(
            1 for ep in episodes if episode_summary_lookup[ep]["n_sig_fdr_global"] > 0
        ),
        "n_episodes_with_any_sig_fdr_episode": sum(
            1 for ep in episodes if episode_summary_lookup[ep]["n_sig_fdr_episode"] > 0
        ),
        "n_episodes_robust_raw": sum(1 for ep in episodes if episode_summary_lookup[ep]["robust_raw"] == 1),
        "n_episodes_robust_fdr_global": sum(
            1 for ep in episodes if episode_summary_lookup[ep]["robust_fdr_global"] == 1
        ),
        "n_episodes_robust_fdr_episode": sum(
            1 for ep in episodes if episode_summary_lookup[ep]["robust_fdr_episode"] == 1
        ),
    }

    with open(output_dir / "summary_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
