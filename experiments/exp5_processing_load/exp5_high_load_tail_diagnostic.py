import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RNG_SEED = 42
np.random.seed(RNG_SEED)

RESPONSE_ORDER = ["Backchannel", "Substantive", "Clarification", "Silence/Abandonment"]
RESPONSE_COLORS = {
    "Backchannel": "#4C78A8",
    "Substantive": "#72B7B2",
    "Clarification": "#F58518",
    "Silence/Abandonment": "#E45756",
}


def safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def finite_or_none(value: float) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def load_turn_rows(input_csv: Path) -> List[Dict]:
    rows: List[Dict] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            load = safe_float(raw.get("implicature_load"))
            duration = safe_float(raw.get("duration_sec"))
            new_count = safe_float(raw.get("new_assumption_count"))
            gap = safe_float(raw.get("gap_to_next_sec"))
            response = str(raw.get("next_response_type") or "").strip()

            turn_idx_raw = raw.get("turn_idx")
            try:
                turn_idx = int(turn_idx_raw)
            except Exception:
                turn_idx = None

            rows.append(
                {
                    "episode_id": str(raw.get("episode_id") or ""),
                    "turn_idx": turn_idx,
                    "implicature_load": load,
                    "duration_sec": duration,
                    "new_assumption_count": new_count,
                    "gap_to_next_sec": gap,
                    "next_response_type": response,
                }
            )
    return rows


def to_array(rows: Sequence[Dict], key: str) -> np.ndarray:
    vals = [safe_float(r.get(key)) for r in rows]
    return np.array(vals, dtype=float)


def quantiles(values: np.ndarray, qs: Sequence[float]) -> Dict[str, Optional[float]]:
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) == 0:
        return {str(q): None for q in qs}
    return {str(q): finite_or_none(float(np.quantile(finite_vals, q))) for q in qs}


def basic_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p99": None}
    return {
        "n": int(len(finite_vals)),
        "mean": finite_or_none(float(np.mean(finite_vals))),
        "median": finite_or_none(float(np.median(finite_vals))),
        "p90": finite_or_none(float(np.quantile(finite_vals, 0.90))),
        "p99": finite_or_none(float(np.quantile(finite_vals, 0.99))),
    }


def response_mix(rows: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    counter = Counter()
    for row in rows:
        label = str(row.get("next_response_type") or "").strip()
        if not label:
            continue
        counter[label] += 1
    total = sum(counter.values())
    out: Dict[str, Dict[str, float]] = {}
    for label in RESPONSE_ORDER:
        count = int(counter.get(label, 0))
        proportion = float(count / total) if total > 0 else 0.0
        out[label] = {"count": count, "proportion": proportion}
    for label, count in counter.items():
        if label in out:
            continue
        out[label] = {"count": int(count), "proportion": float(count / total) if total > 0 else 0.0}
    return out


def latency_array(rows: Sequence[Dict]) -> np.ndarray:
    vals = []
    for row in rows:
        gap = safe_float(row.get("gap_to_next_sec"))
        if math.isfinite(gap) and gap >= 0:
            vals.append(gap)
    return np.array(vals, dtype=float)


def duration_array(rows: Sequence[Dict]) -> np.ndarray:
    vals = []
    for row in rows:
        duration = safe_float(row.get("duration_sec"))
        if math.isfinite(duration) and duration >= 0:
            vals.append(duration)
    return np.array(vals, dtype=float)


def new_count_distribution(rows: Sequence[Dict]) -> Dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        val = safe_float(row.get("new_assumption_count"))
        if not math.isfinite(val):
            continue
        counter[str(int(round(val)))] += 1
    return {k: int(v) for k, v in sorted(counter.items(), key=lambda x: int(x[0]))}


def load_bins_summary(
    rows: Sequence[Dict],
    bin_edges: Sequence[float],
    min_n: int = 20,
) -> List[Dict]:
    loads = to_array(rows, "implicature_load")
    gaps = to_array(rows, "gap_to_next_sec")
    mask = np.isfinite(loads) & np.isfinite(gaps) & (gaps >= 0)
    loads = loads[mask]
    gaps = gaps[mask]

    out: List[Dict] = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if math.isinf(right):
            in_bin = (loads >= left) & np.isfinite(loads)
            label = f"{left:g}+"
            center = left + 2.0
        else:
            in_bin = (loads >= left) & (loads < right)
            label = f"{left:g}-{right:g}"
            center = 0.5 * (left + right)
        vals = gaps[in_bin]
        if len(vals) < min_n:
            out.append(
                {
                    "bin_label": label,
                    "bin_left": float(left),
                    "bin_right": None if math.isinf(right) else float(right),
                    "bin_center": float(center),
                    "n": int(len(vals)),
                    "median_latency_sec": None,
                    "q25_latency_sec": None,
                    "q75_latency_sec": None,
                }
            )
            continue
        out.append(
            {
                "bin_label": label,
                "bin_left": float(left),
                "bin_right": None if math.isinf(right) else float(right),
                "bin_center": float(center),
                "n": int(len(vals)),
                "median_latency_sec": finite_or_none(float(np.median(vals))),
                "q25_latency_sec": finite_or_none(float(np.quantile(vals, 0.25))),
                "q75_latency_sec": finite_or_none(float(np.quantile(vals, 0.75))),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, float) and not math.isfinite(v):
                    out[k] = ""
                else:
                    out[k] = v
            writer.writerow(out)


def json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def plot_dashboard(
    rows_all: Sequence[Dict],
    rows_low: Sequence[Dict],
    rows_high: Sequence[Dict],
    threshold: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel 1: load distribution and threshold position.
    loads = to_array(rows_all, "implicature_load")
    loads = loads[np.isfinite(loads)]
    q_hi = float(np.quantile(loads, 0.999)) if len(loads) > 0 else threshold
    max_x = max(threshold * 1.05, q_hi)
    bins = np.linspace(0.0, max_x, 65)
    sns.histplot(loads, bins=bins, color="#4C78A8", alpha=0.75, edgecolor=None, ax=ax1)
    ax1.axvline(threshold, color="#E45756", linewidth=2.0, linestyle="--", label=f"threshold = {threshold:g}")
    ax1.set_yscale("log")
    ax1.set_title("Implicature Load Distribution (log-count)")
    ax1.set_xlabel("Implicature load")
    ax1.set_ylabel("Turn count (log scale)")
    share = (len(rows_high) / len(rows_all)) if len(rows_all) > 0 else 0.0
    ax1.text(
        0.97,
        0.95,
        f"load > {threshold:g}: {len(rows_high):,}/{len(rows_all):,} ({share:.2%})",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#DDDDDD"},
    )
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: response composition for low vs high groups.
    groups = [rows_low, rows_high]
    labels = [f"<= {threshold:g}", f"> {threshold:g}"]
    totals = [max(1, len(groups[0])), max(1, len(groups[1]))]
    bottom = np.zeros(2, dtype=float)
    for resp in RESPONSE_ORDER:
        perc = []
        for grp, total in zip(groups, totals):
            count = sum(1 for r in grp if str(r.get("next_response_type") or "").strip() == resp)
            perc.append(100.0 * count / total)
        color = RESPONSE_COLORS.get(resp, "#999999")
        ax2.bar(labels, perc, bottom=bottom, label=resp, color=color, alpha=0.9)
        bottom += np.array(perc, dtype=float)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percentage of turns")
    ax2.set_title("Next Response Type Mix: Tail vs Non-tail")
    ax2.legend(loc="upper right", frameon=True)
    for i, total in enumerate([len(rows_low), len(rows_high)]):
        ax2.text(i, 102, f"n={total:,}", ha="center", va="bottom", fontsize=9)

    # Panel 3: median latency and IQR by load bin.
    bin_edges = [0, 5, 10, 15, 20, 25, 35, 50, 75, 120, float("inf")]
    summaries = load_bins_summary(rows_all, bin_edges=bin_edges, min_n=20)
    valid = [r for r in summaries if r.get("median_latency_sec") is not None]
    x = np.arange(len(valid), dtype=float)
    med = np.array([float(r["median_latency_sec"]) for r in valid], dtype=float)
    q25 = np.array([float(r["q25_latency_sec"]) for r in valid], dtype=float)
    q75 = np.array([float(r["q75_latency_sec"]) for r in valid], dtype=float)
    nvals = np.array([int(r["n"]) for r in valid], dtype=float)
    labels_valid = [str(r["bin_label"]) for r in valid]

    ax3.fill_between(x, q25, q75, color="#E45756", alpha=0.22, label="IQR")
    ax3.plot(x, med, color="#E45756", linewidth=2.2, marker="o", markersize=4, label="Median latency")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_valid, rotation=35, ha="right")
    ax3.set_ylabel("Response latency (sec)")
    ax3.set_xlabel("Implicature load bins")
    ax3.set_title("Response Latency Trend Across Load Bins")
    ax3.grid(alpha=0.2)
    ax3b = ax3.twinx()
    ax3b.bar(x, nvals, color="#4C78A8", alpha=0.18, width=0.82, label="Turn count")
    ax3b.set_ylabel("Turn count")
    handles_a, labels_a = ax3.get_legend_handles_labels()
    handles_b, labels_b = ax3b.get_legend_handles_labels()
    ax3.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left", frameon=True)

    # Panel 4: high-load decomposition (duration vs new assumption count).
    high_duration = []
    high_new_count = []
    high_resp = []
    for row in rows_high:
        duration = safe_float(row.get("duration_sec"))
        new_count = safe_float(row.get("new_assumption_count"))
        response = str(row.get("next_response_type") or "").strip()
        if math.isfinite(duration) and duration > 0 and math.isfinite(new_count):
            high_duration.append(float(duration))
            high_new_count.append(int(round(new_count)))
            high_resp.append(response)
    high_duration_arr = np.array(high_duration, dtype=float)
    high_new_arr = np.array(high_new_count, dtype=int)

    if len(high_duration_arr) > 0:
        for resp in RESPONSE_ORDER:
            mask = np.array([r == resp for r in high_resp], dtype=bool)
            if int(np.sum(mask)) == 0:
                continue
            xvals = high_new_arr[mask].astype(float) + np.random.uniform(-0.22, 0.22, size=int(np.sum(mask)))
            yvals = high_duration_arr[mask]
            ax4.scatter(
                xvals,
                yvals,
                s=20,
                alpha=0.35,
                color=RESPONSE_COLORS.get(resp, "#999999"),
                linewidths=0,
                label=resp,
                rasterized=True,
            )

        uniq = sorted(set(high_new_arr.tolist()))
        med_x = []
        med_y = []
        for val in uniq:
            vals = high_duration_arr[high_new_arr == val]
            if len(vals) == 0:
                continue
            med_x.append(float(val))
            med_y.append(float(np.median(vals)))
        if len(med_x) >= 2:
            ax4.plot(med_x, med_y, color="black", linewidth=2.0, marker="o", markersize=4, label="Median duration")

        n10 = int(np.sum(high_new_arr == 10))
        share10 = (n10 / len(high_new_arr)) if len(high_new_arr) > 0 else 0.0
        ax4.text(
            0.97,
            0.95,
            f"new_assumption_count=10: {n10}/{len(high_new_arr)} ({share10:.1%})",
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#DDDDDD"},
        )
    ax4.set_yscale("log")
    ax4.set_xlabel("New assumption count in turn")
    ax4.set_ylabel("Turn duration (sec, log scale)")
    ax4.set_title(f"High-load ({threshold:g}+) Internal Structure")
    ax4.grid(alpha=0.18)
    handles4, labels4 = ax4.get_legend_handles_labels()
    if handles4:
        ax4.legend(loc="upper left", frameon=True)

    fig.suptitle(f"Experiment 5 Tail Diagnostic: Implicature Load > {threshold:g}", y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail diagnostic for high implicature load in Exp5")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="experiments/exp5_processing_load/results/exp5_turn_level_features.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/exp5_processing_load/results/high_load_tail",
    )
    parser.add_argument("--threshold", type=float, default=25.0, help="High-load threshold (strict >).")
    parser.add_argument("--top_k_rows", type=int, default=300, help="Save top-K highest load rows in CSV.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_turn_rows(input_csv)
    finite_load_rows = [
        r
        for r in rows
        if isinstance(r.get("implicature_load"), (int, float)) and math.isfinite(float(r["implicature_load"]))
    ]

    high_rows = [r for r in finite_load_rows if float(r["implicature_load"]) > float(args.threshold)]
    low_rows = [r for r in finite_load_rows if float(r["implicature_load"]) <= float(args.threshold)]

    # Save tail rows sorted by load descending.
    high_rows_sorted = sorted(high_rows, key=lambda r: float(r["implicature_load"]), reverse=True)
    top_rows = high_rows_sorted[: max(0, int(args.top_k_rows))]
    fields = [
        "episode_id",
        "turn_idx",
        "implicature_load",
        "duration_sec",
        "new_assumption_count",
        "gap_to_next_sec",
        "next_response_type",
    ]
    write_csv(output_dir / "exp5_high_load_rows.csv", top_rows, fields)

    # Save load-bin latency table.
    load_bin_rows = load_bins_summary(
        finite_load_rows,
        bin_edges=[0, 5, 10, 15, 20, 25, 35, 50, 75, 120, float("inf")],
        min_n=20,
    )
    write_csv(
        output_dir / "exp5_load_bin_latency_summary.csv",
        load_bin_rows,
        [
            "bin_label",
            "bin_left",
            "bin_right",
            "bin_center",
            "n",
            "median_latency_sec",
            "q25_latency_sec",
            "q75_latency_sec",
        ],
    )

    plot_dashboard(
        rows_all=finite_load_rows,
        rows_low=low_rows,
        rows_high=high_rows,
        threshold=float(args.threshold),
        out_path=output_dir / "exp5_high_load_tail_dashboard.png",
    )

    loads = to_array(finite_load_rows, "implicature_load")
    high_new_dist = new_count_distribution(high_rows)
    high_new_total = sum(high_new_dist.values())
    high_new_10_share = (
        float(high_new_dist.get("10", 0) / high_new_total) if high_new_total > 0 else 0.0
    )

    notes: List[str] = []
    notes.append(
        f"High-load turns are load > {float(args.threshold):g}. They account for "
        f"{(len(high_rows) / max(1, len(finite_load_rows))):.2%} of finite-load turns."
    )
    if high_new_10_share >= 0.5:
        notes.append(
            "In the high-load tail, new_assumption_count=10 dominates. This suggests the tail is mainly driven by long duration, not tiny denominators."
        )
    if len(high_rows) > 0:
        high_dur = duration_array(high_rows)
        low_dur = duration_array(low_rows)
        if len(high_dur) > 0 and len(low_dur) > 0:
            ratio = float(np.median(high_dur) / max(1e-9, np.median(low_dur)))
            notes.append(f"Median duration in high-load turns is about {ratio:.2f}x of non-tail turns.")

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "threshold": float(args.threshold),
        "n_total_rows": len(rows),
        "n_finite_load_rows": len(finite_load_rows),
        "n_high_rows": len(high_rows),
        "high_share": float(len(high_rows) / max(1, len(finite_load_rows))),
        "load_quantiles": quantiles(loads, [0.9, 0.95, 0.975, 0.99, 0.995, 0.999]),
        "response_mix": {
            "non_tail_load_lte_threshold": response_mix(low_rows),
            "tail_load_gt_threshold": response_mix(high_rows),
        },
        "latency_stats_seconds": {
            "non_tail": basic_stats(latency_array(low_rows)),
            "tail": basic_stats(latency_array(high_rows)),
        },
        "duration_stats_seconds": {
            "non_tail": basic_stats(duration_array(low_rows)),
            "tail": basic_stats(duration_array(high_rows)),
        },
        "high_load_new_assumption_count_distribution": high_new_dist,
        "diagnostic_notes": notes,
        "outputs": [
            "exp5_high_load_tail_dashboard.png",
            "exp5_high_load_rows.csv",
            "exp5_load_bin_latency_summary.csv",
            "exp5_high_load_tail_summary.json",
        ],
    }

    (output_dir / "exp5_high_load_tail_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()

