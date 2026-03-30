import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper")


def robust_get(d, key, default=None):
    """Robustly extract values from dict, handling key spacing variants"""
    if not isinstance(d, dict):
        return default
    variants = [key, key.strip(), f"{key} ", f" {key}"]
    for k in variants:
        if k in d:
            return d[k]
    return default


def safe_int(x, default=None):
    """Convert x to int robustly (handles str/np.int/float)."""
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            if np.isfinite(x):
                return int(x)
            return default
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return default
            # allow "2" or "2.0"
            return int(float(s)) if "." in s else int(s)
        return default
    except Exception:
        return default


def analyze_single_episode(json_path):
    """
    Step 1: Collect ALL unique assumptions (no entailment filtering)
    Step 2: Identify accommodated assumptions (entailment_score >= 7)
    Step 3: Calculate conversion rate = accommodated / total_assumptions

    UPDATE:
    - all_turns: include full min..max range so turns with no flow still appear
    - cast all indices to int to avoid link dropping
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None, f"JSON parsing error: {str(e)[:100]}"

    episode_id = robust_get(data, 'episode_id', 'unknown')
    if isinstance(episode_id, str):
        episode_id = episode_id.strip()

    pairs = robust_get(data, 'pairs', [])
    if not pairs:
        return None, "No valid pairs"

    # ---- Collect full turn range (min..max) from a_turn and c_turn ----
    turn_idxs = set()
    for pair in pairs:
        at = safe_int(robust_get(pair, 'a_turn_idx', None), None)
        ct = safe_int(robust_get(pair, 'c_turn_idx', None), None)
        if at is not None and at != -1:
            turn_idxs.add(at)
        if ct is not None and ct != -1:
            turn_idxs.add(ct)

    # Try to get N from episode metadata if available (most robust)
    N = safe_int(robust_get(data, 'num_turns', None), None)

    # Some datasets store full turns list
    if N is None:
        turns_list = robust_get(data, 'turns', None)
        if isinstance(turns_list, list) and len(turns_list) > 0:
            N = len(turns_list)

    # Fallback: infer from maximum index observed in pairs
    if N is None:
        N = max(turn_idxs) if turn_idxs else 0

    # Force 1..N (even if some turns never appear in pairs)
    all_turns_full = list(range(1, N + 1)) if N >= 1 else []
    # === STEP 1: Collect ALL unique assumptions ===
    all_assumptions = {}  # key: (a_turn, a_idx)
    for pair in pairs:
        a_turn = safe_int(robust_get(pair, 'a_turn_idx', -1), -1)
        a_idx = safe_int(robust_get(pair, 'a_idx_in_turn', -1), -1)
        a_time = robust_get(pair, 'a_time', None)
        text = robust_get(pair, 'assumption_text', '')

        if a_turn == -1 or a_idx == -1:
            continue

        key = (a_turn, a_idx)
        if key not in all_assumptions:
            all_assumptions[key] = {
                'a_time': a_time,
                'a_turn': a_turn,
                'text': text
            }

    if not all_assumptions:
        return None, "No valid assumptions"

    # === STEP 2: Identify accommodated assumptions (entailment_score >= 7) ===
    accommodated = {}  # key: (a_turn, a_idx) -> earliest c_time + c_turn
    for pair in pairs:
        a_turn = safe_int(robust_get(pair, 'a_turn_idx', -1), -1)
        a_idx = safe_int(robust_get(pair, 'a_idx_in_turn', -1), -1)
        c_turn = safe_int(robust_get(pair, 'c_turn_idx', None), None)
        c_time = robust_get(pair, 'c_time', None)
        c_time_num = float(c_time) if c_time is not None and np.isfinite(float(c_time)) else float('inf')
        entailment = robust_get(pair, 'entailment_score', 0)

        if a_turn == -1 or a_idx == -1 or entailment < 7:
            continue

        key = (a_turn, a_idx)
        if key in all_assumptions:
            prev_time = accommodated.get(key, {}).get('c_time_num', float('inf'))
            if key not in accommodated or c_time_num < prev_time:
                accommodated[key] = {
                    'c_time': c_time,
                    'c_time_num': c_time_num,
                    'c_turn': c_turn
                }

    # === STEP 3: Metrics ===
    total = len(all_assumptions)
    num_acc = len(accommodated)
    dark_matter = total - num_acc
    conversion_rate = (num_acc / total * 100) if total > 0 else 0

    lags = []
    flow_matrix = defaultdict(lambda: defaultdict(list))

    for key, acc_info in accommodated.items():
        a_info = all_assumptions[key]
        a_time = a_info['a_time']
        c_time = acc_info.get('c_time_num', float('inf'))

        if a_time is not None and c_time < float('inf'):
            lag = c_time - a_time
            if lag >= 0:
                lags.append(lag)

                src_turn = a_info['a_turn']
                tgt_turn = acc_info['c_turn']
                if src_turn is not None and tgt_turn is not None:
                    flow_matrix[src_turn][tgt_turn].append(lag)

    return {
        'episode_id': episode_id,
        'total_assumptions': total,
        'accommodated_assumptions': num_acc,
        'dark_matter_count': dark_matter,
        'conversion_rate': conversion_rate,
        'lags': lags,
        'flow_matrix': flow_matrix,
        'all_turns': all_turns_full
    }, None


def draw_flow_curve(ax, x0, y0, x1, y1, color, lw, alpha=0.4):
    path_x = np.linspace(x0, x1, 80)
    t = np.linspace(0, 1, 80)
    ease = 3 * t**2 - 2 * t**3
    path_y = y0 + (y1 - y0) * ease
    ax.plot(path_x, path_y, color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')


def generate_sankey(metrics, output_dir):
    flow = metrics.get('flow_matrix', None)
    all_turns = metrics.get('all_turns', None)
    if not all_turns:
        return None

    all_turns = sorted(all_turns)
    n = len(all_turns)
    if n == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, max(7.5, 0.38 * n + 3.0)))
    ys = [0.5] if n == 1 else list(np.linspace(0.95, 0.05, n))
    left_pos = {t: ys[i] for i, t in enumerate(all_turns)}
    right_pos = {t: ys[i] for i, t in enumerate(all_turns)}
    colors = sns.color_palette("husl", n)
    turn_colors = {t: colors[i] for i, t in enumerate(all_turns)}

    edges = []
    if flow:
        for src_turn, targets in flow.items():
            try:
                src_turn = int(src_turn)
            except Exception:
                continue
            for tgt_turn, lags in targets.items():
                try:
                    tgt_turn = int(tgt_turn)
                except Exception:
                    continue
                if src_turn in left_pos and tgt_turn in right_pos and lags:
                    edges.append((src_turn, tgt_turn, len(lags), float(np.mean(lags))))

    max_count = max([e[2] for e in edges], default=1)
    for src_turn, tgt_turn, count, avg_lag in sorted(edges, key=lambda x: (x[0], x[1])):
        lw = 1.5 + 8.0 * (count / max_count)
        draw_flow_curve(ax, 0.15, left_pos[src_turn], 0.85, right_pos[tgt_turn], turn_colors[src_turn], lw)
        if count >= max_count * 0.45:
            ax.text(0.5, (left_pos[src_turn] + right_pos[tgt_turn]) / 2, f"{count}",
                    ha="center", va="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    for t in all_turns:
        ax.add_patch(plt.Rectangle((0.03, left_pos[t] - 0.015), 0.07, 0.03, color=turn_colors[t], alpha=0.45))
        ax.add_patch(plt.Rectangle((0.90, right_pos[t] - 0.015), 0.07, 0.03, color="lightgray", alpha=0.9))
        ax.text(0.02, left_pos[t], f"A{t}", ha="right", va="center", fontsize=9)
        ax.text(0.98, right_pos[t], f"C{t}", ha="left", va="center", fontsize=9)

    ax.text(0.065, 1.02, "Implicit Pool", transform=ax.transAxes, ha="center", va="bottom", fontsize=13, weight="bold")
    ax.text(0.935, 1.02, "Explicit Record", transform=ax.transAxes, ha="center", va="bottom", fontsize=13, weight="bold")
    ax.set_title(
        f"Episode {metrics['episode_id']} | Conversion {metrics['conversion_rate']:.1f}% | Mean lag {np.mean(metrics['lags']) if metrics['lags'] else 0:.1f}s",
        fontsize=15,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sankey_{metrics['episode_id']}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return path


def save_conversion_histogram(df, global_conv, out_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.histplot(df['conversion_rate'], bins=25, color="#4C78A8", edgecolor="white", ax=ax)
    ax.axvline(global_conv, color="#E45756", linestyle="--", linewidth=2, label=f"Global avg = {global_conv:.1f}%")
    ax.set_title("Episode-Level Conversion Rate Distribution", fontsize=15)
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("Episode Count")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_lag_histogram(all_lags, lag_stats, out_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.histplot(all_lags, bins=40, color="#72B7B2", edgecolor="white", ax=ax)
    ax.axvline(lag_stats["median"], color="#DD8452", linestyle="--", linewidth=2, label=f"Median = {lag_stats['median']:.1f}s")
    ax.set_title("Time-to-Surface Distribution", fontsize=15)
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_episode_scatter(df, out_path):
    plot_df = df.copy()
    plot_df["dark_ratio"] = 100.0 * plot_df["dark_matter"] / plot_df["total_assumptions"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    sns.scatterplot(
        data=plot_df,
        x="conversion_rate",
        y="mean_lag",
        size="total_assumptions",
        hue="dark_ratio",
        palette="viridis",
        sizes=(20, 180),
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Episode-Level Accommodation Profile", fontsize=15)
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("Mean Time-to-Surface (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_turn_lag_heatmap(all_metrics, out_path):
    rows = []
    for m in all_metrics:
        for src_turn, targets in m["flow_matrix"].items():
            for tgt_turn, lags in targets.items():
                rows.append({"source_turn": int(src_turn), "target_turn": int(tgt_turn), "count": len(lags)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    heat = df.groupby(["source_turn", "target_turn"], as_index=False)["count"].sum()
    src_cap = int(np.ceil(np.quantile(heat["source_turn"], 0.95)))
    tgt_cap = int(np.ceil(np.quantile(heat["target_turn"], 0.95)))
    heat = heat[(heat["source_turn"] <= src_cap) & (heat["target_turn"] <= tgt_cap)].copy()
    pivot = heat.pivot(index="source_turn", columns="target_turn", values="count").fillna(0)
    if pivot.empty:
        return
    plot_vals = np.log1p(pivot)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    cmap = sns.color_palette("mako", as_cmap=True)
    cmap.set_under("white")
    sns.heatmap(
        plot_vals,
        cmap=cmap,
        vmin=1e-9,
        ax=ax,
        cbar_kws={"label": "log(1 + accommodation count)"},
    )
    ax.set_title("Where Assumptions Surface Across Turns", fontsize=15)
    ax.set_xlabel("Explicit Claim Turn")
    ax.set_ylabel("Assumption Turn")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_global_summary_flow(report, out_path):
    total = report["global_metrics"]["total_assumptions"]
    accommodated = report["global_metrics"]["total_accommodated"]
    dark = report["global_metrics"]["dark_matter_count"]
    mean_lag = report["global_metrics"]["lag_distribution_seconds"]["mean"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.add_patch(plt.Rectangle((0.08, 0.35), 0.22, 0.3, color="#4C78A8", alpha=0.35))
    ax.add_patch(plt.Rectangle((0.70, 0.58), 0.22, 0.2, color="#59A14F", alpha=0.45))
    ax.add_patch(plt.Rectangle((0.70, 0.12), 0.22, 0.2, color="#E15759", alpha=0.45))
    ax.text(0.19, 0.5, f"Implicit Pool\n{total:,}", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.81, 0.68, f"Explicit Record\n{accommodated:,}", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.81, 0.22, f"Dark Matter\n{dark:,}", ha="center", va="center", fontsize=14, weight="bold")
    draw_flow_curve(ax, 0.30, 0.52, 0.70, 0.68, "#59A14F", 10)
    draw_flow_curve(ax, 0.30, 0.46, 0.70, 0.22, "#E15759", 10)
    ax.text(0.50, 0.67, f"{report['global_metrics']['conversion_rate_percent']:.1f}%",
            ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.text(0.50, 0.24, f"{report['global_metrics']['dark_matter_ratio_percent']:.1f}%",
            ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_title("Global Implicature Flow Summary", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def batch_analyze(input_dir, output_dir, sankey_limit=50):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    sankey_output_path = output_path.parent / "episode_sankey_html"
    output_path.mkdir(parents=True, exist_ok=True)
    sankey_output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in directory: '{input_dir}'")
        return None

    print(f"\nStarting analysis of {len(json_files):,} dialogue episodes...\n")

    all_metrics = []
    errors = []

    for json_file in tqdm(json_files, desc="Analysis Progress", unit="file"):
        metrics, err = analyze_single_episode(json_file)
        if metrics:
            all_metrics.append(metrics)
        else:
            errors.append((json_file.name, err))

    total_assump = sum(m['total_assumptions'] for m in all_metrics)
    total_acc = sum(m['accommodated_assumptions'] for m in all_metrics)
    dark_matter = total_assump - total_acc
    global_conv = (total_acc / total_assump * 100) if total_assump > 0 else 0
    dark_ratio = (dark_matter / total_assump * 100) if total_assump > 0 else 0

    all_lags = [lag for m in all_metrics for lag in m['lags']]
    lag_stats = {
        'mean': float(np.mean(all_lags)) if all_lags else 0.0,
        'median': float(np.median(all_lags)) if all_lags else 0.0,
        'min': float(np.min(all_lags)) if all_lags else 0.0,
        'max': float(np.max(all_lags)) if all_lags else 0.0,
        'std': float(np.std(all_lags)) if all_lags else 0.0
    }

    report = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'input_directory': str(input_dir),
        'total_episodes_analyzed': len(all_metrics),
        'total_files_processed': len(json_files),
        'episodes_skipped': len(errors),
        'skipped_details': errors[:10],
        'global_metrics': {
            'total_assumptions': total_assump,
            'total_accommodated': total_acc,
            'dark_matter_count': dark_matter,
            'conversion_rate_percent': round(global_conv, 2),
            'dark_matter_ratio_percent': round(dark_ratio, 2),
            'lag_distribution_seconds': lag_stats
        },
        'per_episode_metrics': [
            {
                'episode_id': m['episode_id'],
                'total_assumptions': m['total_assumptions'],
                'accommodated': m['accommodated_assumptions'],
                'dark_matter': m['dark_matter_count'],
                'conversion_rate': round(m['conversion_rate'], 2),
                'mean_lag': round(np.mean(m['lags']) if m['lags'] else 0, 2),
                'num_lags': len(m['lags'])
            }
            for m in all_metrics
        ]
    }

    report_path = output_path / "implicature_flow_global_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(report['per_episode_metrics'])

    if not df.empty:
        save_conversion_histogram(df, global_conv, output_path / "conversion_rate_distribution.png")
        save_episode_scatter(df, output_path / "episode_accommodation_profile.png")

    if all_lags:
        save_lag_histogram(all_lags, lag_stats, output_path / "lag_distribution.png")

    if all_metrics:
        save_turn_lag_heatmap(all_metrics, output_path / "turn_lag_heatmap.png")
        save_global_summary_flow(report, output_path / "global_implicature_flow_summary.png")

    sankey_paths = []
    if all_metrics:
        print(f"\nGenerating Sankey diagrams for first {sankey_limit} episodes (sorted by ID)...")

        # allow episodes even if flow empty, still can draw (ghost links will show columns)
        sorted_episodes = sorted(all_metrics, key=lambda m: str(m['episode_id']))
        target_episodes = sorted_episodes[:sankey_limit]

        for metrics in tqdm(target_episodes, desc="Sankey Generation", unit="diagram", leave=False):
            path = generate_sankey(metrics, sankey_output_path)
            if path:
                sankey_paths.append(str(path.relative_to(output_path.parent)))

    if dark_ratio > 40:
        insight = "⚠️  High Dark Matter ratio → Dialogue contains many unverified implicit premises, potentially impacting collaboration quality"
    elif dark_ratio < 20:
        insight = "✓  Low Dark Matter ratio → Participants actively explicitize implicit premises, indicating strong collaboration"
    else:
        insight = "→  Moderate Dark Matter ratio → Consistent with natural dialogue patterns where some implicit premises remain unverified"

    print("\n" + "=" * 70)
    print("IMPLICATURE FLOW ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"✓ Valid episodes:    {len(all_metrics):,} / {len(json_files):,}")
    print(f"✓ Total assumptions: {total_assump:,}")
    print(f"✓ Accommodated:      {total_acc:,} ({global_conv:.1f}%)")
    print(f"✓ Dark Matter:       {dark_matter:,} ({dark_ratio:.1f}%)")
    print(f"✓ Mean lag:          {lag_stats['mean']:.2f} seconds (median: {lag_stats['median']:.2f}s)")
    print("=" * 70)
    print(f"\nInsight: {insight}")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"  • Global report: implicature_flow_global_report.json")
    print(f"  • Conversion rate distribution: conversion_rate_distribution.png")
    print(f"  • Lag distribution: lag_distribution.png")
    print(f"  • Episode accommodation profile: episode_accommodation_profile.png")
    print(f"  • Turn-lag heatmap: turn_lag_heatmap.png")
    print(f"  • Global summary flow: global_implicature_flow_summary.png")

    if sankey_paths:
        print(f"  • Sankey diagrams: {len(sankey_paths)} generated (Limit: first {sankey_limit} IDs)")
        for p in sankey_paths[:3]:
            print(f"      → {p}")
        if len(sankey_paths) > 3:
            print(f"      → ... and {len(sankey_paths) - 3} more")
    print("\n" + "=" * 70)

    return report


if __name__ == "__main__":
    INPUT_DIR = "data/implicature_flow/entailment_pairs_1to10"
    OUTPUT_DIR = "experiments/exp4_implicature_flow/results"

    if not Path(INPUT_DIR).exists():
        print(f"❌ Input directory does not exist: {INPUT_DIR}")
        print("💡 Please verify the path or modify the INPUT_DIR variable")
        print(f"   Suggested check: {Path('.').absolute() / INPUT_DIR}")
    else:
        batch_analyze(INPUT_DIR, OUTPUT_DIR, sankey_limit=50)
