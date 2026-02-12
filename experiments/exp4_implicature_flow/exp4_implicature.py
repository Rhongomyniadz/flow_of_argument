import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


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
        c_time = robust_get(pair, 'c_time', float('inf'))
        entailment = robust_get(pair, 'entailment_score', 0)

        if a_turn == -1 or a_idx == -1 or entailment < 7:
            continue

        key = (a_turn, a_idx)
        if key in all_assumptions:
            if key not in accommodated or c_time < accommodated[key]['c_time']:
                accommodated[key] = {
                    'c_time': c_time,
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
        c_time = acc_info['c_time']

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


def hsla_color(i, n, alpha=0.65):
    """Deterministic distinct colors using HSL wheel."""
    if n <= 0:
        return f"hsla(0,60%,45%,{alpha})"
    h = int((360.0 * i) / n) % 360
    return f"hsla({h},60%,45%,{alpha})"


def generate_sankey(metrics, output_dir):
    """
    Bipartite Sankey with:
    - strict ordering (1..N) on both sides
    - equal node heights (via invisible balancing links)
    - link colors determined by source node (A Turn i)
    """
    flow = metrics.get('flow_matrix', None)
    all_turns = metrics.get('all_turns', None)
    if not all_turns:
        return None

    # enforce sorted 1..N
    all_turns = sorted(all_turns)
    n = len(all_turns)
    if n == 0:
        return None

    # ---------- nodes ----------
    left_labels  = [f"A Turn {t}" for t in all_turns]
    right_labels = [f"C Turn {t}" for t in all_turns]
    node_labels  = left_labels + right_labels

    left_idx  = {t: i for i, t in enumerate(all_turns)}
    right_idx = {t: i + n for i, t in enumerate(all_turns)}

    # fixed two columns
    node_x = [0.01] * n + [0.99] * n

    # fixed y order (top -> bottom) exactly 1..N
    # use centers, evenly spaced; leave a tiny margin
    margin = 0.02
    if n == 1:
        ys = [0.5]
    else:
        ys = [margin + (1 - 2 * margin) * (i / (n - 1)) for i in range(n)]
    node_y = ys + ys

    # ---------- colors ----------
    def hsla_color(i, n, alpha=0.70):
        h = int((360.0 * i) / max(n, 1)) % 360
        return f"hsla({h},60%,45%,{alpha})"

    src_colors = {t: hsla_color(i, n, alpha=0.70) for i, t in enumerate(all_turns)}
    node_colors = ([hsla_color(i, n, alpha=0.25) for i in range(n)] + ["rgba(200,200,200,0.55)"] * n)

    # ---------- build REAL links ----------
    link_source, link_target, link_value, link_label, link_color = [], [], [], [], []

    # compute out/in totals for balancing
    out_tot = {t: 0.0 for t in all_turns}
    in_tot  = {t: 0.0 for t in all_turns}

    if flow:
        for src_turn, targets in flow.items():
            # robust cast
            try:
                src_turn = int(src_turn)
            except Exception:
                continue
            if src_turn not in left_idx:
                continue

            for tgt_turn, lags in targets.items():
                try:
                    tgt_turn = int(tgt_turn)
                except Exception:
                    continue
                if tgt_turn not in right_idx:
                    continue

                v = float(len(lags))
                if v <= 0:
                    continue

                avg_lag = float(np.mean(lags)) if lags else 0.0

                link_source.append(left_idx[src_turn])
                link_target.append(right_idx[tgt_turn])
                link_value.append(v)
                link_label.append(f"{int(v)} assump.<br>avg {avg_lag:.1f}s")
                link_color.append(src_colors[src_turn])

                out_tot[src_turn] += v
                in_tot[tgt_turn]  += v

    # ---------- BALANCING LINKS (invisible) to force equal node heights ----------
    # Goal: every A node has total out = T, every C node has total in = T
    # Choose minimal T so we only add extra when needed
    T = max(max(out_tot.values()), max(in_tot.values()), 1.0)

    # deficits
    defA = [(t, T - out_tot[t]) for t in all_turns]
    defC = [(t, T - in_tot[t])  for t in all_turns]

    # only keep positive deficits
    defA = [(t, d) for t, d in defA if d > 1e-9]
    defC = [(t, d) for t, d in defC if d > 1e-9]

    # greedy matching deficits: send invisible flow from A deficits to C deficits
    i = j = 0
    while i < len(defA) and j < len(defC):
        a_t, a_d = defA[i]
        c_t, c_d = defC[j]
        x = min(a_d, c_d)

        # invisible link (no label, transparent)
        link_source.append(left_idx[a_t])
        link_target.append(right_idx[c_t])
        link_value.append(x)
        link_label.append("")
        link_color.append("rgba(0,0,0,0)")

        a_d -= x
        c_d -= x

        if a_d <= 1e-9:
            i += 1
        else:
            defA[i] = (a_t, a_d)

        if c_d <= 1e-9:
            j += 1
        else:
            defC[j] = (c_t, c_d)

    # If there are absolutely no links (rare), still draw a tiny invisible diagonal
    if len(link_value) == 0:
        for t in all_turns:
            link_source.append(left_idx[t])
            link_target.append(right_idx[t])
            link_value.append(1.0)
            link_label.append("")
            link_color.append("rgba(0,0,0,0)")

    # ---------- plot ----------
    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",   # ✅ DO NOT reorder nodes
        node=dict(
            pad=10,
            thickness=16,
            line=dict(color="black", width=0.5),
            label=node_labels,
            x=node_x,
            y=node_y,
            color=node_colors
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            label=link_label,
            color=link_color
        )
    )])

    fig.update_layout(
        title_text=f"Implicature Flow (Bipartite): Episode {metrics['episode_id']}<br>"
                   f"<sup>Conversion Rate: {metrics['conversion_rate']:.1f}% | "
                   f"Mean Lag: {np.mean(metrics['lags']) if metrics['lags'] else 0:.1f}s</sup>",
        font_size=11,
        height=max(600, 22 * n + 240)
    )

    path = output_dir / f"sankey_{metrics['episode_id']}.html"
    fig.write_html(path, include_plotlyjs='cdn')
    return path


def batch_analyze(input_dir, output_dir, sankey_limit=50):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in directory: '{input_dir}'")
        return None

    print(f"\n🚀 Starting analysis of {len(json_files):,} dialogue episodes...\n")

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
        fig1 = px.histogram(
            df, x='conversion_rate', nbins=25,
            title='Conversion Rate Distribution Across Episodes',
            labels={'conversion_rate': 'Conversion Rate (%)'}
        )
        fig1.add_vline(
            x=global_conv, line_dash="dash", line_color="red",
            annotation_text=f"Global Avg: {global_conv:.1f}%",
            annotation_position="top right"
        )
        fig1.write_html(output_path / "conversion_rate_distribution.html", include_plotlyjs='cdn')

    if all_lags:
        fig2 = px.histogram(
            x=all_lags, nbins=40,
            title=f'Time-to-Surface Distribution (All Episodes)<br><sup>Mean: {lag_stats["mean"]:.1f}s | Median: {lag_stats["median"]:.1f}s</sup>',
            labels={'x': 'Lag (seconds)', 'y': 'Count'}
        )
        fig2.write_html(output_path / "lag_distribution.html", include_plotlyjs='cdn')

    sankey_paths = []
    if all_metrics:
        print(f"\n🎨 Generating Sankey diagrams for first {sankey_limit} episodes (sorted by ID)...")

        # allow episodes even if flow empty, still can draw (ghost links will show columns)
        sorted_episodes = sorted(all_metrics, key=lambda m: str(m['episode_id']))
        target_episodes = sorted_episodes[:sankey_limit]

        for metrics in tqdm(target_episodes, desc="Sankey Generation", unit="diagram", leave=False):
            path = generate_sankey(metrics, output_path)
            if path:
                sankey_paths.append(path.name)

    if dark_ratio > 40:
        insight = "⚠️  High Dark Matter ratio → Dialogue contains many unverified implicit premises, potentially impacting collaboration quality"
    elif dark_ratio < 20:
        insight = "✓  Low Dark Matter ratio → Participants actively explicitize implicit premises, indicating strong collaboration"
    else:
        insight = "→  Moderate Dark Matter ratio → Consistent with natural dialogue patterns where some implicit premises remain unverified"

    print("\n" + "=" * 70)
    print("✅ IMPLICATURE FLOW ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"✓ Valid episodes:    {len(all_metrics):,} / {len(json_files):,}")
    print(f"✓ Total assumptions: {total_assump:,}")
    print(f"✓ Accommodated:      {total_acc:,} ({global_conv:.1f}%)")
    print(f"✓ Dark Matter:       {dark_matter:,} ({dark_ratio:.1f}%)")
    print(f"✓ Mean lag:          {lag_stats['mean']:.2f} seconds (median: {lag_stats['median']:.2f}s)")
    print("=" * 70)
    print(f"\n💡 Insight: {insight}")
    print(f"\n📁 Output directory: {output_path.absolute()}")
    print(f"  • Global report: implicature_flow_global_report.json")
    print(f"  • Conversion rate distribution: conversion_rate_distribution.html")
    print(f"  • Lag distribution: lag_distribution.html")

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
