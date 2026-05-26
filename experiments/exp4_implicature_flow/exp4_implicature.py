import warnings
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')


def safe_get(record, key, default=None):
    """Robustly extract values from dict, handling key spacing variants"""
    if not isinstance(record, dict):
        return default
    variants = [key, key.strip(), f"{key} ", f" {key}"]
    for k in variants:
        if k in record:
            return record[k]
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

    episode_id = safe_get(data, 'episode_id', 'unknown')
    if isinstance(episode_id, str):
        episode_id = episode_id.strip()

    pairs = safe_get(data, 'pairs', [])
    if not pairs:
        return None, "No valid pairs"

    # ---- Collect full turn range (min..max) from a_turn and c_turn ----
    turn_idxs = set()
    for pair in pairs:
        at = safe_int(safe_get(pair, 'a_turn_idx', None), None)
        ct = safe_int(safe_get(pair, 'c_turn_idx', None), None)
        if at is not None and at != -1:
            turn_idxs.add(at)
        if ct is not None and ct != -1:
            turn_idxs.add(ct)

    # Try to get N from episode metadata if available (most robust)
    N = safe_int(safe_get(data, 'num_turns', None), None)

    # Some datasets store full turns list
    if N is None:
        turns_list = safe_get(data, 'turns', None)
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
        a_turn = safe_int(safe_get(pair, 'a_turn_idx', -1), -1)
        a_idx = safe_int(safe_get(pair, 'a_idx_in_turn', -1), -1)
        a_time = safe_get(pair, 'a_time', None)
        text = safe_get(pair, 'assumption_text', '')

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
        a_turn = safe_int(safe_get(pair, 'a_turn_idx', -1), -1)
        a_idx = safe_int(safe_get(pair, 'a_idx_in_turn', -1), -1)
        c_turn = safe_int(safe_get(pair, 'c_turn_idx', None), None)
        c_time = safe_get(pair, 'c_time', None)
        c_time_num = float(c_time) if c_time is not None and np.isfinite(float(c_time)) else float('inf')
        entailment = safe_get(pair, 'entailment_score', 0)

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
        'turn_count': len(all_turns_full),
        'total_assumptions': total,
        'accommodated_assumptions': num_acc,
        'dark_matter_count': dark_matter,
        'conversion_rate': conversion_rate,
        'lags': lags,
        'flow_matrix': flow_matrix,
        'all_turns': all_turns_full
    }, None


def build_episode_metrics_frame(report):
    return pd.DataFrame(report["per_episode_metrics"])


def build_lag_samples_frame(all_metrics):
    rows = []
    for metrics in all_metrics:
        for lag in metrics["lags"]:
            rows.append({"episode_id": metrics["episode_id"], "lag_seconds": float(lag)})
    return pd.DataFrame(rows, columns=["episode_id", "lag_seconds"])


def build_flow_edges_frame(all_metrics):
    rows = []
    for metrics in all_metrics:
        for source_turn, targets in metrics["flow_matrix"].items():
            for target_turn, lags in targets.items():
                if not lags:
                    continue
                rows.append(
                    {
                        "episode_id": metrics["episode_id"],
                        "source_turn": int(source_turn),
                        "target_turn": int(target_turn),
                        "count": int(len(lags)),
                        "mean_lag_seconds": float(np.mean(lags)),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["episode_id", "source_turn", "target_turn", "count", "mean_lag_seconds"],
    )


def batch_analyze(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
        'plot_data_outputs': [
            'exp4_episode_metrics.csv',
            'exp4_lag_samples.csv',
            'exp4_flow_edges.csv'
        ],
        'per_episode_metrics': [
            {
                'episode_id': m['episode_id'],
                'turn_count': len(m['all_turns']),
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

    episode_metrics_df = build_episode_metrics_frame(report)
    lag_samples_df = build_lag_samples_frame(all_metrics)
    flow_edges_df = build_flow_edges_frame(all_metrics)
    episode_metrics_df.to_csv(output_path / "exp4_episode_metrics.csv", index=False)
    lag_samples_df.to_csv(output_path / "exp4_lag_samples.csv", index=False)
    flow_edges_df.to_csv(output_path / "exp4_flow_edges.csv", index=False)

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
    print(f"  • Episode metrics: exp4_episode_metrics.csv")
    print(f"  • Lag samples: exp4_lag_samples.csv")
    print(f"  • Flow edges: exp4_flow_edges.csv")
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
        batch_analyze(INPUT_DIR, OUTPUT_DIR)
