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

def analyze_single_episode(json_path):
    """
    Corrected analysis: 
    Step 1: Collect ALL unique assumptions (no entailment filtering)
    Step 2: Identify accommodated assumptions (entailment_score >= 7)
    Step 3: Calculate true conversion rate = accommodated / total_assumptions
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None, f"JSON parsing error: {str(e)[:100]}"
    
    episode_id = robust_get(data, 'episode_id', 'unknown').strip()
    pairs = robust_get(data, 'pairs', [])
    
    if not pairs:
        return None, "No valid pairs"
    
    # === STEP 1: Collect ALL unique assumptions (critical fix!) ===
    all_assumptions = {}  # key: (a_turn, a_idx) -> {a_time, a_turn, text}
    for pair in pairs:
        a_turn = robust_get(pair, 'a_turn_idx', -1)
        a_idx = robust_get(pair, 'a_idx_in_turn', -1)
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
    accommodated = {}  # key: (a_turn, a_idx) -> {earliest_c_time, c_turn}
    for pair in pairs:
        a_turn = robust_get(pair, 'a_turn_idx', -1)
        a_idx = robust_get(pair, 'a_idx_in_turn', -1)
        c_time = robust_get(pair, 'c_time', float('inf'))
        c_turn = robust_get(pair, 'c_turn_idx', None)
        entailment = robust_get(pair, 'entailment_score', 0)
        
        if a_turn == -1 or a_idx == -1 or entailment < 7:
            continue
            
        key = (a_turn, a_idx)
        if key in all_assumptions:  # Ensure it's a known assumption
            # Record earliest claim time
            if key not in accommodated or c_time < accommodated[key]['c_time']:
                accommodated[key] = {
                    'c_time': c_time,
                    'c_turn': c_turn
                }
    
    # === STEP 3: Calculate true metrics ===
    total = len(all_assumptions)
    num_acc = len(accommodated)
    dark_matter = total - num_acc
    conversion_rate = (num_acc / total * 100) if total > 0 else 0
    
    # Calculate lags (only for accommodated assumptions)
    lags = []
    flow_matrix = defaultdict(lambda: defaultdict(list))
    
    for key, acc_info in accommodated.items():
        a_info = all_assumptions[key]
        a_time = a_info['a_time']
        c_time = acc_info['c_time']
        
        if a_time is not None and c_time < float('inf'):
            lag = c_time - a_time
            if lag >= 0:  # Valid lag
                lags.append(lag)
                
                # Build flow matrix for Sankey
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
        'flow_matrix': flow_matrix
    }, None

def generate_sankey(metrics, output_dir):
    """Generate Sankey diagram (silent mode)"""
    flow = metrics['flow_matrix']
    if not flow:
        return None
    
    # Extract all involved turns
    all_turns = sorted(set(
        [src for src in flow.keys()] + 
        [tgt for tgts in flow.values() for tgt in tgts.keys()]
    ))
    if len(all_turns) < 2:
        return None
    
    node_idx = {turn: i for i, turn in enumerate(all_turns)}
    node_labels = [f"Turn {turn}" for turn in all_turns]
    
    # Build links
    links = []
    for src_turn, targets in flow.items():
        for tgt_turn, lags in targets.items():
            src_idx = node_idx[src_turn]
            tgt_idx = node_idx[tgt_turn]
            flow_volume = len(lags)
            avg_lag = np.mean(lags) if lags else 0
            
            links.append({
                'source': src_idx,
                'target': tgt_idx,
                'value': flow_volume,
                'avg_lag': avg_lag
            })
    
    if not links:
        return None
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="lightblue"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            label=[f"{link['value']} assump.<br>avg {link['avg_lag']:.1f}s" for link in links],
            color="rgba(44, 160, 44, 0.7)"
        )
    )])
    
    fig.update_layout(
        title_text=f"Implicature Flow: Episode {metrics['episode_id']}<br>"
                   f"<sup>Conversion Rate: {metrics['conversion_rate']:.1f}% | "
                   f"Mean Lag: {np.mean(metrics['lags']) if metrics['lags'] else 0:.1f}s</sup>",
        font_size=11,
        height=500
    )
    
    path = output_dir / f"sankey_{metrics['episode_id']}.html"
    fig.write_html(path, include_plotlyjs='cdn')
    return path

def batch_analyze(input_dir, output_dir, top_n_sankey=10):
    """
    Batch analysis of implicature flow
    Critical fix: Denominator = ALL unique assumptions, Numerator = assumptions with high-confidence entailment
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find JSON files
    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in directory: '{input_dir}'")
        return None
    
    print(f"\n🚀 Starting analysis of {len(json_files):,} dialogue episodes...\n")
    
    # === PHASE 1: Analyze all episodes ===
    all_metrics = []
    errors = []
    
    for json_file in tqdm(json_files, desc="Analysis Progress", unit="file"):
        metrics, err = analyze_single_episode(json_file)
        if metrics:
            all_metrics.append(metrics)
        else:
            errors.append((json_file.name, err))
    
    # === PHASE 2: Global metrics calculation ===
    total_assump = sum(m['total_assumptions'] for m in all_metrics)
    total_acc = sum(m['accommodated_assumptions'] for m in all_metrics)
    dark_matter = total_assump - total_acc
    global_conv = (total_acc / total_assump * 100) if total_assump > 0 else 0
    dark_ratio = (dark_matter / total_assump * 100) if total_assump > 0 else 0
    
    # Aggregate all lags
    all_lags = [lag for m in all_metrics for lag in m['lags']]
    lag_stats = {
        'mean': float(np.mean(all_lags)) if all_lags else 0.0,
        'median': float(np.median(all_lags)) if all_lags else 0.0,
        'min': float(np.min(all_lags)) if all_lags else 0.0,
        'max': float(np.max(all_lags)) if all_lags else 0.0,
        'std': float(np.std(all_lags)) if all_lags else 0.0
    }
    
    # === PHASE 3: Generate report ===
    report = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'input_directory': str(input_dir),
        'total_episodes_analyzed': len(all_metrics),
        'total_files_processed': len(json_files),
        'episodes_skipped': len(errors),
        'skipped_details': errors[:10],  # Record first 10 errors only
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
    
    # Save report
    report_path = output_path / "implicature_flow_global_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # === PHASE 4: Visualizations ===
    df = pd.DataFrame(report['per_episode_metrics'])
    
    # Conversion rate distribution
    if not df.empty:
        fig1 = px.histogram(
            df, x='conversion_rate', nbins=25,
            title='Conversion Rate Distribution Across Episodes',
            labels={'conversion_rate': 'Conversion Rate (%)'},
            color_discrete_sequence=['#2E86AB']
        )
        fig1.add_vline(x=global_conv, line_dash="dash", line_color="red",
                       annotation_text=f"Global Avg: {global_conv:.1f}%", 
                       annotation_position="top right")
        fig1.write_html(output_path / "conversion_rate_distribution.html", include_plotlyjs='cdn')
    
    # Lag distribution
    if all_lags:
        fig2 = px.histogram(
            x=all_lags, nbins=40,
            title=f'Time-to-Surface Distribution (All Episodes)<br><sup>Mean: {lag_stats["mean"]:.1f}s | Median: {lag_stats["median"]:.1f}s</sup>',
            labels={'x': 'Lag (seconds)', 'y': 'Count'},
            color_discrete_sequence=['#A23B72']
        )
        fig2.write_html(output_path / "lag_distribution.html", include_plotlyjs='cdn')
    
    # Top-N Sankey diagrams (sorted by conversion rate)
    sankey_paths = []
    if all_metrics:
        print(f"\n🎨 Generating Sankey diagrams for top-{top_n_sankey} episodes by conversion rate...")
        top_episodes = sorted(
            [m for m in all_metrics if m['flow_matrix']], 
            key=lambda m: m['conversion_rate'], 
            reverse=True
        )[:top_n_sankey]
        
        for metrics in tqdm(top_episodes, desc="Sankey Generation", unit="diagram", leave=False):
            path = generate_sankey(metrics, output_path)
            if path:
                sankey_paths.append(path.name)
    
    # === PHASE 5: Concise summary output & Saving to MD ===
    
    # Determine insight message
    if dark_ratio > 40:
        insight = "⚠️  High Dark Matter ratio → Dialogue contains many unverified implicit premises, potentially impacting collaboration quality"
    elif dark_ratio < 20:
        insight = "✓  Low Dark Matter ratio → Participants actively explicitize implicit premises, indicating strong collaboration"
    else:
        insight = "→  Moderate Dark Matter ratio → Consistent with natural dialogue patterns where some implicit premises remain unverified"

    # Construct the summary text lines
    summary_lines = [
        "="*70,
        "✅ IMPLICATURE FLOW ANALYSIS COMPLETE",
        "="*70,
        f"✓ Valid episodes:    {len(all_metrics):,} / {len(json_files):,}",
        f"✓ Total assumptions: {total_assump:,}",
        f"✓ Accommodated:      {total_acc:,} ({global_conv:.1f}%)",
        f"✓ Dark Matter:       {dark_matter:,} ({dark_ratio:.1f}%)",
        f"✓ Mean lag:          {lag_stats['mean']:.2f} seconds (median: {lag_stats['median']:.2f}s)",
        "="*70,
        f"\n💡 Insight: {insight}",
        f"\n📁 Output directory: {output_path.absolute()}",
        f"  • Global report: implicature_flow_global_report.json",
        f"  • Conversion rate distribution: conversion_rate_distribution.html",
        f"  • Lag distribution: lag_distribution.html"
    ]

    # Add Sankey info to summary
    if sankey_paths:
        summary_lines.append(f"  • Sankey diagrams: {len(sankey_paths)} (top-{top_n_sankey} episodes)")
        for p in sankey_paths[:3]:
            summary_lines.append(f"      → {p}")
        if len(sankey_paths) > 3:
            summary_lines.append(f"      → ... and {len(sankey_paths)-3} more")
    
    summary_lines.append("\n" + "="*70)

    # Join lines into a single string
    full_summary_text = "\n".join(summary_lines)
    
    # 1. Print to Terminal
    print("\n" + full_summary_text)
    
    # 2. Save to MD file
    md_path = output_path / "analysis_summary.md"
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Analysis Summary\n\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("```text\n")
            f.write(full_summary_text)
            f.write("\n```")
        print(f"📄 Summary log saved to: {md_path.name}")
    except Exception as e:
        print(f"❌ Failed to save summary log: {e}")

    return report

# ==================== EXECUTION ENTRY POINT ====================
if __name__ == "__main__":
    INPUT_DIR = "data/implicature_flow/entailment_pairs_1to10"
    OUTPUT_DIR = "experiments/exp4_implicature_flow/results"
    
    if not Path(INPUT_DIR).exists():
        print(f"❌ Input directory does not exist: {INPUT_DIR}")
        print("💡 Please verify the path or modify the INPUT_DIR variable")
        print(f"   Suggested check: {Path('.').absolute() / INPUT_DIR}")
    else:
        batch_analyze(INPUT_DIR, OUTPUT_DIR, top_n_sankey=10)
