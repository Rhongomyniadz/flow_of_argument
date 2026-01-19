import os, json, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Silence statsmodels warning spam (but keep tqdm progress)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="verbose is deprecated since functions should not print results"
)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_episode(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_counts(turn):
    explicit = turn.get("explicit_propositions", []) or []
    implicit = turn.get("assumptions", []) or []
    return len(explicit), len(implicit)

def get_duration(turn):
    if isinstance(turn.get("duration"), (int, float)) and turn["duration"] > 0:
        return float(turn["duration"])
    st, et = turn.get("startTime"), turn.get("endTime")
    if isinstance(st, (int, float)) and isinstance(et, (int, float)) and et > st:
        return float(et - st)
    return None

def compute_df(turns, eps=1e-6):
    rows = []
    for t in turns:
        if t.get("turn_type_label") != "Substantive":
            continue

        exp_cnt, imp_cnt = get_counts(t)
        dur = get_duration(t)
        if dur is None or dur <= 0:
            continue

        stance = t.get("stance_5pt", None)
        if stance is None:
            continue

        # D_iceberg normalized by duration: (explicit/implicit)/sec
        iceberg = (exp_cnt / (imp_cnt + eps)) / dur

        rows.append({
            "episode_id": str(t.get("episode_id", "")),
            "turn_idx": t.get("turn_idx"),
            "startTime": t.get("startTime"),
            "duration": dur,
            "explicit_cnt": exp_cnt,
            "implicit_cnt": imp_cnt,
            "stance_5pt": float(stance),
            "iceberg_norm": float(iceberg),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # episode-local timeline for plotting
    if df["startTime"].notna().any():
        df = df.sort_values(["startTime", "turn_idx"], na_position="last").reset_index(drop=True)
        df["t_local"] = df["startTime"].fillna(df["turn_idx"].astype(float))
    else:
        df = df.sort_values("turn_idx").reset_index(drop=True)
        df["t_local"] = df["turn_idx"].astype(float)

    return df

def rolling_mean(series, win):
    if win <= 1:
        return series
    return series.rolling(win, min_periods=1).mean()

def plot_episode_stance_vs_iceberg(df, out_png, smooth=5):
    """
    Per-episode: one plot with two lines (different colors by default):
      - stance (left y-axis)
      - D_iceberg_norm (right y-axis)
    """
    x = df["t_local"]
    stance = rolling_mean(df["stance_5pt"], smooth)
    iceberg = rolling_mean(df["iceberg_norm"], smooth)

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(x, stance, color="red", linewidth=2.0, label="Stance (1=disagree, 5=agree)")
    ax1.set_ylabel("Stance (1–5)")
    ax1.set_xlabel("Time (startTime if available else turn_idx)")

    ax2 = ax1.twinx()
    ax2.plot(x, iceberg, label="D_iceberg_norm = (explicit/implicit)/sec")
    ax2.set_ylabel("D_iceberg_norm")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def pooled_lagged_corr(pooled_df, max_lag=10):
    """
    Lagged correlation on pooled data (no cross-episode):
    corr( stance(t), iceberg(t-lag) ) pooled over episodes.
    """
    out = {}
    # group once for efficiency
    groups = [(eid, d.sort_values("t_local")) for eid, d in pooled_df.groupby("episode_id", sort=False)]

    for lag in range(0, max_lag + 1):
        xs, ys = [], []
        for _, d in groups:
            if lag == 0:
                x = d["iceberg_norm"].values
                y = d["stance_5pt"].values
            else:
                if len(d) <= lag:
                    continue
                x = d["iceberg_norm"].values[:-lag]
                y = d["stance_5pt"].values[lag:]
            if len(x) >= 3:
                xs.append(x)
                ys.append(y)

        if not xs:
            out[lag] = None
            continue

        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        out[lag] = float(np.corrcoef(X.astype(float), Y.astype(float))[0, 1]) if len(X) >= 3 else None

    return out

def pooled_granger(df_all, max_lag=3):
    """
    ONE Granger test pooled over all episodes, boundary-safe:
    we drop the first `max_lag` rows of each episode so lagged terms
    never cross an episode boundary.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        return None

    segs = []
    for _, d in df_all.groupby("episode_id", sort=False):
        d = d.sort_values("t_local").reset_index(drop=True)
        if len(d) <= (max_lag + 5):
            continue
        d2 = d.iloc[max_lag:].copy()
        seg = np.column_stack([d2["stance_5pt"].values, d2["iceberg_norm"].values])
        segs.append(seg)

    if not segs:
        return None

    arr = np.vstack(segs)
    if len(arr) < (max_lag + 10):
        return None

    lag = int(max_lag)
    while lag >= 1:
        try:
            res = grangercausalitytests(arr, maxlag=lag, verbose=False)
            out = {}
            for L, r in res.items():
                out[int(L)] = float(r[0]["ssr_ftest"][1])
            return out
        except ValueError:
            lag -= 1

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--in_dir", type=str, default="data/stance_labeled")
    ap.add_argument("--out_dir", type=str, default="experiments/exp2_iceberg/plots")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--max_lag", type=int, default=3)       # pooled Granger
    ap.add_argument("--max_corr_lag", type=int, default=10)  # pooled lagged corr curve
    args = ap.parse_args()

    in_k_dir = os.path.join(args.in_dir, str(args.k))
    paths = sorted(glob.glob(os.path.join(in_k_dir, "*.json")))
    ensure_dir(args.out_dir)

    all_dfs = []
    per_episode_meta = []

    # Per-episode plots + collect pooled df
    for p in tqdm(paths, desc="Episodes", unit="ep"):
        episode_id = os.path.splitext(os.path.basename(p))[0]
        turns = load_episode(p)
        df = compute_df(turns)

        if df.empty:
            per_episode_meta.append({
                "episode_id": episode_id,
                "n_substantive": 0,
                "plot": None
            })
            continue

        df = df.assign(episode_id=episode_id)
        all_dfs.append(df)

        out_plot = os.path.join(args.out_dir, f"{episode_id}_stance_vs_iceberg_k{args.k}.png")
        plot_episode_stance_vs_iceberg(df, out_plot, smooth=args.smooth)

        per_episode_meta.append({
            "episode_id": episode_id,
            "n_substantive": int(len(df)),
            "plot": out_plot
        })

    pooled_summary_path = os.path.join(args.out_dir, f"pooled_summary_k{args.k}.json")
    if not all_dfs:
        save_json(pooled_summary_path, {
            "k": args.k,
            "error": "no substantive turns found",
            "per_episode": per_episode_meta
        })
        return

    pooled = pd.concat(all_dfs, ignore_index=True)

    # pooled stats with progress bars
    lagcorr = pooled_lagged_corr(pooled, max_lag=args.max_corr_lag)
    granger = pooled_granger(pooled, max_lag=args.max_lag)

    save_json(pooled_summary_path, {
        "k": args.k,
        "n_substantive_total": int(len(pooled)),
        "lagged_corr(stance_t, iceberg_t-lag)_pooled": lagcorr,
        "granger_pvals(iceberg->stance)_pooled": granger,
        "per_episode": per_episode_meta
    })

if __name__ == "__main__":
    main()
