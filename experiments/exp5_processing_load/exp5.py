import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
from tqdm.auto import tqdm

RNG_SEED = 42
np.random.seed(RNG_SEED)
    
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def maybe_tqdm(iterable, enabled: bool = True, **kwargs):
    return tqdm(iterable, **kwargs) if enabled else iterable


def safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def finite_or_none(x: float) -> Optional[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def extract_unique_texts(raw: object) -> List[str]:
    if not isinstance(raw, list):
        return []

    seen = set()
    texts: List[str] = []
    for item in raw:
        txt = ""
        if isinstance(item, dict):
            txt = str(item.get("text", "")).strip()
        elif isinstance(item, str):
            txt = item.strip()
        if not txt:
            continue

        norm = normalize_text(txt)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        texts.append(norm)
    return texts


def extract_assumption_texts(turn: Dict) -> List[str]:
    return extract_unique_texts(turn.get("assumptions", []))


def extract_explicit_texts(turn: Dict) -> List[str]:
    return extract_unique_texts(turn.get("explicit_propositions", []))


def get_duration_sec(turn: Dict) -> float:
    duration = safe_float(turn.get("duration"))
    if math.isfinite(duration) and duration >= 0:
        return duration

    start_t = safe_float(turn.get("startTime", turn.get("start_time")))
    end_t = safe_float(turn.get("endTime", turn.get("end_time")))
    if math.isfinite(start_t) and math.isfinite(end_t):
        return max(0.0, end_t - start_t)
    return float("nan")


def get_gap_sec(curr_turn: Dict, next_turn: Dict) -> float:
    end_t = safe_float(curr_turn.get("endTime", curr_turn.get("end_time")))
    start_t = safe_float(next_turn.get("startTime", next_turn.get("start_time")))
    if math.isfinite(start_t) and math.isfinite(end_t):
        return start_t - end_t
    return float("nan")


def get_word_count(turn: Dict) -> float:
    wc = safe_float(turn.get("wordCount", turn.get("word_count")))
    if math.isfinite(wc):
        return wc
    text = str(turn.get("turn_text", "") or "").strip()
    if text:
        return float(len(text.split()))
    return float("nan")


def sort_turns(turns: Iterable[Dict]) -> List[Dict]:
    indexed: List[Tuple[int, int, Dict]] = []
    for i, t in enumerate(turns):
        idx = t.get("turn_idx")
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = i
        indexed.append((idx_int, i, t))
    indexed.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in indexed]


def load_episodes(input_dir: Path, show_progress: bool = True) -> Tuple[List[Tuple[str, List[Dict]]], np.ndarray]:
    episodes: List[Tuple[str, List[Dict]]] = []
    gaps: List[float] = []
    files = sorted(input_dir.glob("*.json"))
    file_iter = maybe_tqdm(files, enabled=show_progress, desc="Loading episodes", unit="file")
    for fp in file_iter:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list) or not payload:
            continue

        turns = sort_turns(payload)
        episode_id = str(turns[0].get("episode_id", fp.stem))
        episodes.append((episode_id, turns))

        for curr, nxt in zip(turns, turns[1:]):
            gap = get_gap_sec(curr, nxt)
            if math.isfinite(gap):
                gaps.append(gap)

    return episodes, np.array(gaps, dtype=float)


def classify_response_type(
    next_turn: Optional[Dict],
    gap_sec: float,
    silence_gap_threshold: float,
    backchannel_agree_duration_max: float,
    backchannel_agree_words_max: int,
) -> str:
    if next_turn is None:
        return "Silence/Abandonment"

    move = str(next_turn.get("conversation_move_label") or "").strip()
    ttype = str(next_turn.get("turn_type_label") or "").strip()

    if math.isfinite(gap_sec) and gap_sec >= silence_gap_threshold:
        return "Silence/Abandonment"
    if move == "Topic Shift":
        return "Silence/Abandonment"

    if move in {"Clarification Request (Generic)", "Clarification Request (Specific)"}:
        return "Clarification"

    if ttype == "Backchannel":
        return "Backchannel"

    if move == "Agree / Align":
        dur = get_duration_sec(next_turn)
        wc = get_word_count(next_turn)
        short_dur = (not math.isfinite(dur)) or (dur <= backchannel_agree_duration_max)
        short_wc = (not math.isfinite(wc)) or (wc <= backchannel_agree_words_max)
        if short_dur and short_wc:
            return "Backchannel"

    return "Substantive"


def build_turn_rows(
    episodes: List[Tuple[str, List[Dict]]],
    silence_gap_threshold: float,
    backchannel_agree_duration_max: float,
    backchannel_agree_words_max: int,
    show_progress: bool = True,
) -> List[Dict]:
    rows: List[Dict] = []

    episode_iter = maybe_tqdm(
        episodes,
        enabled=show_progress,
        desc="Building turn rows",
        unit="episode",
    )
    for episode_id, turns in episode_iter:
        history_assumptions = set()
        prior_gap_sum = 0.0
        prior_gap_count = 0

        for i, turn in enumerate(turns):
            next_turn = turns[i + 1] if i + 1 < len(turns) else None

            try:
                turn_idx = int(turn.get("turn_idx", i))
            except Exception:
                turn_idx = i

            duration = get_duration_sec(turn)
            assumptions = extract_assumption_texts(turn)
            explicit_props = extract_explicit_texts(turn)
            new_assumptions = [a for a in assumptions if a not in history_assumptions]
            new_count = len(new_assumptions)

            load = float("nan")
            if math.isfinite(duration) and duration >= 0 and new_count > 0:
                load = duration / new_count

            gap_sec = get_gap_sec(turn, next_turn) if next_turn is not None else float("nan")
            avg_prev_gap = (prior_gap_sum / prior_gap_count) if prior_gap_count > 0 else float("nan")
            response_type = classify_response_type(
                next_turn=next_turn,
                gap_sec=gap_sec,
                silence_gap_threshold=silence_gap_threshold,
                backchannel_agree_duration_max=backchannel_agree_duration_max,
                backchannel_agree_words_max=backchannel_agree_words_max,
            )

            row = {
                "episode_id": episode_id,
                "turn_idx": turn_idx,
                "duration_sec": duration,
                "assumption_count_in_turn": len(assumptions),
                "explicit_statement_count": len(explicit_props),
                "new_assumption_count": new_count,
                "implicature_load": load,
                "response_delay_at_time_n": gap_sec,
                "gap_to_next_sec": gap_sec,
                "average_response_time_0_to_n_minus_1": avg_prev_gap,
                "next_response_type": response_type,
                "next_turn_type_label": (next_turn or {}).get("turn_type_label"),
                "next_conversation_move_label": (next_turn or {}).get("conversation_move_label"),
            }
            rows.append(row)
            history_assumptions.update(assumptions)
            if math.isfinite(gap_sec) and gap_sec >= 0:
                prior_gap_sum += gap_sec
                prior_gap_count += 1

    rows.sort(key=lambda r: (str(r["episode_id"]), int(r["turn_idx"])))
    return rows


def standardize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not math.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)


def fit_multinomial_softmax(
    x: np.ndarray,
    y_labels: Sequence[str],
    max_iter: int = 5000,
    lr: float = 0.05,
    reg: float = 1e-4,
    show_progress: bool = False,
) -> Optional[Dict]:
    classes = sorted(set(y_labels))
    if len(classes) < 2:
        return None

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[v] for v in y_labels], dtype=int)

    x_std, mu, sd = standardize(x)
    X = np.column_stack([np.ones_like(x_std), x_std])

    n = X.shape[0]
    p = X.shape[1]
    k = len(classes)

    Y = np.zeros((n, k), dtype=float)
    Y[np.arange(n), y] = 1.0

    counts = np.bincount(y, minlength=k).astype(float)
    weights = np.array([n / (k * counts[yi]) for yi in y], dtype=float)

    W = np.zeros((p, k), dtype=float)

    iter_range = maybe_tqdm(
        range(max_iter),
        enabled=show_progress,
        desc="Fitting softmax",
        unit="iter",
        leave=False,
    )
    for _ in iter_range:
        logits = X @ W
        probs = softmax(logits)

        diff = (probs - Y) * weights[:, None]
        grad = (X.T @ diff) / n
        grad[1:, :] += reg * W[1:, :]

        W -= lr * grad

        if np.max(np.abs(grad)) < 1e-7:
            break

    return {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "W": W,
        "x_mean": mu,
        "x_std": sd,
    }


def predict_multinomial_proba(model: Dict, x: np.ndarray) -> np.ndarray:
    x_std = (x - model["x_mean"]) / model["x_std"]
    X = np.column_stack([np.ones_like(x_std), x_std])
    logits = X @ model["W"]
    return softmax(logits)


def rankdata_avg_ties(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank_val = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank_val
        i = j + 1
    return ranks


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = math.sqrt(float(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom <= 0:
        return float("nan"), float("nan")
    r = float(np.sum(x0 * y0) / denom)
    r = min(max(r, -0.999999), 0.999999)

    # Fisher z approximation for p-value
    n = len(x)
    if n <= 3:
        return r, float("nan")
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n - 3)
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return r, p


def spearman_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    rx = rankdata_avg_ties(x)
    ry = rankdata_avg_ties(y)
    return pearson_corr(rx, ry)


def correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    if len(x) < 3 or len(y) < 3:
        return {
            "n": int(min(len(x), len(y))),
            "pearson_r": None,
            "pearson_p_approx": None,
            "spearman_rho": None,
            "spearman_p_approx": None,
        }

    pear_r, pear_p = pearson_corr(x, y)
    spr_r, spr_p = spearman_corr(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": finite_or_none(pear_r),
        "pearson_p_approx": finite_or_none(pear_p),
        "spearman_rho": finite_or_none(spr_r),
        "spearman_p_approx": finite_or_none(spr_p),
    }


def distribution_stats(x: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "q90": None,
            "q99": None,
            "max": None,
            "p_zero": None,
            "skewness": None,
            "q99_over_median": None,
            "supports_log1p": False,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    min_val = float(np.min(arr))
    median = float(np.quantile(arr, 0.50))
    q90 = float(np.quantile(arr, 0.90))
    q99 = float(np.quantile(arr, 0.99))
    max_val = float(np.max(arr))
    p_zero = float(np.mean(arr == 0.0))
    if std > 1e-12 and len(arr) >= 3:
        skewness = float(np.mean(((arr - mean) / std) ** 3))
    else:
        skewness = float("nan")

    return {
        "n": int(len(arr)),
        "mean": mean,
        "std": std,
        "min": min_val,
        "median": median,
        "q90": q90,
        "q99": q99,
        "max": max_val,
        "p_zero": p_zero,
        "skewness": finite_or_none(skewness),
        "q99_over_median": finite_or_none(q99 / median) if median > 0 else None,
        "supports_log1p": bool(min_val >= 0.0),
    }


def transform_feature(
    values: np.ndarray,
    apply_log1p: bool,
    standardize_feature: bool,
) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
    out = np.asarray(values, dtype=float)
    if apply_log1p:
        out = np.log1p(out)

    transform_meta: Dict[str, Optional[float]] = {
        "used_log1p": bool(apply_log1p),
        "used_standardize": bool(standardize_feature),
        "standardize_mean": None,
        "standardize_std": None,
    }
    if standardize_feature:
        out, mu, sd = standardize(out)
        transform_meta["standardize_mean"] = finite_or_none(mu)
        transform_meta["standardize_std"] = finite_or_none(sd)
    return out, transform_meta


def transformed_term_name(name: str, use_log1p: bool, standardize_feature: bool) -> str:
    term = f"log1p({name})" if use_log1p else name
    if standardize_feature:
        term = f"z({term})"
    return term


def coefficient_row(name: str, estimate: float, std_error: float) -> Dict[str, Optional[float]]:
    z_score = estimate / std_error if std_error > 1e-12 else float("nan")
    p_value = 2.0 * (1.0 - normal_cdf(abs(z_score))) if math.isfinite(z_score) else float("nan")
    ci_low = estimate - (1.96 * std_error)
    ci_high = estimate + (1.96 * std_error)
    return {
        "term": name,
        "estimate": finite_or_none(estimate),
        "std_error_hc3": finite_or_none(std_error),
        "z_score": finite_or_none(z_score),
        "p_value_approx": finite_or_none(p_value),
        "ci95_low": finite_or_none(ci_low),
        "ci95_high": finite_or_none(ci_high),
    }


def fit_linear_regression(
    outcome: np.ndarray,
    predictors: Sequence[Tuple[str, np.ndarray]],
) -> Optional[Dict[str, object]]:
    y = np.asarray(outcome, dtype=float)
    if len(y) < 10:
        return None

    columns = [np.ones(len(y), dtype=float)]
    term_names = ["Intercept"]
    for name, values in predictors:
        arr = np.asarray(values, dtype=float)
        if arr.shape != y.shape:
            return None
        columns.append(arr)
        term_names.append(name)

    X = np.column_stack(columns)
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if int(np.sum(finite_mask)) < max(10, len(term_names) + 2):
        return None

    X = X[finite_mask]
    y = y[finite_mask]
    n, p = X.shape
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted

    dof = max(n - p, 1)
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (sse / sst) if sst > 1e-12 else float("nan")
    adj_r_squared = (
        1.0 - ((1.0 - r_squared) * (n - 1) / dof)
        if math.isfinite(r_squared) and dof > 0
        else float("nan")
    )
    rmse = math.sqrt(max(sse / dof, 0.0))

    leverage = np.sum(X * (X @ xtx_inv), axis=1)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-8)
    omega = (resid / (1.0 - leverage)) ** 2
    meat = X.T @ (X * omega[:, None])
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    std_errors = np.sqrt(np.clip(np.diag(cov_hc3), 0.0, None))

    resid_std = float(np.std(resid))
    resid_skew = (
        float(np.mean(((resid - np.mean(resid)) / resid_std) ** 3))
        if resid_std > 1e-12 and len(resid) >= 3
        else float("nan")
    )

    coef_rows = [
        coefficient_row(str(term_names[i]), float(beta[i]), float(std_errors[i]))
        for i in range(len(term_names))
    ]

    return {
        "n": int(n),
        "num_parameters": int(p),
        "r_squared": finite_or_none(r_squared),
        "adjusted_r_squared": finite_or_none(adj_r_squared),
        "rmse": finite_or_none(rmse),
        "residual_skewness": finite_or_none(resid_skew),
        "coefficients": coef_rows,
    }


def kde_1d(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    samples = samples[np.isfinite(samples)]
    n = len(samples)
    if n < 2:
        return np.zeros_like(grid)

    std = float(np.std(samples, ddof=1))
    if std <= 1e-12:
        std = 1e-3
    bw = 1.06 * std * (n ** (-1.0 / 5.0))
    bw = max(bw, 1e-3)

    z = (grid[:, None] - samples[None, :]) / bw
    dens = np.mean(np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi), axis=1) / bw
    return dens


def json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if math.isfinite(val):
            return val
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_plotly_html(out_path: Path, title: str, data: List[Dict], layout: Dict) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://cdn.plot.ly/plotly-3.1.0.min.js\"></script>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }}
    #plot {{ width: 100%; height: 78vh; min-height: 520px; }}
  </style>
</head>
<body>
  <div id=\"plot\"></div>
  <script>
    const data = {json.dumps(data, default=json_default)};
    const layout = {json.dumps(layout, default=json_default)};
    Plotly.newPlot('plot', data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def collect_load_by_response(rows: List[Dict], min_n: int = 5) -> Dict[str, np.ndarray]:
    load_by_cls: Dict[str, np.ndarray] = {}
    for cls in ["Backchannel", "Substantive", "Clarification", "Silence/Abandonment"]:
        vals = [
            float(r["implicature_load"])
            for r in rows
            if r.get("next_response_type") == cls
            and isinstance(r.get("implicature_load"), (int, float))
            and math.isfinite(float(r["implicature_load"]))
        ]
        if len(vals) >= min_n:
            load_by_cls[cls] = np.array(vals, dtype=float)
    return load_by_cls


def save_probability_curves_png(curve_rows: List[Dict], out_path: Path) -> bool:
    if not curve_rows:
        return False
    try:
        x_vals = np.array([float(r["implicature_load"]) for r in curve_rows], dtype=float)
        prob_cols = sorted([k for k in curve_rows[0].keys() if k.startswith("p_")])
        if not prob_cols:
            return False
        plt.figure(figsize=(10, 6))
        for col in prob_cols:
            label = col.replace("p_", "")
            y_vals = np.array([float(r[col]) for r in curve_rows], dtype=float)
            sns.lineplot(x=x_vals, y=y_vals, linewidth=2.0, label=label)
        plt.title("Logistic Regression Probability Curves by Load")
        plt.xlabel("Implicature Load L")
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def save_backchannel_vs_clarification_png(curve_rows: List[Dict], out_path: Path) -> bool:
    if not curve_rows:
        return False
    if "p_Backchannel" not in curve_rows[0] or "p_Clarification" not in curve_rows[0]:
        return False
    try:
        x_vals = np.array([float(r["implicature_load"]) for r in curve_rows], dtype=float)
        y_bc = np.array([float(r["p_Backchannel"]) for r in curve_rows], dtype=float)
        y_cl = np.array([float(r["p_Clarification"]) for r in curve_rows], dtype=float)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x_vals, y=y_bc, linewidth=2.5, label="Backchannel")
        sns.lineplot(x=x_vals, y=y_cl, linewidth=2.5, label="Clarification")
        plt.title("Backchannel vs Clarification Probability by Load")
        plt.xlabel("Implicature Load L")
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def save_response_time_correlation_png(
    response_latency: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    title: str,
    ylabel: str,
) -> bool:
    if len(response_latency) < 3 or len(y) < 3:
        return False
    try:
        if len(response_latency) > 30000:
            idx = np.random.choice(len(response_latency), size=30000, replace=False)
            x_plot = response_latency[idx]
            y_plot = y[idx]
        else:
            x_plot = response_latency
            y_plot = y

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_plot, y=y_plot, s=10, alpha=0.18, linewidth=0, color="#4C78A8")
        sns.regplot(
            x=response_latency,
            y=y,
            scatter=False,
            ci=None,
            line_kws={"color": "#E45756", "linewidth": 2.2},
        )
        plt.title(title)
        plt.xlabel("Response latency (seconds)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def save_ridge_png(load_by_cls: Dict[str, np.ndarray], out_path: Path) -> bool:
    if not load_by_cls:
        return False
    try:
        cls_order = [
            c
            for c in ["Backchannel", "Substantive", "Clarification", "Silence/Abandonment"]
            if c in load_by_cls
        ]
        if not cls_order:
            return False

        all_vals = np.concatenate([load_by_cls[c] for c in cls_order])
        xmin = float(np.quantile(all_vals, 0.01))
        xmax = float(np.quantile(all_vals, 0.99))
        if xmin == xmax:
            xmax = xmin + 1e-6

        fig, axes = plt.subplots(len(cls_order), 1, figsize=(11, 1.8 * len(cls_order)), sharex=True)
        if len(cls_order) == 1:
            axes = [axes]
        colors = sns.color_palette("Set2", len(cls_order))

        for i, cls in enumerate(cls_order):
            ax = axes[i]
            vals = load_by_cls[cls]
            clipped = np.clip(vals, xmin, xmax)
            sns.kdeplot(x=clipped, fill=True, alpha=0.7, linewidth=1.0, color=colors[i], ax=ax)
            sns.kdeplot(x=clipped, fill=False, linewidth=1.0, color="black", ax=ax)
            ax.set_ylabel(cls, rotation=0, ha="right", va="center", labelpad=35)
            ax.set_yticks([])
            ax.grid(axis="x", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            if i < len(cls_order) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Implicature Load L")

        fig.suptitle("Ridge Plot: Implicature Load Distribution by Response Type", y=1.02)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def build_probability_curves(
    model: Dict,
    loads: np.ndarray,
) -> List[Dict]:
    if len(loads) == 0:
        return []

    low = float(np.quantile(loads, 0.01))
    high = float(np.quantile(loads, 0.99))
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        low = float(np.min(loads))
        high = float(np.max(loads))
    if low == high:
        high = low + 1e-6

    grid = np.linspace(low, high, 260)
    proba = predict_multinomial_proba(model, grid)
    classes = model["classes"]

    rows: List[Dict] = []
    for i, x in enumerate(grid):
        row = {"implicature_load": float(x)}
        for j, cls in enumerate(classes):
            row[f"p_{cls}"] = float(proba[i, j])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5: Implicature Load vs Engagement")
    parser.add_argument("--input_dir", type=str, default="data/conversation_moves_labeled")
    parser.add_argument("--output_dir", type=str, default="experiments/exp5_processing_load/results")
    parser.add_argument("--silence_gap_quantile", type=float, default=0.95)
    parser.add_argument("--min_silence_gap", type=float, default=5.0)
    parser.add_argument("--backchannel_agree_duration_max", type=float, default=4.0)
    parser.add_argument("--backchannel_agree_words_max", type=int, default=12)
    parser.add_argument("--softmax_max_iter", type=int, default=5000)
    parser.add_argument("--softmax_lr", type=float, default=0.05)
    parser.add_argument("--softmax_reg", type=float, default=1e-4)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    show_progress = not args.no_tqdm

    episodes, gaps = load_episodes(input_dir, show_progress=show_progress)
    nonneg_gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]

    if len(nonneg_gaps) > 0:
        q = float(np.quantile(nonneg_gaps, args.silence_gap_quantile))
        silence_gap_threshold = max(float(args.min_silence_gap), q)
    else:
        silence_gap_threshold = float(args.min_silence_gap)

    rows = build_turn_rows(
        episodes=episodes,
        silence_gap_threshold=silence_gap_threshold,
        backchannel_agree_duration_max=args.backchannel_agree_duration_max,
        backchannel_agree_words_max=args.backchannel_agree_words_max,
        show_progress=show_progress,
    )

    # Prepare arrays for modeling
    model_rows = [
        r
        for r in rows
        if isinstance(r.get("implicature_load"), (int, float))
        and math.isfinite(float(r["implicature_load"]))
        and r.get("next_response_type") is not None
    ]

    model = None
    curve_rows: List[Dict] = []
    coef_rows: List[Dict] = []

    if model_rows:
        x = np.array([float(r["implicature_load"]) for r in model_rows], dtype=float)
        y = [str(r["next_response_type"]) for r in model_rows]

        # keep classes with at least 2 examples
        cnt = Counter(y)
        keep = {k for k, v in cnt.items() if v >= 2}
        x = np.array([x[i] for i, yi in enumerate(y) if yi in keep], dtype=float)
        y = [yi for yi in y if yi in keep]

        if len(set(y)) >= 2 and len(x) >= 10:
            model = fit_multinomial_softmax(
                x=x,
                y_labels=y,
                max_iter=args.softmax_max_iter,
                lr=args.softmax_lr,
                reg=args.softmax_reg,
                show_progress=show_progress,
            )

        if model is not None:
            curve_rows = build_probability_curves(model, x)
            classes = model["classes"]
            W = model["W"]
            for j, cls in enumerate(classes):
                coef_rows.append(
                    {
                        "class": cls,
                        "intercept": float(W[0, j]),
                        "coef_scaled_load": float(W[1, j]),
                        "scaler_mean": float(model["x_mean"]),
                        "scaler_std": float(model["x_std"]),
                    }
                )

    # Correlations with response latency (gap_to_next_sec)
    assumption_corr_pairs = [
        r
        for r in rows
        if isinstance(r.get("assumption_count_in_turn"), (int, float))
        and isinstance(r.get("gap_to_next_sec"), (int, float))
        and math.isfinite(float(r["assumption_count_in_turn"]))
        and math.isfinite(float(r["gap_to_next_sec"]))
        and float(r["gap_to_next_sec"]) >= 0
    ]
    assumption_x = np.array([float(r["assumption_count_in_turn"]) for r in assumption_corr_pairs], dtype=float)
    assumption_y = np.array([float(r["gap_to_next_sec"]) for r in assumption_corr_pairs], dtype=float)
    assumption_count_latency_corr = correlation_stats(
        assumption_x,
        assumption_y,
    )

    load_corr_pairs = [
        r
        for r in rows
        if isinstance(r.get("implicature_load"), (int, float))
        and isinstance(r.get("gap_to_next_sec"), (int, float))
        and math.isfinite(float(r["implicature_load"]))
        and math.isfinite(float(r["gap_to_next_sec"]))
        and float(r["gap_to_next_sec"]) >= 0
    ]
    load_x = np.array([float(r["implicature_load"]) for r in load_corr_pairs], dtype=float)
    load_y = np.array([float(r["gap_to_next_sec"]) for r in load_corr_pairs], dtype=float)
    implicature_load_latency_corr = correlation_stats(
        load_x,
        load_y,
    )

    # Response-delay regression:
    # response_delay_at_time_n ~ implicature_load + explicit_statement_count + average_response_time_0_to_n_minus_1
    regression_rows = [
        r
        for r in rows
        if isinstance(r.get("response_delay_at_time_n"), (int, float))
        and isinstance(r.get("implicature_load"), (int, float))
        and isinstance(r.get("explicit_statement_count"), (int, float))
        and isinstance(r.get("average_response_time_0_to_n_minus_1"), (int, float))
        and math.isfinite(float(r["response_delay_at_time_n"]))
        and math.isfinite(float(r["implicature_load"]))
        and math.isfinite(float(r["explicit_statement_count"]))
        and math.isfinite(float(r["average_response_time_0_to_n_minus_1"]))
        and float(r["response_delay_at_time_n"]) >= 0
        and float(r["average_response_time_0_to_n_minus_1"]) >= 0
    ]

    response_delay_regression: Optional[Dict[str, object]] = None
    response_delay_regression_coeffs: List[Dict[str, Optional[float]]] = []
    response_delay_distribution_checks: Dict[str, Dict[str, object]] = {}

    if len(regression_rows) >= 10:
        delay_raw = np.array([float(r["response_delay_at_time_n"]) for r in regression_rows], dtype=float)
        load_raw = np.array([float(r["implicature_load"]) for r in regression_rows], dtype=float)
        explicit_raw = np.array([float(r["explicit_statement_count"]) for r in regression_rows], dtype=float)
        avg_prev_raw = np.array([float(r["average_response_time_0_to_n_minus_1"]) for r in regression_rows], dtype=float)

        raw_stats = {
            "response_delay_at_time_n": distribution_stats(delay_raw),
            "implicature_load": distribution_stats(load_raw),
            "explicit_statement_count": distribution_stats(explicit_raw),
            "average_response_time_0_to_n_minus_1": distribution_stats(avg_prev_raw),
        }

        selected_transforms = {
            "response_delay_at_time_n": {"use_log1p": True, "standardize": False},
            "implicature_load": {"use_log1p": False, "standardize": False},
            "explicit_statement_count": {"use_log1p": False, "standardize": False},
            "average_response_time_0_to_n_minus_1": {"use_log1p": True, "standardize": False},
        }

        delay_modeled, delay_meta = transform_feature(
            delay_raw,
            apply_log1p=selected_transforms["response_delay_at_time_n"]["use_log1p"],
            standardize_feature=selected_transforms["response_delay_at_time_n"]["standardize"],
        )
        load_modeled, load_meta = transform_feature(
            load_raw,
            apply_log1p=selected_transforms["implicature_load"]["use_log1p"],
            standardize_feature=selected_transforms["implicature_load"]["standardize"],
        )
        explicit_modeled, explicit_meta = transform_feature(
            explicit_raw,
            apply_log1p=selected_transforms["explicit_statement_count"]["use_log1p"],
            standardize_feature=selected_transforms["explicit_statement_count"]["standardize"],
        )
        avg_prev_modeled, avg_prev_meta = transform_feature(
            avg_prev_raw,
            apply_log1p=selected_transforms["average_response_time_0_to_n_minus_1"]["use_log1p"],
            standardize_feature=selected_transforms["average_response_time_0_to_n_minus_1"]["standardize"],
        )

        response_delay_distribution_checks = {
            "response_delay_at_time_n": {
                "raw": raw_stats["response_delay_at_time_n"],
                "modeled": distribution_stats(delay_modeled),
                **delay_meta,
            },
            "implicature_load": {
                "raw": raw_stats["implicature_load"],
                "modeled": distribution_stats(load_modeled),
                **load_meta,
            },
            "explicit_statement_count": {
                "raw": raw_stats["explicit_statement_count"],
                "modeled": distribution_stats(explicit_modeled),
                **explicit_meta,
            },
            "average_response_time_0_to_n_minus_1": {
                "raw": raw_stats["average_response_time_0_to_n_minus_1"],
                "modeled": distribution_stats(avg_prev_modeled),
                **avg_prev_meta,
            },
        }

        fitted_outcome_name = transformed_term_name(
            "response_delay_at_time_n",
            use_log1p=bool(selected_transforms["response_delay_at_time_n"]["use_log1p"]),
            standardize_feature=bool(selected_transforms["response_delay_at_time_n"]["standardize"]),
        )
        fitted_predictor_names = [
            transformed_term_name(
                "implicature_load",
                use_log1p=bool(selected_transforms["implicature_load"]["use_log1p"]),
                standardize_feature=bool(selected_transforms["implicature_load"]["standardize"]),
            ),
            transformed_term_name(
                "explicit_statement_count",
                use_log1p=bool(selected_transforms["explicit_statement_count"]["use_log1p"]),
                standardize_feature=bool(selected_transforms["explicit_statement_count"]["standardize"]),
            ),
            transformed_term_name(
                "average_response_time_0_to_n_minus_1",
                use_log1p=bool(selected_transforms["average_response_time_0_to_n_minus_1"]["use_log1p"]),
                standardize_feature=bool(selected_transforms["average_response_time_0_to_n_minus_1"]["standardize"]),
            ),
        ]

        regression_fit = fit_linear_regression(
            outcome=delay_modeled,
            predictors=[
                (fitted_predictor_names[0], load_modeled),
                (fitted_predictor_names[1], explicit_modeled),
                (fitted_predictor_names[2], avg_prev_modeled),
            ],
        )

        if regression_fit is not None:
            response_delay_regression_coeffs = list(regression_fit["coefficients"])
            response_delay_regression = {
                "requested_formula": (
                    "response_delay_at_time_n ~ implicature_load + explicit_statement_count + "
                    "average_response_time_0_to_n_minus_1"
                ),
                "fitted_formula": (
                    f"{fitted_outcome_name} ~ "
                    f"{fitted_predictor_names[0]} + {fitted_predictor_names[1]} + {fitted_predictor_names[2]}"
                ),
                "n": int(regression_fit["n"]),
                "num_parameters": int(regression_fit["num_parameters"]),
                "r_squared": regression_fit["r_squared"],
                "adjusted_r_squared": regression_fit["adjusted_r_squared"],
                "rmse_on_modeled_scale": regression_fit["rmse"],
                "residual_skewness": regression_fit["residual_skewness"],
                "transform_selection": {
                    "source": "exp5_response_delay_transform_grid",
                    "selected_model_id": "log1p__raw__raw__log1p",
                    "selection_rule": "highest adjusted_r_squared in the transform grid",
                },
                "operationalization": {
                    "response_delay_at_time_n": "gap_to_next_sec",
                    "explicit_statement_count": "count of unique explicit_propositions in the current turn",
                    "average_response_time_0_to_n_minus_1": (
                        "within-episode mean of prior non-negative response delays before turn n"
                    ),
                },
                "coefficients": {
                    str(row["term"]): row["estimate"]
                    for row in response_delay_regression_coeffs
                    if row.get("term") is not None
                },
            }

    # Save CSV tables
    feature_fields = [
        "episode_id",
        "turn_idx",
        "duration_sec",
        "assumption_count_in_turn",
        "explicit_statement_count",
        "new_assumption_count",
        "implicature_load",
        "response_delay_at_time_n",
        "gap_to_next_sec",
        "average_response_time_0_to_n_minus_1",
        "next_response_type",
        "next_turn_type_label",
        "next_conversation_move_label",
    ]
    write_csv(output_dir / "exp5_turn_level_features.csv", rows, feature_fields)

    if curve_rows:
        prob_fields = ["implicature_load"] + [k for k in curve_rows[0].keys() if k != "implicature_load"]
        write_csv(output_dir / "exp5_probability_curves.csv", curve_rows, prob_fields)

    if coef_rows:
        coef_fields = ["class", "intercept", "coef_scaled_load", "scaler_mean", "scaler_std"]
        write_csv(output_dir / "exp5_logit_coefficients.csv", coef_rows, coef_fields)

    if response_delay_regression_coeffs:
        regression_coef_fields = [
            "term",
            "estimate",
            "std_error_hc3",
            "z_score",
            "p_value_approx",
            "ci95_low",
            "ci95_high",
        ]
        write_csv(
            output_dir / "exp5_response_delay_regression_coefficients.csv",
            response_delay_regression_coeffs,
            regression_coef_fields,
        )

    if response_delay_regression is not None:
        (output_dir / "exp5_response_delay_regression_summary.json").write_text(
            json.dumps(
                {
                    "regression": response_delay_regression,
                    "distribution_checks": response_delay_distribution_checks,
                },
                ensure_ascii=False,
                indent=2,
                default=json_default,
            ),
            encoding="utf-8",
        )

    response_counts = Counter([str(r.get("next_response_type")) for r in rows if r.get("next_response_type")])
    count_rows = [{"response_type": k, "count": v} for k, v in sorted(response_counts.items())]
    write_csv(output_dir / "exp5_response_type_counts.csv", count_rows, ["response_type", "count"])

    png_outputs: List[str] = []
    if curve_rows and save_probability_curves_png(
        curve_rows,
        output_dir / "exp5_probability_curves.png",
    ):
        png_outputs.append("exp5_probability_curves.png")

    if curve_rows and save_backchannel_vs_clarification_png(
        curve_rows,
        output_dir / "exp5_backchannel_vs_clarification_curves.png",
    ):
        png_outputs.append("exp5_backchannel_vs_clarification_curves.png")

    if save_response_time_correlation_png(
        response_latency=assumption_y,
        y=assumption_x,
        out_path=output_dir / "exp5_assumption_count_vs_response_time.png",
        title="Assumption Count by Response Time",
        ylabel="Assumption count in turn",
    ):
        png_outputs.append("exp5_assumption_count_vs_response_time.png")

    if save_response_time_correlation_png(
        response_latency=load_y,
        y=load_x,
        out_path=output_dir / "exp5_implicature_load_vs_response_time.png",
        title="Implicature Load by Response Time",
        ylabel="Implicature Load L",
    ):
        png_outputs.append("exp5_implicature_load_vs_response_time.png")

    load_by_cls = collect_load_by_response(rows, min_n=5)
    if save_ridge_png(
        load_by_cls,
        output_dir / "exp5_load_ridge_by_response_type.png",
    ):
        png_outputs.append("exp5_load_ridge_by_response_type.png")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "num_turn_rows": len(rows),
        "silence_gap_threshold_sec": silence_gap_threshold,
        "silence_gap_quantile": args.silence_gap_quantile,
        "min_silence_gap_sec": args.min_silence_gap,
        "assumption_count_vs_response_time_correlation": assumption_count_latency_corr,
        "implicature_load_vs_response_time_correlation": implicature_load_latency_corr,
        "response_delay_regression": response_delay_regression,
        "response_delay_regression_distribution_checks": response_delay_distribution_checks,
        "response_type_counts": {k: int(v) for k, v in response_counts.items()},
        "tqdm_enabled": show_progress,
        "png_outputs": png_outputs,
    }

    (output_dir / "exp5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
