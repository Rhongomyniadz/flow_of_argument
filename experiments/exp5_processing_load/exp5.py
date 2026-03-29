import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)
RESPONSE_CLASSES = ["Backchannel", "Substantive", "Clarification", "Silence/Abandonment"]
_ASSUMPTION_EMBEDDER = None


def assumption_embedder():
    global _ASSUMPTION_EMBEDDER
    if _ASSUMPTION_EMBEDDER is None:
        _ASSUMPTION_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ASSUMPTION_EMBEDDER


def assumption_embeddings(texts):
    texts = [t for t in texts if t]
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    return assumption_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def jdefault(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    raise TypeError(type(x).__name__)


def norm_text(s):
    s = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(s or "").strip().lower()))
    return s.strip()


def uniq_texts(raw):
    seen, out = set(), []
    for item in raw if isinstance(raw, list) else []:
        text = item.get("text", "") if isinstance(item, dict) else item if isinstance(item, str) else ""
        text = norm_text(text)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def sec(turn):
    dur = f(turn.get("duration"))
    if math.isfinite(dur) and dur >= 0:
        return dur
    start, end = f(turn.get("startTime", turn.get("start_time"))), f(turn.get("endTime", turn.get("end_time")))
    return max(0.0, end - start) if math.isfinite(start) and math.isfinite(end) else float("nan")


def gap(a, b):
    end, start = f(a.get("endTime", a.get("end_time"))), f(b.get("startTime", b.get("start_time")))
    return start - end if math.isfinite(start) and math.isfinite(end) else float("nan")


def words(turn):
    wc = f(turn.get("wordCount", turn.get("word_count")))
    return wc if math.isfinite(wc) else float(len(str(turn.get("turn_text", "") or "").split()) or "nan")


def sort_turns(turns):
    return [x[2] for x in sorted(((int(t.get("turn_idx", i)) if str(t.get("turn_idx", i)).lstrip("-").isdigit() else i, i, t) for i, t in enumerate(turns)))]


def load_episodes(input_dir, show_progress):
    files = sorted(input_dir.glob("*.json")) or sorted(input_dir.glob("*/*.json"))
    episodes, gaps = [], []
    for fp in tqdm(files, desc="Loading episodes", unit="file", disable=not show_progress):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list) or not payload:
            continue
        turns = sort_turns(payload)
        episodes.append((str(turns[0].get("episode_id", fp.stem)), turns))
        gaps.extend(g for a, b in zip(turns, turns[1:]) if math.isfinite(g := gap(a, b)))
    return episodes, np.asarray(gaps, dtype=float)


def classify(next_turn, gap_sec, silence_gap, agree_dur_max, agree_words_max):
    if next_turn is None:
        return "Silence/Abandonment"
    move = str(next_turn.get("conversation_move_label") or "").strip()
    ttype = str(next_turn.get("turn_type_label") or "").strip()
    if (math.isfinite(gap_sec) and gap_sec >= silence_gap) or move == "Topic Shift":
        return "Silence/Abandonment"
    if move in {"Clarification Request (Generic)", "Clarification Request (Specific)"}:
        return "Clarification"
    if ttype == "Backchannel":
        return "Backchannel"
    if move == "Agree / Align":
        if ((not math.isfinite(sec(next_turn))) or sec(next_turn) <= agree_dur_max) and (
            (not math.isfinite(words(next_turn))) or words(next_turn) <= agree_words_max
        ):
            return "Backchannel"
    return "Substantive"


def build_rows(episodes, silence_gap, agree_dur_max, agree_words_max, show_progress, assumption_similarity_threshold):
    rows = []
    for episode_id, turns in tqdm(episodes, desc="Building turn rows", unit="episode", disable=not show_progress):
        history_embs, gap_sum, gap_n = None, 0.0, 0
        for i, turn in enumerate(turns):
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            assumptions = uniq_texts(turn.get("assumptions", []))
            explicit = uniq_texts(turn.get("explicit_propositions", []))
            new_n, current_new = 0, []
            if assumptions:
                current_embs = assumption_embeddings(assumptions)
                if history_embs is None or len(history_embs) == 0:
                    new_n, current_new = len(assumptions), list(assumptions)
                else:
                    sims = history_embs @ current_embs.T
                    for idx, text in enumerate(assumptions):
                        max_sim = float(np.max(sims[:, idx])) if sims.size else 0.0
                        if max_sim < assumption_similarity_threshold:
                            new_n += 1
                            current_new.append(text)
            duration = sec(turn)
            g = gap(turn, nxt) if nxt else float("nan")
            rows.append(
                {
                    "episode_id": episode_id,
                    "turn_idx": int(turn.get("turn_idx", i)) if str(turn.get("turn_idx", i)).lstrip("-").isdigit() else i,
                    "duration_sec": duration,
                    "assumption_count_in_turn": len(assumptions),
                    "explicit_statement_count": len(explicit),
                    "new_assumption_count": new_n,
                    "implicature_load": duration / new_n if math.isfinite(duration) and duration >= 0 and new_n > 0 else float("nan"),
                    "response_delay_at_time_n": g,
                    "gap_to_next_sec": g,
                    "average_response_time_0_to_n_minus_1": (gap_sum / gap_n) if gap_n else float("nan"),
                    "next_response_type": classify(nxt, g, silence_gap, agree_dur_max, agree_words_max),
                    "next_turn_type_label": (nxt or {}).get("turn_type_label"),
                    "next_conversation_move_label": (nxt or {}).get("conversation_move_label"),
                }
            )
            if current_new:
                new_embs = assumption_embeddings(current_new)
                history_embs = new_embs if history_embs is None else np.vstack([history_embs, new_embs])
            if math.isfinite(g) and g >= 0:
                gap_sum += g
                gap_n += 1
    return sorted(rows, key=lambda r: (str(r["episode_id"]), int(r["turn_idx"])))


def standardize(x):
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / (sd if math.isfinite(sd) and sd > 1e-12 else 1.0), mu, (sd if math.isfinite(sd) and sd > 1e-12 else 1.0)


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def unpack(theta, p, k):
    w = np.zeros((p, k), dtype=float)
    if k > 1:
        w[:, :-1] = theta.reshape(p, k - 1)
    return w


def fit_multinomial_bayes(x, y_labels, prior_sd=2.5, posterior_draws=1200):
    classes = sorted(set(y_labels))
    if len(classes) < 2:
        return None
    y = np.asarray([classes.index(v) for v in y_labels], dtype=int)
    xz, mu, sd = standardize(np.asarray(x, dtype=float))
    X, p, k = np.column_stack([np.ones_like(xz), xz]), 2, len(classes)
    prior_var = max(prior_sd**2, 1e-12)

    def obj(theta):
        w = unpack(theta, p, k)
        probs = softmax(X @ w)
        Y = np.zeros_like(probs)
        Y[np.arange(len(y)), y] = 1.0
        logp = float(np.sum(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)))) - 0.5 * float(np.sum(theta * theta) / prior_var)
        grad = (X.T @ (Y - probs))[:, :-1].reshape(-1) - theta / prior_var
        return -logp, -grad

    opt = minimize(fun=lambda t: obj(t)[0], x0=np.zeros(p * (k - 1), dtype=float), jac=lambda t: obj(t)[1], method="L-BFGS-B")
    if not opt.success:
        return None
    theta_map = np.asarray(opt.x, dtype=float)
    probs = softmax(X @ unpack(theta_map, p, k))
    info = np.eye(p * (k - 1), dtype=float) / prior_var
    for i in range(len(y)):
        info += np.kron(np.diag(probs[i, :-1]) - np.outer(probs[i, :-1], probs[i, :-1]), np.outer(X[i], X[i]))
    cov = np.linalg.pinv(info)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.multivariate_normal(theta_map, cov, size=max(int(posterior_draws), 200))
    w_draws = np.stack([unpack(d, p, k) for d in draws], axis=0)
    return {
        "classes": classes,
        "reference_class": classes[-1],
        "x_mean": mu,
        "x_std": sd,
        "W_map": unpack(theta_map, p, k),
        "posterior_W": w_draws,
        "prior_sd": float(prior_sd),
        "posterior_draws": int(len(w_draws)),
        "approximation": "laplace_map_gaussian_posterior",
    }


def posterior_multinomial_proba(model, x):
    x = (np.asarray(x, dtype=float) - model["x_mean"]) / model["x_std"]
    X = np.column_stack([np.ones_like(x), x])
    return softmax(np.einsum("np,dpk->dnk", X, model["posterior_W"]))


def corr_stats(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return {"n": int(min(len(x), len(y))), "pearson_r": None, "pearson_p_approx": None, "spearman_rho": None, "spearman_p_approx": None}

    def pear(a, b):
        a, b = a - np.mean(a), b - np.mean(b)
        denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
        if denom <= 0:
            return float("nan"), float("nan")
        r = min(max(float(np.sum(a * b) / denom), -0.999999), 0.999999)
        z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(len(a) - 3, 0))
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))) if len(a) > 3 else float("nan")
        return r, p

    def rank(a):
        order, out, i, n = np.argsort(a), np.empty(len(a), dtype=float), 0, len(a)
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            out[order[i : j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return out

    pr, pp = pear(x, y)
    sr, sp = pear(rank(x), rank(y))
    return {"n": int(len(x)), "pearson_r": pr if math.isfinite(pr) else None, "pearson_p_approx": pp if math.isfinite(pp) else None, "spearman_rho": sr if math.isfinite(sr) else None, "spearman_p_approx": sp if math.isfinite(sp) else None}


def dist_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "q90": None, "q99": None, "max": None, "p_zero": None, "skewness": None, "q99_over_median": None, "supports_log1p": False}
    mean, std, median, q90, q99 = float(np.mean(x)), float(np.std(x)), float(np.quantile(x, 0.5)), float(np.quantile(x, 0.9)), float(np.quantile(x, 0.99))
    skew = float(np.mean(((x - mean) / std) ** 3)) if std > 1e-12 and len(x) >= 3 else float("nan")
    return {
        "n": int(len(x)),
        "mean": mean,
        "std": std,
        "min": float(np.min(x)),
        "median": median,
        "q90": q90,
        "q99": q99,
        "max": float(np.max(x)),
        "p_zero": float(np.mean(x == 0.0)),
        "skewness": skew if math.isfinite(skew) else None,
        "q99_over_median": (q99 / median) if median > 0 else None,
        "supports_log1p": bool(float(np.min(x)) >= 0),
    }


def transform(x, name):
    x = np.asarray(x, dtype=float)
    return {"raw": x.copy(), "log1p": np.log1p(x), "sqrt": np.sqrt(x), "asinh": np.arcsinh(x)}[name]


def term(name, transform_name):
    return name if transform_name == "raw" else f"{transform_name}({name})"


def coef_summary(name, draws):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {"term": name, "posterior_mean": None, "posterior_sd": None}
    return {"term": name, "posterior_mean": float(np.mean(draws)), "posterior_sd": float(np.std(draws, ddof=1)) if len(draws) >= 2 else 0.0}


def fit_linear_bayes(y, predictors, prior_precision_scale=0.1, prior_a0=2.0, prior_b0=1.0, posterior_draws=4000):
    y = np.asarray(y, dtype=float)
    cols, names = [np.ones(len(y), dtype=float)], ["Intercept"]
    for name, values in predictors:
        values = np.asarray(values, dtype=float)
        if values.shape != y.shape:
            return None
        cols.append(values)
        names.append(name)
    X = np.column_stack(cols)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(m)) < max(10, len(names) + 2):
        return None
    X, y = X[m], y[m]
    n, p = X.shape
    diag = [1e-6] + [float(prior_precision_scale / max(s * s, 1e-6)) for s in np.std(X[:, 1:], axis=0)] if p > 1 else [1e-6]
    V0i, b0 = np.diag(diag), np.zeros(p, dtype=float)
    Vni = V0i + X.T @ X
    Vn = np.linalg.pinv(Vni)
    bn = Vn @ (V0i @ b0 + X.T @ y)
    an = prior_a0 + n / 2.0
    bnn = max(prior_b0 + 0.5 * (float(y.T @ y) + float(b0.T @ V0i @ b0) - float(bn.T @ Vni @ bn)), 1e-9)
    fitted, resid = X @ bn, y - X @ bn
    sse, sst = float(np.sum(resid**2)), float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-12 else float("nan")
    adj = 1.0 - ((1.0 - r2) * (n - 1) / max(n - p, 1)) if math.isfinite(r2) else float("nan")
    resid_std = float(np.std(resid))
    resid_skew = float(np.mean(((resid - np.mean(resid)) / resid_std) ** 3)) if resid_std > 1e-12 and len(resid) >= 3 else float("nan")
    rng = np.random.default_rng(RNG_SEED)
    sig2 = 1.0 / np.clip(rng.gamma(shape=max(an, 1e-9), scale=1.0 / max(bnn, 1e-9), size=max(int(posterior_draws), 500)), 1e-12, None)
    beta = bn[None, :] + np.sqrt(sig2)[:, None] * (rng.normal(size=(len(sig2), p)) @ np.linalg.cholesky(Vn + 1e-10 * np.eye(p)).T)
    sigma = np.sqrt(np.clip(sig2, 0.0, None))
    return {
        "n": int(n),
        "num_parameters": int(p),
        "r_squared": r2 if math.isfinite(r2) else None,
        "adjusted_r_squared": adj if math.isfinite(adj) else None,
        "rmse": math.sqrt(max(sse / max(n - p, 1), 0.0)),
        "residual_skewness": resid_skew if math.isfinite(resid_skew) else None,
        "coefficients": [coef_summary(names[i], beta[:, i]) for i in range(len(names))],
        "prior": {"beta_prior_center": [0.0] * p, "beta_prior_precision_diagonal": diag, "sigma2_prior_a0": float(prior_a0), "sigma2_prior_b0": float(prior_b0)},
        "sigma_summary": coef_summary("sigma", sigma),
        "posterior_df": float(2.0 * an),
    }


def robust_xlim(x, lo=0.01, hi=0.99, cap=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    xmin, xmax = float(np.quantile(x, lo)), float(np.quantile(x, hi))
    if cap is not None:
        xmax = min(xmax, float(cap))
    xmin = min(xmin, float(np.min(x)))
    return (xmin, xmax if xmax > xmin else xmin + 1e-6)


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(row.get(k), float) and not math.isfinite(row.get(k)) else row.get(k)) for k in fields})


def save_plot_curve(rows, out_path):
    if not rows:
        return False
    try:
        x = np.asarray([r["implicature_load"] for r in rows], dtype=float)
        xmin, xmax = robust_xlim(x)
        plt.figure(figsize=(10, 6))
        cols = sorted(k for k in rows[0] if k.startswith("p_"))
        for col in cols:
            sns.lineplot(x=x, y=np.asarray([r[col] for r in rows], dtype=float), linewidth=2.0, label=col.replace("p_", ""))
        plt.title("Bayesian Response Probability Curves by Load")
        plt.xlabel("Implicature Load L")
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.xlim(xmin, xmax)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def save_plot_scatter(x, y, out_path, title, ylabel):
    if len(x) < 3 or len(y) < 3:
        return False
    try:
        xmin, xmax = robust_xlim(x, lo=0.0, hi=0.995, cap=2000.0)
        m = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
        x, y = np.asarray(x)[m], np.asarray(y)[m]
        if len(x) < 3:
            return False
        if len(x) > 30000:
            idx = np.random.choice(len(x), size=30000, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_plot, y=y_plot, s=10, alpha=0.18, linewidth=0, color="#4C78A8")
        sns.regplot(x=x, y=y, scatter=False, ci=None, line_kws={"color": "#E45756", "linewidth": 2.2})
        plt.title(title)
        plt.xlabel("Response latency (seconds)")
        plt.ylabel(ylabel)
        plt.xlim(xmin, xmax)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def save_ridge(rows, out_path):
    by_cls = {cls: np.asarray([float(r["implicature_load"]) for r in rows if r.get("next_response_type") == cls and isinstance(r.get("implicature_load"), (int, float)) and math.isfinite(float(r["implicature_load"]))], dtype=float) for cls in RESPONSE_CLASSES}
    by_cls = {k: v for k, v in by_cls.items() if len(v) >= 5}
    if not by_cls:
        return False
    try:
        all_vals = np.concatenate(list(by_cls.values()))
        xmin, xmax = robust_xlim(all_vals, cap=40.0)
        order = [c for c in RESPONSE_CLASSES if c in by_cls]
        fig, axes = plt.subplots(len(order), 1, figsize=(11, 1.8 * len(order)), sharex=True)
        axes = [axes] if len(order) == 1 else axes
        colors = sns.color_palette("Set2", len(order))
        for i, cls in enumerate(order):
            ax, vals = axes[i], np.clip(by_cls[cls], xmin, xmax)
            sns.kdeplot(x=vals, fill=True, alpha=0.7, linewidth=1.0, color=colors[i], ax=ax)
            sns.kdeplot(x=vals, fill=False, linewidth=1.0, color="black", ax=ax)
            ax.set_ylabel(cls, rotation=0, ha="right", va="center", labelpad=35)
            ax.set_yticks([])
            ax.grid(axis="x", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel("" if i < len(order) - 1 else "Implicature Load L")
        fig.suptitle("Ridge Plot: Implicature Load Distribution by Response Type", y=1.02)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def build_probability_curves(model, loads):
    if len(loads) == 0:
        return []
    lo, hi = robust_xlim(loads)
    probs = np.mean(posterior_multinomial_proba(model, np.linspace(lo, hi, 260)), axis=0)
    grid = np.linspace(lo, hi, 260)
    return [{"implicature_load": float(grid[i]), **{f"p_{cls}": float(probs[i, j]) for j, cls in enumerate(model["classes"])}} for i in range(len(grid))]


def select_delay_transform(delay, load, avg_prev, prior_precision_scale, prior_a0, prior_b0, posterior_draws):
    predictors = {
        "implicature_load": ("raw", transform(load, "raw")),
        "average_response_time_0_to_n_minus_1": ("log1p", transform(avg_prev, "log1p")),
    }
    predictor_names = [term(k, t) for k, (t, _) in predictors.items()]
    candidates = []
    for out_t in ["raw", "log1p", "sqrt", "asinh"]:
        y = transform(delay, out_t)
        fit = fit_linear_bayes(
            y,
            list(zip(predictor_names, [predictors[k][1] for k in predictors])),
            prior_precision_scale,
            prior_a0,
            prior_b0,
            posterior_draws,
        )
        if fit is None:
            continue
        candidates.append(
            {
                "outcome_transform": out_t,
                "outcome_name": term("response_delay_at_time_n", out_t),
                "delay_modeled": y,
                "load_modeled": predictors["implicature_load"][1],
                "avg_prev_modeled": predictors["average_response_time_0_to_n_minus_1"][1],
                "predictor_names": predictor_names,
                "fit": fit,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-float(c["fit"]["adjusted_r_squared"]) if c["fit"]["adjusted_r_squared"] is not None else float("inf"), abs(float(c["fit"]["residual_skewness"])) if c["fit"]["residual_skewness"] is not None else float("inf")))
    best = candidates[0]
    return {
        "best": best,
        "candidates": [{"outcome_transform": c["outcome_transform"], "outcome_name": c["outcome_name"], "adjusted_r_squared": c["fit"]["adjusted_r_squared"], "r_squared": c["fit"]["r_squared"], "rmse_on_modeled_scale": c["fit"]["rmse"], "residual_skewness": c["fit"]["residual_skewness"]} for c in candidates],
    }


def main():
    ap = argparse.ArgumentParser(description="Experiment 5: Implicature Load vs Engagement")
    ap.add_argument("--input_dir", default="data/conversation_moves_labeled")
    ap.add_argument("--output_dir", default="experiments/exp5_processing_load/results")
    ap.add_argument("--silence_gap_quantile", type=float, default=0.95)
    ap.add_argument("--min_silence_gap", type=float, default=5.0)
    ap.add_argument("--backchannel_agree_duration_max", type=float, default=4.0)
    ap.add_argument("--backchannel_agree_words_max", type=int, default=12)
    ap.add_argument("--assumption_similarity_threshold", type=float, default=0.80)
    ap.add_argument("--bayes_multinomial_prior_sd", type=float, default=2.5)
    ap.add_argument("--bayes_multinomial_draws", type=int, default=1200)
    ap.add_argument("--bayes_linear_draws", type=int, default=4000)
    ap.add_argument("--bayes_linear_prior_precision_scale", type=float, default=0.1)
    ap.add_argument("--bayes_linear_prior_a0", type=float, default=2.0)
    ap.add_argument("--bayes_linear_prior_b0", type=float, default=1.0)
    ap.add_argument("--no_tqdm", action="store_true")
    args = ap.parse_args()

    input_dir, output_dir, show_progress = Path(args.input_dir), Path(args.output_dir), not args.no_tqdm
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, gaps = load_episodes(input_dir, show_progress)
    nonneg = gaps[np.isfinite(gaps) & (gaps >= 0)]
    silence_gap = max(float(args.min_silence_gap), float(np.quantile(nonneg, args.silence_gap_quantile))) if len(nonneg) else float(args.min_silence_gap)
    rows = build_rows(
        episodes,
        silence_gap,
        args.backchannel_agree_duration_max,
        args.backchannel_agree_words_max,
        show_progress,
        args.assumption_similarity_threshold,
    )

    model_rows = [r for r in rows if isinstance(r.get("implicature_load"), (int, float)) and math.isfinite(float(r["implicature_load"])) and r.get("next_response_type")]
    model = curve_rows = None
    coef_rows = []
    if model_rows:
        x = np.asarray([float(r["implicature_load"]) for r in model_rows], dtype=float)
        y = [str(r["next_response_type"]) for r in model_rows]
        keep = {k for k, v in Counter(y).items() if v >= 2}
        x = np.asarray([x[i] for i, yi in enumerate(y) if yi in keep], dtype=float)
        y = [yi for yi in y if yi in keep]
        if len(set(y)) >= 2 and len(x) >= 10:
            model = fit_multinomial_bayes(x, y, args.bayes_multinomial_prior_sd, args.bayes_multinomial_draws)
        if model:
            curve_rows = build_probability_curves(model, x)
            for j, cls in enumerate(model["classes"]):
                w = model["posterior_W"][:, :, j]
                coef_rows.append(
                    {
                        "class": cls,
                        "reference_class": model["reference_class"],
                        "posterior_mean_intercept": float(np.mean(w[:, 0])),
                        "posterior_mean_coef_scaled_load": float(np.mean(w[:, 1])),
                        "posterior_sd_intercept": float(np.std(w[:, 0], ddof=1)),
                        "posterior_sd_coef_scaled_load": float(np.std(w[:, 1], ddof=1)),
                        "map_intercept": float(model["W_map"][0, j]),
                        "map_coef_scaled_load": float(model["W_map"][1, j]),
                        "scaler_mean": float(model["x_mean"]),
                        "scaler_std": float(model["x_std"]),
                    }
                )

    pairs = lambda a, b: [r for r in rows if all(isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k])) for k in [a, b]) and float(r[b]) >= 0]
    apairs, lpairs = pairs("assumption_count_in_turn", "gap_to_next_sec"), pairs("implicature_load", "gap_to_next_sec")
    assumption_x, assumption_y = np.asarray([float(r["assumption_count_in_turn"]) for r in apairs]), np.asarray([float(r["gap_to_next_sec"]) for r in apairs])
    load_x, load_y = np.asarray([float(r["implicature_load"]) for r in lpairs]), np.asarray([float(r["gap_to_next_sec"]) for r in lpairs])

    reg_rows = [
        r
        for r in rows
        if all(isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k])) for k in ["response_delay_at_time_n", "implicature_load", "average_response_time_0_to_n_minus_1"])
        and float(r["response_delay_at_time_n"]) >= 0
        and float(r["average_response_time_0_to_n_minus_1"]) >= 0
    ]

    response_delay_regression = None
    response_delay_regression_coeffs = []
    response_delay_distribution_checks = {}
    if len(reg_rows) >= 10:
        delay_raw = np.asarray([float(r["response_delay_at_time_n"]) for r in reg_rows], dtype=float)
        load_raw = np.asarray([float(r["implicature_load"]) for r in reg_rows], dtype=float)
        avg_prev_raw = np.asarray([float(r["average_response_time_0_to_n_minus_1"]) for r in reg_rows], dtype=float)
        search = select_delay_transform(
            delay_raw,
            load_raw,
            avg_prev_raw,
            args.bayes_linear_prior_precision_scale,
            args.bayes_linear_prior_a0,
            args.bayes_linear_prior_b0,
            args.bayes_linear_draws,
        )
        if search:
            best, fit = search["best"], search["best"]["fit"]
            response_delay_regression_coeffs = fit["coefficients"]
            response_delay_distribution_checks = {
                "response_delay_at_time_n": {"transform": best["outcome_transform"], "raw": dist_stats(delay_raw), "modeled": dist_stats(best["delay_modeled"])},
                "implicature_load": {"transform": "raw", "raw": dist_stats(load_raw), "modeled": dist_stats(best["load_modeled"])},
                "average_response_time_0_to_n_minus_1": {"transform": "log1p", "raw": dist_stats(avg_prev_raw), "modeled": dist_stats(best["avg_prev_modeled"])},
            }
            response_delay_regression = {
                "specification": "primary_spec",
                "specification_description": (
                    "Primary Experiment 5 response-delay model aligned with the stated setup: "
                    "implicature_load plus prior response-time baseline, without explicit_statement_count."
                ),
                "requested_formula": "response_delay_at_time_n ~ implicature_load + average_response_time_0_to_n_minus_1",
                "fitted_formula": f"{best['outcome_name']} ~ {' + '.join(best['predictor_names'])}",
                "n": fit["n"],
                "num_parameters": fit["num_parameters"],
                "r_squared": fit["r_squared"],
                "adjusted_r_squared": fit["adjusted_r_squared"],
                "rmse_on_modeled_scale": fit["rmse"],
                "residual_skewness": fit["residual_skewness"],
                "model_family": "bayesian_linear_regression_normal_inverse_gamma",
                "posterior_df": fit["posterior_df"],
                "sigma": fit["sigma_summary"],
                "prior": fit["prior"],
                "transform_selection": {"selected_model_id": f"{best['outcome_transform']}__raw__log1p", "selection_rule": "highest adjusted_r_squared, tie-broken by lowest absolute residual skewness", "candidates": search["candidates"]},
                "operationalization": {
                    "response_delay_at_time_n": "gap_to_next_sec",
                    "average_response_time_0_to_n_minus_1": "within-episode mean of prior non-negative response delays before turn n",
                },
                "coefficients": {str(r["term"]): r["posterior_mean"] for r in response_delay_regression_coeffs if r.get("term") is not None},
            }

    save_csv(output_dir / "exp5_turn_level_features.csv", rows, ["episode_id", "turn_idx", "duration_sec", "assumption_count_in_turn", "explicit_statement_count", "new_assumption_count", "implicature_load", "response_delay_at_time_n", "gap_to_next_sec", "average_response_time_0_to_n_minus_1", "next_response_type", "next_turn_type_label", "next_conversation_move_label"])
    if curve_rows:
        save_csv(output_dir / "exp5_probability_curves.csv", curve_rows, ["implicature_load"] + [k for k in curve_rows[0] if k != "implicature_load"])
    if coef_rows:
        save_csv(output_dir / "exp5_logit_coefficients.csv", coef_rows, ["class", "reference_class", "posterior_mean_intercept", "posterior_mean_coef_scaled_load", "posterior_sd_intercept", "posterior_sd_coef_scaled_load", "map_intercept", "map_coef_scaled_load", "scaler_mean", "scaler_std"])
    if response_delay_regression_coeffs:
        save_csv(output_dir / "exp5_response_delay_regression_coefficients.csv", response_delay_regression_coeffs, ["term", "posterior_mean", "posterior_sd"])
    if response_delay_regression is not None:
        (output_dir / "exp5_response_delay_regression_summary.json").write_text(json.dumps({"regression": response_delay_regression, "distribution_checks": response_delay_distribution_checks}, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")

    response_counts = Counter(str(r.get("next_response_type")) for r in rows if r.get("next_response_type"))
    save_csv(output_dir / "exp5_response_type_counts.csv", [{"response_type": k, "count": v} for k, v in sorted(response_counts.items())], ["response_type", "count"])

    png_outputs = []
    if curve_rows and save_plot_curve(curve_rows, output_dir / "exp5_probability_curves.png"):
        png_outputs.append("exp5_probability_curves.png")
    if save_plot_scatter(assumption_y, assumption_x, output_dir / "exp5_assumption_count_vs_response_time.png", "Assumption Count by Response Time", "Assumption count in turn"):
        png_outputs.append("exp5_assumption_count_vs_response_time.png")
    if save_plot_scatter(load_y, load_x, output_dir / "exp5_implicature_load_vs_response_time.png", "Implicature Load by Response Time", "Implicature Load L"):
        png_outputs.append("exp5_implicature_load_vs_response_time.png")
    if save_ridge(rows, output_dir / "exp5_load_ridge_by_response_type.png"):
        png_outputs.append("exp5_load_ridge_by_response_type.png")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "num_turn_rows": len(rows),
        "silence_gap_threshold_sec": silence_gap,
        "silence_gap_quantile": args.silence_gap_quantile,
        "min_silence_gap_sec": args.min_silence_gap,
        "assumption_sharedness_method": {
            "method": "MiniLM cosine similarity against prior episode assumptions",
            "model": "all-MiniLM-L6-v2",
            "similarity_threshold": args.assumption_similarity_threshold,
        },
        "assumption_count_vs_response_time_correlation": corr_stats(assumption_x, assumption_y),
        "implicature_load_vs_response_time_correlation": corr_stats(load_x, load_y),
        "response_type_model": None if not model else {"model_family": "bayesian_multinomial_logit_laplace", "prior_sd": model["prior_sd"], "posterior_draws": model["posterior_draws"], "reference_class": model["reference_class"], "approximation": model["approximation"]},
        "response_delay_model_specification": {
            "active_spec": "primary_spec",
            "description": (
                "Experiment 5 primary response-delay specification uses implicature_load and "
                "average_response_time_0_to_n_minus_1 only."
            ),
            "excluded_controls": ["explicit_statement_count"],
        },
        "response_delay_regression": response_delay_regression,
        "response_delay_regression_distribution_checks": response_delay_distribution_checks,
        "response_type_counts": {k: int(v) for k, v in response_counts.items()},
        "tqdm_enabled": show_progress,
        "png_outputs": png_outputs,
    }
    (output_dir / "exp5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=jdefault))


if __name__ == "__main__":
    main()
