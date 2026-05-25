import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

RNG_SEED = 42
np.random.seed(RNG_SEED)
RESPONSE_CLASSES = ["Backchannel", "Substantive", "Clarification", "Silence/Abandonment"]
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QWEN_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
TURN_LEVEL_FEATURE_FIELDS = [
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


def assumption_embedder(embedding_model_name):
    embedding_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assumption_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
    assumption_model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True).to(embedding_device)
    assumption_model.eval()
    return assumption_tokenizer, assumption_model, embedding_device


def assumption_embeddings(texts, assumption_tokenizer, assumption_model, embedding_device, embedding_batch_size):
    texts = [t for t in texts if t]
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embedding_batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), embedding_batch_size):
            batch_texts = texts[start:start + embedding_batch_size]
            batch = assumption_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            batch = {key: value.to(embedding_device) for key, value in batch.items()}
            output = assumption_model(**batch)
            hidden = output.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.shape).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embedding_batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(embedding_batches)


def resolve_output_dir(base_output_dir, embedding_model_name):
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embedding_model_name.replace("/", "__").strip())
    if not model_slug:
        raise ValueError("embedding_model_name must not be empty.")
    return Path(base_output_dir) / model_slug


def validate_patch_args(num_patches, patch_index, episodes_per_patch):
    if num_patches < 1:
        raise ValueError(f"num_patches must be >= 1, got {num_patches}")
    if patch_index < 0 or patch_index >= num_patches:
        raise ValueError(f"patch_index must be in [0, {num_patches - 1}], got {patch_index}")
    if episodes_per_patch is not None and episodes_per_patch < 1:
        raise ValueError(f"episodes_per_patch must be >= 1, got {episodes_per_patch}")


def resolve_patch_output_dir(base_output_dir, num_patches, patch_index):
    if num_patches == 1:
        return Path(base_output_dir)
    return Path(base_output_dir) / "patches" / f"patch_{patch_index:04d}_of_{num_patches:04d}"


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    raise TypeError(type(x).__name__)


def normalize_text(s):
    s = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(s or "").strip().lower()))
    return s.strip()


def unique_texts(raw):
    seen, out = set(), []
    for item in raw if isinstance(raw, list) else []:
        text = item.get("text", "") if isinstance(item, dict) else item if isinstance(item, str) else ""
        text = normalize_text(text)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def turn_duration_seconds(turn):
    dur = to_float(turn.get("duration"))
    if math.isfinite(dur) and dur >= 0:
        return dur
    start = to_float(turn.get("startTime", turn.get("start_time")))
    end = to_float(turn.get("endTime", turn.get("end_time")))
    return max(0.0, end - start) if math.isfinite(start) and math.isfinite(end) else float("nan")


def gap_seconds(a, b):
    end = to_float(a.get("endTime", a.get("end_time")))
    start = to_float(b.get("startTime", b.get("start_time")))
    return start - end if math.isfinite(start) and math.isfinite(end) else float("nan")


def word_count(turn):
    wc = to_float(turn.get("wordCount", turn.get("word_count")))
    return wc if math.isfinite(wc) else float(len(str(turn.get("turn_text", "") or "").split()) or "nan")


def sort_turns(turns):
    return [x[2] for x in sorted(((int(t.get("turn_idx", i)) if str(t.get("turn_idx", i)).lstrip("-").isdigit() else i, i, t) for i, t in enumerate(turns)))]


def collect_episode_paths(input_dir):
    return sorted(input_dir.glob("*.json")) or sorted(input_dir.glob("*/*.json"))


def select_patch_paths(paths, num_patches, patch_index, episodes_per_patch):
    if episodes_per_patch is not None:
        start = patch_index * episodes_per_patch
        end = min(start + episodes_per_patch, len(paths))
        return paths[start:end]
    return [path for idx, path in enumerate(paths) if idx % num_patches == patch_index]


def load_episodes_from_paths(paths, show_progress, desc):
    episodes, gaps = [], []
    for fp in tqdm(paths, desc=desc, unit="file", disable=not show_progress):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list) or not payload:
            continue
        turns = sort_turns(payload)
        episodes.append((str(turns[0].get("episode_id", fp.stem)), turns))
        gaps.extend(g for a, b in zip(turns, turns[1:]) if math.isfinite(g := gap_seconds(a, b)))
    return episodes, np.asarray(gaps, dtype=float)


def load_episodes(input_dir, show_progress):
    return load_episodes_from_paths(collect_episode_paths(input_dir), show_progress, "Loading episodes")


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
        if ((not math.isfinite(turn_duration_seconds(next_turn))) or turn_duration_seconds(next_turn) <= agree_dur_max) and (
            (not math.isfinite(word_count(next_turn))) or word_count(next_turn) <= agree_words_max
        ):
            return "Backchannel"
    return "Substantive"


def build_rows(episodes, silence_gap, agree_dur_max, agree_words_max, show_progress, assumption_similarity_threshold, assumption_tokenizer, assumption_model, embedding_device, embedding_batch_size):
    rows = []
    for episode_id, turns in tqdm(episodes, desc="Building turn rows", unit="episode", disable=not show_progress):
        history_embs, gap_sum, gap_n = None, 0.0, 0
        for i, turn in enumerate(turns):
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            assumptions = unique_texts(turn.get("assumptions", []))
            explicit = unique_texts(turn.get("explicit_propositions", []))
            new_n, current_new, current_new_indices = 0, [], []
            if assumptions:
                current_embs = assumption_embeddings(
                    assumptions,
                    assumption_tokenizer,
                    assumption_model,
                    embedding_device,
                    embedding_batch_size,
                )
                if history_embs is None or len(history_embs) == 0:
                    new_n, current_new = len(assumptions), list(assumptions)
                    current_new_indices = list(range(len(assumptions)))
                else:
                    sims = history_embs @ current_embs.T
                    for idx, text in enumerate(assumptions):
                        max_sim = float(np.max(sims[:, idx])) if sims.size else 0.0
                        if max_sim < assumption_similarity_threshold:
                            new_n += 1
                            current_new.append(text)
                            current_new_indices.append(idx)
            duration = turn_duration_seconds(turn)
            g = gap_seconds(turn, nxt) if nxt else float("nan")
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
                new_embs = current_embs[np.asarray(current_new_indices, dtype=int)]
                history_embs = new_embs if history_embs is None else np.vstack([history_embs, new_embs])
            if math.isfinite(g) and g >= 0:
                gap_sum += g
                gap_n += 1
    return sorted(rows, key=lambda r: (str(r["episode_id"]), int(r["turn_idx"])))


def standardize(x):
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / (sd if math.isfinite(sd) and sd > 1e-12 else 1.0), mu, (sd if math.isfinite(sd) and sd > 1e-12 else 1.0)


def standardize_matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be a two-dimensional predictor matrix, got shape {x.shape}")
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (x - mu) / sd, mu, sd


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def unpack(theta, p, k):
    w = np.zeros((p, k), dtype=float)
    if k > 1:
        w[:, :-1] = theta.reshape(p, k - 1)
    return w


def fit_multinomial_bayes(x, y_labels, predictor_names, prior_sd, posterior_draws):
    if len(predictor_names) == 0:
        raise ValueError("predictor_names must contain at least one predictor.")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be a two-dimensional predictor matrix, got shape {x.shape}")
    if x.shape[1] != len(predictor_names):
        raise ValueError(
            f"x has {x.shape[1]} predictor columns, but predictor_names has {len(predictor_names)} entries."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite predictor values.")
    classes = sorted(set(y_labels))
    if len(classes) < 2:
        raise ValueError(f"Multinomial logit requires at least two classes, got {classes}")
    if len(y_labels) != x.shape[0]:
        raise ValueError(f"x row count {x.shape[0]} does not match label count {len(y_labels)}.")
    y = np.asarray([classes.index(v) for v in y_labels], dtype=int)
    xz, mu, sd = standardize_matrix(x)
    X, p, k = np.column_stack([np.ones(xz.shape[0], dtype=float), xz]), xz.shape[1] + 1, len(classes)
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
        raise RuntimeError(f"Multinomial logit fit failed with status {opt.status}: {opt.message}")
    theta_map = np.asarray(opt.x, dtype=float)
    probs = softmax(X @ unpack(theta_map, p, k))
    info = np.eye(p * (k - 1), dtype=float) / prior_var
    for i in range(len(y)):
        class_info = np.diag(probs[i, :-1]) - np.outer(probs[i, :-1], probs[i, :-1])
        predictor_info = np.outer(X[i], X[i])
        info += np.kron(predictor_info, class_info)
    cov = np.linalg.pinv(info)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.multivariate_normal(theta_map, cov, size=max(int(posterior_draws), 200))
    w_draws = np.stack([unpack(d, p, k) for d in draws], axis=0)
    return {
        "classes": classes,
        "reference_class": classes[-1],
        "predictor_names": list(predictor_names),
        "x_mean": mu,
        "x_std": sd,
        "W_map": unpack(theta_map, p, k),
        "posterior_W": w_draws,
        "prior_sd": float(prior_sd),
        "posterior_draws": int(len(w_draws)),
        "approximation": "laplace_map_gaussian_posterior",
    }


def posterior_multinomial_proba(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be a two-dimensional predictor matrix, got shape {x.shape}")
    if x.shape[1] != len(model["predictor_names"]):
        raise ValueError(
            f"x has {x.shape[1]} predictor columns, but model has {len(model['predictor_names'])} predictors."
        )
    x = (x - model["x_mean"]) / model["x_std"]
    X = np.column_stack([np.ones(x.shape[0], dtype=float), x])
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


def prefixed_coef_interval_summary(prefix, draws, map_value):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {
            f"posterior_mean_{prefix}": None,
            f"posterior_sd_{prefix}": None,
            f"posterior_q025_{prefix}": None,
            f"posterior_q975_{prefix}": None,
            f"posterior_prob_{prefix}_gt_0": None,
            f"posterior_prob_{prefix}_lt_0": None,
            f"map_{prefix}": float(map_value),
        }
    return {
        f"posterior_mean_{prefix}": float(np.mean(draws)),
        f"posterior_sd_{prefix}": float(np.std(draws, ddof=1)) if len(draws) >= 2 else 0.0,
        f"posterior_q025_{prefix}": float(np.quantile(draws, 0.025)),
        f"posterior_q975_{prefix}": float(np.quantile(draws, 0.975)),
        f"posterior_prob_{prefix}_gt_0": float(np.mean(draws > 0.0)),
        f"posterior_prob_{prefix}_lt_0": float(np.mean(draws < 0.0)),
        f"map_{prefix}": float(map_value),
    }


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


def write_turn_level_feature_csv(rows, output_dir):
    save_csv(output_dir / "exp5_turn_level_features.csv", rows, TURN_LEVEL_FEATURE_FIELDS)


def build_base_summary(rows, output_dir, input_dir, num_episodes, silence_gap, args, show_progress, selected_episode_file_count, candidate_episode_file_count):
    response_counts = Counter(str(r.get("next_response_type")) for r in rows if r.get("next_response_type"))
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_episodes": int(num_episodes),
        "num_turn_rows": len(rows),
        "selected_episode_file_count": int(selected_episode_file_count),
        "candidate_episode_file_count": int(candidate_episode_file_count),
        "silence_gap_threshold_sec": silence_gap,
        "silence_gap_threshold_source": (
            "provided"
            if args.silence_gap_threshold_sec is not None
            else "computed_from_input"
        ),
        "silence_gap_quantile": args.silence_gap_quantile,
        "min_silence_gap_sec": args.min_silence_gap,
        "embedding_model_name": args.embedding_model_name,
        "embedding_batch_size": int(args.embedding_batch_size),
        "embedding_device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_patches": int(args.num_patches),
        "patch_index": int(args.patch_index),
        "episodes_per_patch": int(args.episodes_per_patch) if args.episodes_per_patch is not None else None,
        "bayes_multinomial_prior_sd": args.bayes_multinomial_prior_sd,
        "bayes_multinomial_draws": args.bayes_multinomial_draws,
        "bayes_linear_draws": args.bayes_linear_draws,
        "bayes_linear_prior_precision_scale": args.bayes_linear_prior_precision_scale,
        "bayes_linear_prior_a0": args.bayes_linear_prior_a0,
        "bayes_linear_prior_b0": args.bayes_linear_prior_b0,
        "assumption_sharedness_method": {
            "method": "Selected embedding model cosine similarity against prior episode assumptions",
            "model": args.embedding_model_name,
            "default_model": DEFAULT_EMBEDDING_MODEL_NAME,
            "recommended_qwen_model": DEFAULT_QWEN_EMBEDDING_MODEL_NAME,
            "similarity_threshold": args.assumption_similarity_threshold,
        },
        "response_type_model": None,
        "response_delay_regression": None,
        "response_delay_regression_distribution_checks": {},
        "response_type_counts": {k: int(v) for k, v in response_counts.items()},
        "tqdm_enabled": show_progress,
    }


def write_patch_outputs(rows, output_dir, input_dir, num_episodes, silence_gap, args, show_progress, selected_episode_file_count, candidate_episode_file_count):
    write_turn_level_feature_csv(rows, output_dir)
    summary = build_base_summary(
        rows,
        output_dir,
        input_dir,
        num_episodes,
        silence_gap,
        args,
        show_progress,
        selected_episode_file_count,
        candidate_episode_file_count,
    )
    summary.update(
        {
            "analysis_stage": "patch_feature_extraction_only",
            "deferred_outputs": [
                "exp5_probability_curves.csv",
                "exp5_logit_coefficients.csv",
                "exp5_response_delay_regression_coefficients.csv",
                "exp5_response_delay_regression_summary.json",
                "exp5_response_type_counts.csv",
            ],
            "notes": [
                "Patch mode writes turn-level features only.",
                "Fit the response models by running merge_exp5_patches.py after all patches finish.",
            ],
        }
    )
    (output_dir / "exp5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    return summary


def build_probability_curves(model, loads, duration_values):
    if list(model["predictor_names"]) != ["implicature_load", "duration_sec"]:
        raise ValueError(f"Expected predictors ['implicature_load', 'duration_sec'], got {model['predictor_names']}")
    if len(loads) == 0:
        return []
    duration_values = np.asarray(duration_values, dtype=float)
    duration_values = duration_values[np.isfinite(duration_values)]
    if len(duration_values) == 0:
        raise ValueError("duration_values must contain at least one finite value.")
    duration_control = float(np.median(duration_values))
    lo, hi = robust_xlim(loads)
    grid = np.linspace(lo, hi, 260)
    predictors = np.column_stack([grid, np.full(len(grid), duration_control, dtype=float)])
    probs = np.mean(posterior_multinomial_proba(model, predictors), axis=0)
    return [
        {
            "implicature_load": float(grid[i]),
            "duration_sec_control_value": duration_control,
            **{f"p_{cls}": float(probs[i, j]) for j, cls in enumerate(model["classes"])},
        }
        for i in range(len(grid))
    ]


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


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Experiment 5: Implicature Load vs Engagement")
    ap.add_argument("--input_dir", default="data/conversation_moves_labeled")
    ap.add_argument("--output_dir", default="experiments/exp5_processing_load/results")
    ap.add_argument("--embedding_model_name", default=DEFAULT_EMBEDDING_MODEL_NAME)
    ap.add_argument("--embedding_batch_size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    ap.add_argument("--num_patches", type=int, default=1)
    ap.add_argument("--patch_index", type=int, default=0)
    ap.add_argument("--episodes_per_patch", type=int, default=None)
    ap.add_argument("--silence_gap_quantile", type=float, default=0.95)
    ap.add_argument("--min_silence_gap", type=float, default=5.0)
    ap.add_argument("--silence_gap_threshold_sec", type=float, default=None)
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
    return ap


def analyze_rows(rows, output_dir, input_dir, num_episodes, silence_gap, args, show_progress, summary_extra=None):
    model_rows = [
        r
        for r in rows
        if all(isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k])) for k in ["implicature_load", "duration_sec"])
        and float(r["duration_sec"]) >= 0
        and r.get("next_response_type")
    ]
    model = curve_rows = None
    coef_rows = []
    if model_rows:
        predictor_names = ["implicature_load", "duration_sec"]
        x = np.asarray([[float(r["implicature_load"]), float(r["duration_sec"])] for r in model_rows], dtype=float)
        y = [str(r["next_response_type"]) for r in model_rows]
        keep = {k for k, v in Counter(y).items() if v >= 2}
        keep_indices = [i for i, yi in enumerate(y) if yi in keep]
        x = x[np.asarray(keep_indices, dtype=int)]
        y = [y[i] for i in keep_indices]
        if len(set(y)) >= 2 and len(x) >= 10:
            model = fit_multinomial_bayes(x, y, predictor_names, args.bayes_multinomial_prior_sd, args.bayes_multinomial_draws)
        if model:
            curve_rows = build_probability_curves(model, x[:, 0], x[:, 1])
            for j, cls in enumerate(model["classes"]):
                w = model["posterior_W"][:, :, j]
                coef_rows.append(
                    {
                        "class": cls,
                        "reference_class": model["reference_class"],
                        **prefixed_coef_interval_summary("intercept", w[:, 0], model["W_map"][0, j]),
                        **prefixed_coef_interval_summary("coef_scaled_load", w[:, 1], model["W_map"][1, j]),
                        **prefixed_coef_interval_summary("coef_scaled_duration_sec", w[:, 2], model["W_map"][2, j]),
                        "load_scaler_mean": float(model["x_mean"][0]),
                        "load_scaler_std": float(model["x_std"][0]),
                        "duration_sec_scaler_mean": float(model["x_mean"][1]),
                        "duration_sec_scaler_std": float(model["x_std"][1]),
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

    write_turn_level_feature_csv(rows, output_dir)
    if curve_rows:
        save_csv(output_dir / "exp5_probability_curves.csv", curve_rows, ["implicature_load"] + [k for k in curve_rows[0] if k != "implicature_load"])
    if coef_rows:
        save_csv(
            output_dir / "exp5_logit_coefficients.csv",
            coef_rows,
            [
                "class",
                "reference_class",
                "posterior_mean_intercept",
                "posterior_sd_intercept",
                "posterior_q025_intercept",
                "posterior_q975_intercept",
                "posterior_prob_intercept_gt_0",
                "posterior_prob_intercept_lt_0",
                "map_intercept",
                "posterior_mean_coef_scaled_load",
                "posterior_sd_coef_scaled_load",
                "posterior_q025_coef_scaled_load",
                "posterior_q975_coef_scaled_load",
                "posterior_prob_coef_scaled_load_gt_0",
                "posterior_prob_coef_scaled_load_lt_0",
                "map_coef_scaled_load",
                "posterior_mean_coef_scaled_duration_sec",
                "posterior_sd_coef_scaled_duration_sec",
                "posterior_q025_coef_scaled_duration_sec",
                "posterior_q975_coef_scaled_duration_sec",
                "posterior_prob_coef_scaled_duration_sec_gt_0",
                "posterior_prob_coef_scaled_duration_sec_lt_0",
                "map_coef_scaled_duration_sec",
                "load_scaler_mean",
                "load_scaler_std",
                "duration_sec_scaler_mean",
                "duration_sec_scaler_std",
            ],
        )
    if response_delay_regression_coeffs:
        save_csv(output_dir / "exp5_response_delay_regression_coefficients.csv", response_delay_regression_coeffs, ["term", "posterior_mean", "posterior_sd"])
    if response_delay_regression is not None:
        (output_dir / "exp5_response_delay_regression_summary.json").write_text(json.dumps({"regression": response_delay_regression, "distribution_checks": response_delay_distribution_checks}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    response_counts = Counter(str(r.get("next_response_type")) for r in rows if r.get("next_response_type"))
    save_csv(output_dir / "exp5_response_type_counts.csv", [{"response_type": k, "count": v} for k, v in sorted(response_counts.items())], ["response_type", "count"])

    preferred_png_outputs = [
        "exp5_probability_curves.png",
        "exp5_assumption_count_vs_response_time.png",
        "exp5_implicature_load_vs_response_time.png",
        "exp5_load_ridge_by_response_type.png",
    ]
    observed_png_outputs = {path.name for path in output_dir.glob("*.png")}
    png_outputs = [name for name in preferred_png_outputs if name in observed_png_outputs]
    png_outputs.extend(sorted(observed_png_outputs - set(preferred_png_outputs)))
    response_type_model_summary = None
    if model:
        response_type_model_summary = {
            "model_family": "bayesian_multinomial_logit_laplace",
            "prior_sd": model["prior_sd"],
            "posterior_draws": model["posterior_draws"],
            "reference_class": model["reference_class"],
            "approximation": model["approximation"],
            "fitted_formula": "next_response_type ~ implicature_load + duration_sec",
            "predictors": list(model["predictor_names"]),
            "controlled_predictor": "duration_sec",
            "predictor_standardization": {
                str(name): {"mean": float(model["x_mean"][i]), "std": float(model["x_std"][i])}
                for i, name in enumerate(model["predictor_names"])
            },
            "coefficient_interval_level": 0.95,
        }
        if curve_rows:
            response_type_model_summary["probability_curve_control_values"] = {
                "duration_sec": float(curve_rows[0]["duration_sec_control_value"])
            }

    summary = build_base_summary(
        rows,
        output_dir,
        input_dir,
        num_episodes,
        silence_gap,
        args,
        show_progress,
        0,
        0,
    )
    summary.update(
        {
            "png_outputs": png_outputs,
            "analysis_stage": "full_analysis",
            "assumption_count_vs_response_time_correlation": corr_stats(assumption_x, assumption_y),
            "implicature_load_vs_response_time_correlation": corr_stats(load_x, load_y),
            "response_type_model": response_type_model_summary,
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
        }
    )
    if summary_extra:
        summary.update(summary_extra)
    (output_dir / "exp5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    return summary


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    validate_patch_args(args.num_patches, args.patch_index, args.episodes_per_patch)
    input_dir = Path(args.input_dir)
    model_output_dir = resolve_output_dir(args.output_dir, args.embedding_model_name)
    output_dir = resolve_patch_output_dir(model_output_dir, args.num_patches, args.patch_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_tqdm

    all_paths = collect_episode_paths(input_dir)
    selected_paths = select_patch_paths(all_paths, args.num_patches, args.patch_index, args.episodes_per_patch)
    if not selected_paths:
        raise RuntimeError(
            f"No episode files selected for patch {args.patch_index} out of {args.num_patches}. "
            f"Candidate file count: {len(all_paths)}"
        )
    print(json.dumps({"selected_episode_file_count": len(selected_paths), "candidate_episode_file_count": len(all_paths), "patch_index": args.patch_index, "num_patches": args.num_patches}))

    if args.num_patches == 1:
        episodes, gaps = load_episodes_from_paths(selected_paths, show_progress, "Loading episodes")
    else:
        episodes, _ = load_episodes_from_paths(selected_paths, show_progress, "Loading selected episodes")
        gaps = np.asarray([], dtype=float)

    assumption_tokenizer, assumption_model, embedding_device = assumption_embedder(args.embedding_model_name)
    if args.silence_gap_threshold_sec is not None:
        silence_gap = float(args.silence_gap_threshold_sec)
    else:
        nonneg = gaps[np.isfinite(gaps) & (gaps >= 0)]
        silence_gap = max(float(args.min_silence_gap), float(np.quantile(nonneg, args.silence_gap_quantile))) if len(nonneg) else float(args.min_silence_gap)
    rows = build_rows(
        episodes,
        silence_gap,
        args.backchannel_agree_duration_max,
        args.backchannel_agree_words_max,
        show_progress,
        args.assumption_similarity_threshold,
        assumption_tokenizer,
        assumption_model,
        embedding_device,
        args.embedding_batch_size,
    )
    summary_extra = {
        "selected_episode_file_count": int(len(selected_paths)),
        "candidate_episode_file_count": int(len(all_paths)),
    }
    if args.num_patches > 1:
        write_patch_outputs(
            rows,
            output_dir,
            input_dir,
            len(episodes),
            silence_gap,
            args,
            show_progress,
            int(len(selected_paths)),
            int(len(all_paths)),
        )
        return
    analyze_rows(
        rows=rows,
        output_dir=output_dir,
        input_dir=input_dir,
        num_episodes=len(episodes),
        silence_gap=silence_gap,
        args=args,
        show_progress=show_progress,
        summary_extra=summary_extra,
    )


if __name__ == "__main__":
    main()
