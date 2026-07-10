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

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # Merge-only baseline analysis can run from extracted CSVs without Transformers.
    AutoModel = None
    AutoTokenizer = None

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
    "word_count_in_turn",
    "assumption_count_in_turn",
    "explicit_statement_count",
    "new_assumption_count",
    "new_assumptions_per_second",
    "implicature_load",
    "turn_similarity_to_previous",
    "previous_response_type",
    "current_turn_type_label",
    "current_conversation_move_label",
    "previous_conversation_move_label",
    "response_delay_at_time_n",
    "gap_to_next_sec",
    "average_response_time_0_to_n_minus_1",
    "next_response_type",
    "next_turn_type_label",
    "next_conversation_move_label",
]

BASELINE_MODEL_SPECS = [
    {
        "model_id": "surface_basic",
        "description": "Turn duration and word count only.",
        "numeric_features": ["duration_sec", "word_count_in_turn"],
        "categorical_features": [],
    },
    {
        "model_id": "surface_plus_explicit",
        "description": "Basic surface features plus explicit-proposition count.",
        "numeric_features": ["duration_sec", "word_count_in_turn", "explicit_statement_count"],
        "categorical_features": [],
    },
    {
        "model_id": "surface_plus_history",
        "description": "Surface and explicit features plus prior response history.",
        "numeric_features": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
        ],
        "categorical_features": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
    {
        "model_id": "surface_plus_similarity",
        "description": "Surface, explicit, and history features plus adjacent-turn embedding similarity.",
        "numeric_features": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
            "turn_similarity_to_previous",
        ],
        "categorical_features": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
    {
        "model_id": "surface_plus_implicit_density",
        "description": "Full surface baseline plus implicit-assumption count and density.",
        "numeric_features": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
            "turn_similarity_to_previous",
            "assumption_count_in_turn",
            "new_assumption_count",
            "new_assumptions_per_second",
        ],
        "categorical_features": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
    {
        "model_id": "surface_plus_implicit_load",
        "description": "Full surface baseline plus implicit-assumption counts and seconds per new assumption.",
        "numeric_features": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
            "turn_similarity_to_previous",
            "assumption_count_in_turn",
            "new_assumption_count",
            "implicature_load",
        ],
        "categorical_features": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
]

LOG1P_BASELINE_FEATURES = {
    "duration_sec",
    "word_count_in_turn",
    "explicit_statement_count",
    "average_response_time_0_to_n_minus_1",
    "assumption_count_in_turn",
    "new_assumption_count",
    "new_assumptions_per_second",
    "implicature_load",
}


def assumption_embedder(embedding_model_name):
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            "Transformers is required for feature extraction. Install `transformers`, or run "
            "merge_exp5_patches.py on already extracted patch CSVs."
        )
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


def embeddings_preserve_order(texts, assumption_tokenizer, assumption_model, embedding_device, embedding_batch_size):
    """Embed non-empty strings while preserving the original row alignment."""
    clean = [str(text or "").strip() for text in texts]
    valid_indices = [idx for idx, text in enumerate(clean) if text]
    if not valid_indices:
        return [None] * len(clean)
    embedded = assumption_embeddings(
        [clean[idx] for idx in valid_indices],
        assumption_tokenizer,
        assumption_model,
        embedding_device,
        embedding_batch_size,
    )
    result = [None] * len(clean)
    for idx, vector in zip(valid_indices, embedded):
        result[idx] = vector
    return result


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
        turn_text_embeddings = embeddings_preserve_order(
            [turn.get("turn_text", "") for turn in turns],
            assumption_tokenizer,
            assumption_model,
            embedding_device,
            embedding_batch_size,
        )
        history_embs, gap_sum, gap_n = None, 0.0, 0
        for i, turn in enumerate(turns):
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            prev = turns[i - 1] if i > 0 else None
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
            prev_gap = gap_seconds(prev, turn) if prev is not None else float("nan")
            previous_response_type = (
                classify(turn, prev_gap, silence_gap, agree_dur_max, agree_words_max)
                if prev is not None
                else "EpisodeStart"
            )
            previous_similarity = float("nan")
            if i > 0 and turn_text_embeddings[i] is not None and turn_text_embeddings[i - 1] is not None:
                previous_similarity = float(np.dot(turn_text_embeddings[i], turn_text_embeddings[i - 1]))
            rows.append(
                {
                    "episode_id": episode_id,
                    "turn_idx": int(turn.get("turn_idx", i)) if str(turn.get("turn_idx", i)).lstrip("-").isdigit() else i,
                    "duration_sec": duration,
                    "word_count_in_turn": word_count(turn),
                    "assumption_count_in_turn": len(assumptions),
                    "explicit_statement_count": len(explicit),
                    "new_assumption_count": new_n,
                    "new_assumptions_per_second": (
                        new_n / duration
                        if math.isfinite(duration) and duration > 0 and new_n >= 0
                        else float("nan")
                    ),
                    "implicature_load": duration / new_n if math.isfinite(duration) and duration >= 0 and new_n > 0 else float("nan"),
                    "turn_similarity_to_previous": previous_similarity,
                    "previous_response_type": previous_response_type,
                    "current_turn_type_label": turn.get("turn_type_label"),
                    "current_conversation_move_label": turn.get("conversation_move_label"),
                    "previous_conversation_move_label": (prev or {}).get("conversation_move_label"),
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


def ordered_response_classes(labels):
    observed = set(str(label) for label in labels)
    return [label for label in RESPONSE_CLASSES if label in observed] + sorted(observed - set(RESPONSE_CLASSES))


def fit_multinomial_map_matrix(x, y_labels, feature_names, prior_sd=2.5):
    """Regularized multinomial logit with a fixed zero reference class."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(y_labels):
        raise ValueError("x must be a two-dimensional matrix aligned with y_labels.")
    classes = ordered_response_classes(y_labels)
    if len(classes) < 2:
        return None
    y = np.asarray([classes.index(str(value)) for value in y_labels], dtype=int)
    X = np.column_stack([np.ones(len(x), dtype=float), x])
    p, k = X.shape[1], len(classes)
    prior_var = max(float(prior_sd) ** 2, 1e-12)

    def objective(theta):
        w = unpack(theta, p, k)
        probs = softmax(X @ w)
        log_likelihood = float(np.sum(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))
        log_prior = -0.5 * float(np.sum(theta * theta) / prior_var)
        residual = -probs
        residual[np.arange(len(y)), y] += 1.0
        gradient = (X.T @ residual)[:, :-1].reshape(-1) - theta / prior_var
        return -(log_likelihood + log_prior), -gradient

    opt = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=np.zeros(p * (k - 1), dtype=float),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    if not opt.success:
        return None
    return {
        "classes": classes,
        "reference_class": classes[-1],
        "feature_names": list(feature_names),
        "W_map": unpack(np.asarray(opt.x, dtype=float), p, k),
        "prior_sd": float(prior_sd),
        "optimization_message": str(opt.message),
        "optimization_iterations": int(getattr(opt, "nit", 0)),
    }


def predict_multinomial_map_matrix(model, x):
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones(len(x), dtype=float), x])
    return softmax(X @ model["W_map"])


def baseline_numeric_value(row, feature_name):
    value = to_float(row.get(feature_name))
    if not math.isfinite(value):
        return float("nan")
    if feature_name in LOG1P_BASELINE_FEATURES:
        return math.log1p(max(value, 0.0))
    return value


def fit_baseline_preprocessor(train_rows, spec):
    numeric_state = []
    feature_names = []
    for feature_name in spec["numeric_features"]:
        values = np.asarray([baseline_numeric_value(row, feature_name) for row in train_rows], dtype=float)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else 0.0
        add_missing_indicator = bool(np.any(~np.isfinite(values)))
        imputed = np.where(np.isfinite(values), values, median)
        mean = float(np.mean(imputed))
        std = float(np.std(imputed))
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0
        numeric_state.append(
            {
                "feature": feature_name,
                "transform": "log1p" if feature_name in LOG1P_BASELINE_FEATURES else "raw",
                "median": median,
                "mean": mean,
                "std": std,
                "add_missing_indicator": add_missing_indicator,
            }
        )
        feature_names.append(term(feature_name, "log1p") if feature_name in LOG1P_BASELINE_FEATURES else feature_name)
        if add_missing_indicator:
            feature_names.append(f"{feature_name}_missing")

    categorical_state = []
    for feature_name in spec["categorical_features"]:
        if feature_name == "previous_response_type":
            categories = ["EpisodeStart"] + [label for label in RESPONSE_CLASSES if label != "Substantive"]
            reference = "Substantive"
        else:
            observed = [str(row.get(feature_name) or "__MISSING__") for row in train_rows]
            counts = Counter(observed)
            reference = counts.most_common(1)[0][0] if counts else "__MISSING__"
            categories = sorted(value for value in counts if value != reference)
        categorical_state.append({"feature": feature_name, "categories": categories, "reference": reference})
        feature_names.extend([f"{feature_name}={category}" for category in categories])

    return {
        "numeric": numeric_state,
        "categorical": categorical_state,
        "feature_names": feature_names,
    }


def apply_baseline_preprocessor(rows, preprocessor):
    columns = []
    for state in preprocessor["numeric"]:
        values = np.asarray([baseline_numeric_value(row, state["feature"]) for row in rows], dtype=float)
        missing = ~np.isfinite(values)
        values = np.where(np.isfinite(values), values, state["median"])
        columns.append((values - state["mean"]) / state["std"])
        if state.get("add_missing_indicator"):
            columns.append(missing.astype(float))
    for state in preprocessor["categorical"]:
        if state["feature"] == "previous_response_type":
            raw = [str(row.get(state["feature"]) or "EpisodeStart") for row in rows]
        else:
            raw = [str(row.get(state["feature"]) or "__MISSING__") for row in rows]
        for category in state["categories"]:
            columns.append(np.asarray([1.0 if value == category else 0.0 for value in raw], dtype=float))
    if not columns:
        return np.empty((len(rows), 0), dtype=float)
    return np.column_stack(columns)


def split_rows_by_episode(rows, test_fraction, seed):
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    if len(episode_ids) < 2:
        raise RuntimeError("At least two episodes are required for held-out baseline evaluation.")
    n_test = min(max(int(round(len(episode_ids) * float(test_fraction))), 1), len(episode_ids) - 1)
    all_classes = set(str(row["next_response_type"]) for row in rows)
    best = None
    for attempt in range(250):
        rng = np.random.default_rng(int(seed) + attempt)
        shuffled = np.asarray(episode_ids, dtype=object)
        rng.shuffle(shuffled)
        test_ids = set(str(value) for value in shuffled[:n_test])
        train_rows = [row for row in rows if str(row["episode_id"]) not in test_ids]
        test_rows = [row for row in rows if str(row["episode_id"]) in test_ids]
        train_classes = set(str(row["next_response_type"]) for row in train_rows)
        test_classes = set(str(row["next_response_type"]) for row in test_rows)
        score = len(train_classes & all_classes) + len(test_classes & all_classes)
        if best is None or score > best[0]:
            best = (score, train_rows, test_rows, test_ids, attempt)
        if train_classes == all_classes and test_classes == all_classes:
            return train_rows, test_rows, test_ids, int(seed) + attempt
    _, train_rows, test_rows, test_ids, attempt = best
    if set(str(row["next_response_type"]) for row in train_rows) != all_classes:
        raise RuntimeError("Could not construct an episode split containing every response class in training data.")
    return train_rows, test_rows, test_ids, int(seed) + attempt


def confusion_and_logloss(y_true, probabilities, classes):
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[str(label)] for label in y_true], dtype=int)
    pred_idx = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(classes), len(classes)), dtype=float)
    np.add.at(confusion, (y_idx, pred_idx), 1.0)
    logloss_sum = -float(np.sum(np.log(np.clip(probabilities[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))))
    return confusion, logloss_sum


def metrics_from_confusion(confusion, logloss_sum):
    n = float(np.sum(confusion))
    recalls, f1s = [], []
    for idx in range(confusion.shape[0]):
        tp = confusion[idx, idx]
        support = float(np.sum(confusion[idx, :]))
        predicted = float(np.sum(confusion[:, idx]))
        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted if predicted > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "n_test_rows": int(n),
        "log_loss": float(logloss_sum / n) if n > 0 else None,
        "accuracy": float(np.trace(confusion) / n) if n > 0 else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
    }


def episode_prediction_statistics(y_true, probabilities, episode_ids, classes):
    stats = {}
    episode_ids = np.asarray([str(value) for value in episode_ids], dtype=object)
    y_true = np.asarray([str(value) for value in y_true], dtype=object)
    for episode_id in sorted(set(episode_ids.tolist())):
        mask = episode_ids == episode_id
        confusion, logloss_sum = confusion_and_logloss(y_true[mask], probabilities[mask], classes)
        stats[episode_id] = {"confusion": confusion, "logloss_sum": logloss_sum}
    return stats


def aggregate_episode_statistics(stats, sampled_episode_ids):
    first = next(iter(stats.values()))
    confusion = np.zeros_like(first["confusion"], dtype=float)
    logloss_sum = 0.0
    for episode_id in sampled_episode_ids:
        confusion += stats[str(episode_id)]["confusion"]
        logloss_sum += float(stats[str(episode_id)]["logloss_sum"])
    return metrics_from_confusion(confusion, logloss_sum)


def bootstrap_metric_intervals(y_true, probabilities, episode_ids, classes, draws, seed):
    stats = episode_prediction_statistics(y_true, probabilities, episode_ids, classes)
    unique_episode_ids = sorted(stats)
    base = aggregate_episode_statistics(stats, unique_episode_ids)
    rng = np.random.default_rng(int(seed))
    sampled_metrics = {key: [] for key in ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]}
    for _ in range(max(int(draws), 1)):
        sampled = rng.choice(unique_episode_ids, size=len(unique_episode_ids), replace=True)
        metrics = aggregate_episode_statistics(stats, sampled)
        for key in sampled_metrics:
            sampled_metrics[key].append(float(metrics[key]))
    for key, values in sampled_metrics.items():
        base[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
        base[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
    return base


def paired_bootstrap_incremental_gain(
    y_true,
    surface_probabilities,
    implicit_probabilities,
    episode_ids,
    classes,
    draws,
    seed,
):
    surface_stats = episode_prediction_statistics(y_true, surface_probabilities, episode_ids, classes)
    implicit_stats = episode_prediction_statistics(y_true, implicit_probabilities, episode_ids, classes)
    unique_episode_ids = sorted(set(surface_stats) & set(implicit_stats))

    def gains(sampled):
        surface = aggregate_episode_statistics(surface_stats, sampled)
        implicit = aggregate_episode_statistics(implicit_stats, sampled)
        return {
            "log_loss_improvement": float(surface["log_loss"] - implicit["log_loss"]),
            "accuracy_improvement": float(implicit["accuracy"] - surface["accuracy"]),
            "balanced_accuracy_improvement": float(implicit["balanced_accuracy"] - surface["balanced_accuracy"]),
            "macro_f1_improvement": float(implicit["macro_f1"] - surface["macro_f1"]),
        }

    base = gains(unique_episode_ids)
    rng = np.random.default_rng(int(seed))
    sampled_values = {key: [] for key in base}
    for _ in range(max(int(draws), 1)):
        sampled = rng.choice(unique_episode_ids, size=len(unique_episode_ids), replace=True)
        draw = gains(sampled)
        for key, value in draw.items():
            sampled_values[key].append(value)
    for key, values in sampled_values.items():
        base[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
        base[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
        base[f"{key}_probability_positive"] = float(np.mean(np.asarray(values) > 0.0))
    return base


def run_response_type_baselines(rows, output_dir, args):
    if args.disable_response_type_baselines:
        return {"status": "disabled"}
    eligible = [
        row
        for row in rows
        if row.get("episode_id") is not None and str(row.get("next_response_type")) in RESPONSE_CLASSES
    ]
    required = sorted(
        set().union(
            *(set(spec["numeric_features"] + spec["categorical_features"]) for spec in BASELINE_MODEL_SPECS)
        )
    )
    missing = [feature for feature in required if not any(feature in row for row in eligible)]
    if missing:
        return {
            "status": "skipped",
            "reason": "Turn-level feature files were generated by an older code version.",
            "missing_features": missing,
        }
    if len(eligible) < 20:
        return {"status": "skipped", "reason": "Too few eligible rows for held-out evaluation."}

    train_rows, test_rows, test_ids, realized_seed = split_rows_by_episode(
        eligible,
        args.baseline_test_fraction,
        args.baseline_split_seed,
    )
    split_rows = [
        {"episode_id": episode_id, "split": "test" if episode_id in test_ids else "train"}
        for episode_id in sorted({str(row["episode_id"]) for row in eligible})
    ]
    save_csv(output_dir / "exp5_response_type_episode_split.csv", split_rows, ["episode_id", "split"])

    y_train = [str(row["next_response_type"]) for row in train_rows]
    y_test = [str(row["next_response_type"]) for row in test_rows]
    test_episode_ids = [str(row["episode_id"]) for row in test_rows]
    comparison_rows, coefficient_rows, fitted = [], [], {}
    for spec in BASELINE_MODEL_SPECS:
        preprocessor = fit_baseline_preprocessor(train_rows, spec)
        x_train = apply_baseline_preprocessor(train_rows, preprocessor)
        x_test = apply_baseline_preprocessor(test_rows, preprocessor)
        model = fit_multinomial_map_matrix(
            x_train,
            y_train,
            preprocessor["feature_names"],
            prior_sd=args.bayes_multinomial_prior_sd,
        )
        if model is None:
            comparison_rows.append(
                {
                    "model_id": spec["model_id"],
                    "description": spec["description"],
                    "status": "fit_failed",
                    "n_train_rows": len(train_rows),
                    "n_test_rows": len(test_rows),
                    "num_features": len(preprocessor["feature_names"]),
                }
            )
            continue
        probabilities = predict_multinomial_map_matrix(model, x_test)
        metrics = bootstrap_metric_intervals(
            y_test,
            probabilities,
            test_episode_ids,
            model["classes"],
            args.baseline_bootstrap_draws,
            args.baseline_split_seed + 1000,
        )
        comparison_rows.append(
            {
                "model_id": spec["model_id"],
                "description": spec["description"],
                "status": "ok",
                "n_train_rows": len(train_rows),
                "n_test_rows": len(test_rows),
                "num_features": len(preprocessor["feature_names"]),
                **metrics,
            }
        )
        fitted[spec["model_id"]] = {
            "model": model,
            "probabilities": probabilities,
            "preprocessor": preprocessor,
            "spec": spec,
        }
        for class_idx, class_name in enumerate(model["classes"]):
            terms = ["Intercept"] + list(model["feature_names"])
            for term_idx, term_name in enumerate(terms):
                coefficient_rows.append(
                    {
                        "model_id": spec["model_id"],
                        "class": class_name,
                        "reference_class": model["reference_class"],
                        "term": term_name,
                        "map_coefficient": float(model["W_map"][term_idx, class_idx]),
                    }
                )

    comparison_fields = [
        "model_id",
        "description",
        "status",
        "n_train_rows",
        "n_test_rows",
        "num_features",
        "log_loss",
        "log_loss_ci_low",
        "log_loss_ci_high",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "macro_f1",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
    ]
    save_csv(output_dir / "exp5_response_type_baseline_comparison.csv", comparison_rows, comparison_fields)
    if coefficient_rows:
        save_csv(
            output_dir / "exp5_response_type_baseline_coefficients.csv",
            coefficient_rows,
            ["model_id", "class", "reference_class", "term", "map_coefficient"],
        )

    full_surface_id = "surface_plus_similarity"
    gain_rows = []
    if full_surface_id in fitted:
        for implicit_id in ["surface_plus_implicit_density", "surface_plus_implicit_load"]:
            if implicit_id not in fitted:
                continue
            surface = fitted[full_surface_id]
            implicit = fitted[implicit_id]
            if surface["model"]["classes"] != implicit["model"]["classes"]:
                continue
            gains = paired_bootstrap_incremental_gain(
                y_test,
                surface["probabilities"],
                implicit["probabilities"],
                test_episode_ids,
                surface["model"]["classes"],
                args.baseline_bootstrap_draws,
                args.baseline_split_seed + 2000,
            )
            gain_rows.append(
                {
                    "surface_model_id": full_surface_id,
                    "implicit_model_id": implicit_id,
                    **gains,
                }
            )
    if gain_rows:
        save_csv(
            output_dir / "exp5_response_type_baseline_incremental_gains.csv",
            gain_rows,
            list(gain_rows[0].keys()),
        )

    return {
        "status": "ok",
        "split": {
            "requested_test_fraction": float(args.baseline_test_fraction),
            "realized_test_fraction_by_episode": float(len(test_ids) / len(split_rows)),
            "split_seed": int(realized_seed),
            "num_train_episodes": int(len(split_rows) - len(test_ids)),
            "num_test_episodes": int(len(test_ids)),
            "num_train_rows": int(len(train_rows)),
            "num_test_rows": int(len(test_rows)),
        },
        "bootstrap_draws": int(args.baseline_bootstrap_draws),
        "full_surface_model_id": full_surface_id,
        "model_comparison": comparison_rows,
        "incremental_gains": gain_rows,
        "interpretation": (
            "Positive log_loss_improvement means the implicit model has lower held-out log loss. "
            "Positive accuracy, balanced-accuracy, and macro-F1 improvements favor the implicit model."
        ),
    }


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
        "response_type_baseline_config": {
            "enabled": not args.disable_response_type_baselines,
            "test_fraction": args.baseline_test_fraction,
            "split_seed": args.baseline_split_seed,
            "episode_bootstrap_draws": args.baseline_bootstrap_draws,
            "split_unit": "episode_id",
            "models": BASELINE_MODEL_SPECS,
        },
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
                "exp5_response_type_episode_split.csv",
                "exp5_response_type_baseline_comparison.csv",
                "exp5_response_type_baseline_coefficients.csv",
                "exp5_response_type_baseline_incremental_gains.csv",
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
    ap.add_argument("--baseline_test_fraction", type=float, default=0.20)
    ap.add_argument("--baseline_split_seed", type=int, default=RNG_SEED)
    ap.add_argument("--baseline_bootstrap_draws", type=int, default=500)
    ap.add_argument("--disable_response_type_baselines", action="store_true")
    ap.add_argument("--no_tqdm", action="store_true")
    return ap


def analyze_rows(rows, output_dir, input_dir, num_episodes, silence_gap, args, show_progress, summary_extra=None):
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

    write_turn_level_feature_csv(rows, output_dir)
    if curve_rows:
        save_csv(output_dir / "exp5_probability_curves.csv", curve_rows, ["implicature_load"] + [k for k in curve_rows[0] if k != "implicature_load"])
    if coef_rows:
        save_csv(output_dir / "exp5_logit_coefficients.csv", coef_rows, ["class", "reference_class", "posterior_mean_intercept", "posterior_mean_coef_scaled_load", "posterior_sd_intercept", "posterior_sd_coef_scaled_load", "map_intercept", "map_coef_scaled_load", "scaler_mean", "scaler_std"])
    if response_delay_regression_coeffs:
        save_csv(output_dir / "exp5_response_delay_regression_coefficients.csv", response_delay_regression_coeffs, ["term", "posterior_mean", "posterior_sd"])
    if response_delay_regression is not None:
        (output_dir / "exp5_response_delay_regression_summary.json").write_text(json.dumps({"regression": response_delay_regression, "distribution_checks": response_delay_distribution_checks}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    response_counts = Counter(str(r.get("next_response_type")) for r in rows if r.get("next_response_type"))
    save_csv(output_dir / "exp5_response_type_counts.csv", [{"response_type": k, "count": v} for k, v in sorted(response_counts.items())], ["response_type", "count"])
    response_type_baselines = run_response_type_baselines(rows, output_dir, args)

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
            "analysis_stage": "full_analysis",
            "assumption_count_vs_response_time_correlation": corr_stats(assumption_x, assumption_y),
            "implicature_load_vs_response_time_correlation": corr_stats(load_x, load_y),
            "response_type_model": None if not model else {"model_family": "bayesian_multinomial_logit_laplace", "prior_sd": model["prior_sd"], "posterior_draws": model["posterior_draws"], "reference_class": model["reference_class"], "approximation": model["approximation"]},
            "response_type_baselines": response_type_baselines,
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
    if not 0.0 < args.baseline_test_fraction < 1.0:
        raise ValueError("baseline_test_fraction must be strictly between 0 and 1.")
    if args.baseline_bootstrap_draws < 1:
        raise ValueError("baseline_bootstrap_draws must be >= 1.")
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