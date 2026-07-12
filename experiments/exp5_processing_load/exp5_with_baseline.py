from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

RNG_SEED = 42
RESPONSE_CLASSES = ["Backchannel", "Substantive", "Clarification", "Silence/Abandonment"]

BASELINE_MODEL_SPECS = [
    {
        "model_id": "surface_basic",
        "description": "Turn duration and word count only.",
        "numeric": ["duration_sec", "word_count_in_turn"],
        "categorical": [],
    },
    {
        "model_id": "surface_plus_explicit",
        "description": "Basic surface features plus explicit-proposition count.",
        "numeric": ["duration_sec", "word_count_in_turn", "explicit_statement_count"],
        "categorical": [],
    },
    {
        "model_id": "surface_plus_history",
        "description": "Surface and explicit features plus local response/move history.",
        "numeric": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
        ],
        "categorical": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
    {
        "model_id": "surface_plus_similarity",
        "description": "Surface, explicit, and history features plus adjacent-turn embedding similarity.",
        "numeric": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
            "turn_similarity_to_previous",
        ],
        "categorical": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
    {
        "model_id": "surface_plus_implicit_density",
        "description": "Strongest surface baseline plus implicit-assumption count and density.",
        "numeric": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
            "turn_similarity_to_previous",
            "assumption_count_in_turn",
            "new_assumption_count",
            "new_assumptions_per_second",
        ],
        "categorical": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
    {
        "model_id": "surface_plus_implicit_load",
        "description": "Strongest surface baseline plus implicit-assumption counts and seconds per new assumption.",
        "numeric": [
            "duration_sec",
            "word_count_in_turn",
            "explicit_statement_count",
            "average_response_time_0_to_n_minus_1",
            "turn_similarity_to_previous",
            "assumption_count_in_turn",
            "new_assumption_count",
            "implicature_load",
        ],
        "categorical": [
            "previous_response_type",
            "current_turn_type_label",
            "current_conversation_move_label",
            "previous_conversation_move_label",
        ],
    },
]

LOG1P_FEATURES = {
    "duration_sec",
    "word_count_in_turn",
    "explicit_statement_count",
    "average_response_time_0_to_n_minus_1",
    "assumption_count_in_turn",
    "new_assumption_count",
    "new_assumptions_per_second",
    "implicature_load",
}


def model_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.replace("/", "__").strip())
    if not slug:
        raise ValueError("Model name must not be empty.")
    return slug


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def turn_idx(turn: dict[str, Any], fallback: int) -> int:
    value = turn.get("turn_idx", fallback)
    try:
        return int(value)
    except Exception:
        return fallback


def word_count(turn: dict[str, Any]) -> float:
    existing = to_float(turn.get("wordCount", turn.get("word_count")))
    if math.isfinite(existing):
        return existing
    text = str(turn.get("turn_text", "") or "")
    return float(len(text.split()))


def collect_episode_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*.json"))
    return paths if paths else sorted(input_dir.glob("*/*.json"))


def load_existing_exp5_rows(results_dir: Path) -> pd.DataFrame:
    patch_paths = sorted(results_dir.glob("patches/patch_*/exp5_turn_level_features.csv"))
    if patch_paths:
        print(f"Loading {len(patch_paths)} existing Exp5 patch CSVs.")
        frames = [pd.read_csv(path) for path in tqdm(patch_paths, desc="Loading old Exp5 patches")]
        df = pd.concat(frames, ignore_index=True)
    else:
        merged = results_dir / "exp5_turn_level_features.csv"
        if not merged.exists():
            raise FileNotFoundError(
                "Could not find existing Exp5 turn-level features. Expected either\n"
                f"  {results_dir}/patches/patch_*/exp5_turn_level_features.csv\n"
                f"or {merged}"
            )
        print(f"Loading existing merged Exp5 feature CSV: {merged}")
        df = pd.read_csv(merged)

    required = {
        "episode_id",
        "turn_idx",
        "duration_sec",
        "assumption_count_in_turn",
        "explicit_statement_count",
        "new_assumption_count",
        "implicature_load",
        "average_response_time_0_to_n_minus_1",
        "next_response_type",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Existing Exp5 CSVs are missing required old-result columns: {missing}")

    df["episode_id"] = df["episode_id"].astype(str)
    df["turn_idx"] = pd.to_numeric(df["turn_idx"], errors="coerce")
    df = df.dropna(subset=["turn_idx"]).copy()
    df["turn_idx"] = df["turn_idx"].astype(int)
    duplicate_mask = df.duplicated(["episode_id", "turn_idx"], keep=False)
    if duplicate_mask.any():
        print(
            f"Warning: dropping {int(duplicate_mask.sum())} duplicate episode/turn rows.",
            file=sys.stderr,
        )
        df = df.drop_duplicates(["episode_id", "turn_idx"], keep="first")
    return df.sort_values(["episode_id", "turn_idx"]).reset_index(drop=True)


def load_turn_metadata(input_dir: Path, wanted_keys: set[tuple[str, int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = collect_episode_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No JSON episode files found under {input_dir}")

    for path in tqdm(paths, desc="Joining labeled turn metadata", unit="episode"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, list) or not payload:
            continue
        turns = sorted(payload, key=lambda t: turn_idx(t, 0))
        episode_id = str(turns[0].get("episode_id", path.stem))
        previous_move = None
        for fallback_idx, turn in enumerate(turns):
            idx = turn_idx(turn, fallback_idx)
            key = (episode_id, idx)
            if key in wanted_keys:
                rows.append(
                    {
                        "episode_id": episode_id,
                        "turn_idx": idx,
                        "word_count_in_turn": word_count(turn),
                        "turn_text": str(turn.get("turn_text", "") or ""),
                        "current_turn_type_label": turn.get("turn_type_label"),
                        "current_conversation_move_label": turn.get("conversation_move_label"),
                        "previous_conversation_move_label": previous_move,
                    }
                )
            previous_move = turn.get("conversation_move_label")

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise RuntimeError(
            "No old Exp5 rows matched the labeled-turn JSONs. Check that --input_dir and --results_dir "
            "refer to the same dataset/version."
        )
    metadata["episode_id"] = metadata["episode_id"].astype(str)
    metadata["turn_idx"] = metadata["turn_idx"].astype(int)
    return metadata.drop_duplicates(["episode_id", "turn_idx"], keep="first")


def load_embedder(name: str):
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError("transformers is required only when building the feature cache with turn embeddings.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device)
    model.eval()
    print(f"Embedding adjacent turns with {name} on {device}.")
    return tokenizer, model, device


def embed_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        output = model(**batch)
        hidden = output.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.shape).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype(np.float32, copy=False)


def compute_adjacent_similarity(
    df: pd.DataFrame,
    embedding_model_name: str,
    batch_size: int,
) -> np.ndarray:
    tokenizer, model, device = load_embedder(embedding_model_name)
    similarities = np.full(len(df), np.nan, dtype=float)
    previous_vector: np.ndarray | None = None
    previous_episode: str | None = None

    texts = df["turn_text"].fillna("").astype(str).tolist()
    episode_ids = df["episode_id"].astype(str).tolist()
    for start in tqdm(range(0, len(df), batch_size), desc="Embedding turn text"):
        end = min(start + batch_size, len(df))
        batch_texts = texts[start:end]
        nonempty_positions = [j for j, text in enumerate(batch_texts) if text.strip()]
        vectors_by_position: dict[int, np.ndarray] = {}
        if nonempty_positions:
            vectors = embed_batch(
                [batch_texts[j] for j in nonempty_positions],
                tokenizer,
                model,
                device,
            )
            vectors_by_position = {j: vector for j, vector in zip(nonempty_positions, vectors)}

        for local_idx in range(end - start):
            global_idx = start + local_idx
            episode_id = episode_ids[global_idx]
            vector = vectors_by_position.get(local_idx)
            if (
                vector is not None
                and previous_vector is not None
                and previous_episode == episode_id
            ):
                similarities[global_idx] = float(np.dot(previous_vector, vector))
            previous_vector = vector
            previous_episode = episode_id

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return similarities


def build_or_load_feature_cache(args: argparse.Namespace, results_dir: Path) -> pd.DataFrame:
    cache_path = results_dir / args.feature_cache_name
    if cache_path.exists() and not args.rebuild_feature_cache:
        print(f"Reusing cached baseline features: {cache_path}")
        return pd.read_csv(cache_path)

    old = load_existing_exp5_rows(results_dir)
    wanted_keys = set(zip(old["episode_id"].astype(str), old["turn_idx"].astype(int)))
    metadata = load_turn_metadata(Path(args.input_dir), wanted_keys)
    df = old.merge(metadata, on=["episode_id", "turn_idx"], how="left", validate="one_to_one")

    unmatched = int(df["turn_text"].isna().sum())
    if unmatched:
        print(f"Warning: {unmatched} rows did not match raw turn metadata.", file=sys.stderr)

    # The previous row's target is the current turn's response type. This is valid history,
    # unlike the current row's gap_to_next_sec/response_delay_at_time_n, which help define y.
    df = df.sort_values(["episode_id", "turn_idx"]).reset_index(drop=True)
    df["previous_response_type"] = (
        df.groupby("episode_id", sort=False)["next_response_type"].shift(1).fillna("EpisodeStart")
    )
    duration = pd.to_numeric(df["duration_sec"], errors="coerce")
    new_count = pd.to_numeric(df["new_assumption_count"], errors="coerce")
    df["new_assumptions_per_second"] = np.where(
        duration > 0,
        new_count.clip(lower=0) / duration,
        np.nan,
    )
    df["turn_similarity_to_previous"] = compute_adjacent_similarity(
        df,
        args.similarity_model_name,
        args.embedding_batch_size,
    )

    # Do not cache raw transcript text in the modeling table.
    df = df.drop(columns=["turn_text"])
    df.to_csv(cache_path, index=False)
    print(f"Wrote enriched baseline feature cache: {cache_path}")
    return df


def transform_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].copy()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
        if feature in LOG1P_FEATURES:
            values = out[feature].to_numpy(dtype=float)
            out[feature] = np.where(np.isfinite(values) & (values >= 0), np.log1p(values), np.nan)
    return out


def build_model_frame(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    numeric = transform_numeric_frame(df, spec["numeric"])
    if not spec["categorical"]:
        return numeric
    categorical = df[spec["categorical"]].copy()
    for feature in spec["categorical"]:
        categorical[feature] = categorical[feature].fillna("Missing").astype(str)
    return pd.concat([numeric, categorical], axis=1)


def make_pipeline(spec: dict[str, Any], seed: int) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", numeric_pipe, spec["numeric"])]
    if spec["categorical"]:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipe, spec["categorical"]))

    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )


def split_by_episode(df: pd.DataFrame, test_fraction: float, seed: int):
    episode_ids = np.asarray(sorted(df["episode_id"].astype(str).unique()), dtype=object)
    if len(episode_ids) < 2:
        raise ValueError("Need at least two episodes for an episode-held-out split.")
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    n_test = min(max(1, int(round(len(shuffled) * test_fraction))), len(shuffled) - 1)
    test_ids = set(str(value) for value in shuffled[:n_test])
    test_mask = df["episode_id"].astype(str).isin(test_ids)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy(), test_ids


def confusion_metrics(confusion: np.ndarray, logloss_sum: float, n_rows: int) -> dict[str, float]:
    total = float(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else float("nan")
    recalls = []
    f1s = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - tp)
        fp = float(confusion[:, idx].sum() - tp)
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "log_loss": float(logloss_sum / n_rows) if n_rows else float("nan"),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "accuracy": accuracy,
    }


def per_episode_stats(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    episode_ids: np.ndarray,
    classes: list[str],
) -> dict[str, tuple[np.ndarray, float, int]]:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_indices = np.asarray([class_to_idx[label] for label in y_true], dtype=int)
    clipped = np.clip(probabilities, 1e-15, 1.0)
    predicted = np.argmax(clipped, axis=1)
    stats: dict[str, tuple[np.ndarray, float, int]] = {}
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        confusion = confusion_matrix(
            y_indices[mask],
            predicted[mask],
            labels=np.arange(len(classes)),
        ).astype(float)
        ll_sum = float(-np.log(clipped[mask, y_indices[mask]]).sum())
        stats[str(episode_id)] = (confusion, ll_sum, int(mask.sum()))
    return stats


def aggregate_episode_stats(
    stats: dict[str, tuple[np.ndarray, float, int]],
    sampled_ids: Iterable[str],
) -> dict[str, float]:
    sampled = list(sampled_ids)
    confusion = np.zeros_like(next(iter(stats.values()))[0], dtype=float)
    logloss_sum = 0.0
    n_rows = 0
    for episode_id in sampled:
        episode_confusion, episode_logloss, episode_n = stats[str(episode_id)]
        confusion += episode_confusion
        logloss_sum += episode_logloss
        n_rows += episode_n
    return confusion_metrics(confusion, logloss_sum, n_rows)


def bootstrap_intervals(
    stats: dict[str, tuple[np.ndarray, float, int]],
    draws: int,
    seed: int,
) -> dict[str, float]:
    episode_ids = np.asarray(sorted(stats), dtype=object)
    base = aggregate_episode_stats(stats, episode_ids)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in base}
    for _ in range(max(draws, 1)):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        metrics = aggregate_episode_stats(stats, sampled)
        for metric, value in metrics.items():
            samples[metric].append(value)
    result: dict[str, float] = {}
    for metric, value in base.items():
        values = np.asarray(samples[metric], dtype=float)
        result[metric] = value
        result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    return result


def paired_incremental_gain(
    surface_stats: dict[str, tuple[np.ndarray, float, int]],
    implicit_stats: dict[str, tuple[np.ndarray, float, int]],
    draws: int,
    seed: int,
) -> dict[str, float]:
    episode_ids = np.asarray(sorted(set(surface_stats) & set(implicit_stats)), dtype=object)
    if not len(episode_ids):
        raise ValueError("No shared test episodes for paired bootstrap.")

    def gain(sampled_ids: Iterable[str]) -> dict[str, float]:
        surface = aggregate_episode_stats(surface_stats, sampled_ids)
        implicit = aggregate_episode_stats(implicit_stats, sampled_ids)
        return {
            "log_loss_improvement": surface["log_loss"] - implicit["log_loss"],
            "macro_f1_improvement": implicit["macro_f1"] - surface["macro_f1"],
            "balanced_accuracy_improvement": (
                implicit["balanced_accuracy"] - surface["balanced_accuracy"]
            ),
            "accuracy_improvement": implicit["accuracy"] - surface["accuracy"],
        }

    base = gain(episode_ids)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in base}
    for _ in range(max(draws, 1)):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        draw = gain(sampled)
        for metric, value in draw.items():
            samples[metric].append(value)

    result: dict[str, float] = dict(base)
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        result[f"{metric}_ci_low"] = float(np.quantile(array, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(array, 0.975))
        result[f"{metric}_probability_positive"] = float(np.mean(array > 0.0))
    return result


def run_baselines(df: pd.DataFrame, results_dir: Path, args: argparse.Namespace) -> None:
    df = df[df["next_response_type"].astype(str).isin(RESPONSE_CLASSES)].copy()
    df["episode_id"] = df["episode_id"].astype(str)
    if len(df) < 20:
        raise ValueError("Too few eligible RQ3 rows after filtering.")

    train_df, test_df, test_ids = split_by_episode(df, args.test_fraction, args.seed)
    split_df = pd.DataFrame(
        {
            "episode_id": sorted(df["episode_id"].unique()),
        }
    )
    split_df["split"] = np.where(split_df["episode_id"].isin(test_ids), "test", "train")
    split_df.to_csv(results_dir / "exp5_response_type_episode_split.csv", index=False)

    y_train = train_df["next_response_type"].astype(str).to_numpy()
    y_test = test_df["next_response_type"].astype(str).to_numpy()
    test_episode_ids = test_df["episode_id"].astype(str).to_numpy()

    comparison_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fitted_stats: dict[str, dict[str, tuple[np.ndarray, float, int]]] = {}

    for model_number, spec in enumerate(BASELINE_MODEL_SPECS):
        print(f"\nFitting {spec['model_id']} ...")
        train_x = build_model_frame(train_df, spec)
        test_x = build_model_frame(test_df, spec)
        pipeline = make_pipeline(spec, args.seed)
        pipeline.fit(train_x, y_train)
        probabilities = pipeline.predict_proba(test_x)
        classes = [str(label) for label in pipeline.named_steps["classifier"].classes_]
        stats = per_episode_stats(y_test, probabilities, test_episode_ids, classes)
        metrics = bootstrap_intervals(stats, args.bootstrap_draws, args.seed + 1000 + model_number)
        fitted_stats[spec["model_id"]] = stats

        predictions = pipeline.predict(test_x)
        comparison_rows.append(
            {
                "model_id": spec["model_id"],
                "description": spec["description"],
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
                "n_train_episodes": int(train_df["episode_id"].nunique()),
                "n_test_episodes": int(test_df["episode_id"].nunique()),
                "num_transformed_features": int(
                    len(pipeline.named_steps["preprocess"].get_feature_names_out())
                ),
                **metrics,
                # Direct sklearn values are included as a sanity check.
                "sklearn_log_loss_check": float(log_loss(y_test, probabilities, labels=classes)),
                "sklearn_macro_f1_check": float(
                    confusion_metrics(
                        confusion_matrix(y_test, predictions, labels=classes),
                        0.0,
                        len(y_test),
                    )["macro_f1"]
                ),
                "sklearn_balanced_accuracy_check": float(balanced_accuracy_score(y_test, predictions)),
                "sklearn_accuracy_check": float(accuracy_score(y_test, predictions)),
            }
        )

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        classifier = pipeline.named_steps["classifier"]
        for class_idx, class_name in enumerate(classes):
            for feature_name, coefficient in zip(feature_names, classifier.coef_[class_idx]):
                coefficient_rows.append(
                    {
                        "model_id": spec["model_id"],
                        "class": class_name,
                        "feature": str(feature_name),
                        "coefficient": float(coefficient),
                    }
                )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(results_dir / "exp5_response_type_baseline_comparison.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(
        results_dir / "exp5_response_type_baseline_coefficients.csv",
        index=False,
    )

    surface_id = "surface_plus_similarity"
    gain_rows = []
    for implicit_id in ["surface_plus_implicit_density", "surface_plus_implicit_load"]:
        gains = paired_incremental_gain(
            fitted_stats[surface_id],
            fitted_stats[implicit_id],
            args.bootstrap_draws,
            args.seed + 5000 + len(gain_rows),
        )
        gain_rows.append(
            {
                "surface_model_id": surface_id,
                "implicit_model_id": implicit_id,
                **gains,
            }
        )
    gains_df = pd.DataFrame(gain_rows)
    gains_df.to_csv(results_dir / "exp5_response_type_baseline_incremental_gains.csv", index=False)

    summary = {
        "feature_source": "existing Exp5 patch outputs; old assumption-newness features were not recomputed",
        "similarity_model_name": args.similarity_model_name,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "num_rows": int(len(df)),
        "num_episodes": int(df["episode_id"].nunique()),
        "models": comparison_rows,
        "incremental_gains": gain_rows,
    }
    (results_dir / "exp5_response_type_baseline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Held-out baseline comparison ===")
    display_columns = [
        "model_id",
        "log_loss",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
    ]
    print(comparison[display_columns].to_string(index=False))
    print("\n=== Paired gains over strongest surface baseline ===")
    print(gains_df.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RQ3 nested baselines from existing Exp5 feature patches without recomputing Exp5."
    )
    parser.add_argument("--input_dir", default="data/conversation_moves_labeled")
    parser.add_argument(
        "--results_dir",
        default="experiments/exp5_processing_load/results/Qwen__Qwen3-Embedding-4B",
        help="Existing Exp5 model result directory containing patch outputs.",
    )
    parser.add_argument(
        "--similarity_model_name",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model used only for adjacent-turn cosine similarity.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--feature_cache_name", default="exp5_baseline_features.csv")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--bootstrap_draws", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test_fraction must be between 0 and 1.")
    if args.embedding_batch_size < 1:
        raise ValueError("--embedding_batch_size must be >= 1.")
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap_draws must be >= 1.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    features = build_or_load_feature_cache(args, results_dir)
    run_baselines(features, results_dir, args)


if __name__ == "__main__":
    main()