import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PAIRS_DIR = "data/implicature_flow/entailment_pairs_1to10"
DEFAULT_TURNS_DIR = "data/stance_labeled"
DEFAULT_OUTPUT_DIR = "experiments/exp4_implicature_flow/results"


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_turns(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        return obj["turns"]
    raise ValueError("Expected a list of turns or a dict containing a 'turns' list")


def extract_turn_text(turn: Dict[str, Any]) -> str:
    key = turn.get("chosen_text_key")
    if isinstance(key, str) and isinstance(turn.get(key), str):
        return turn[key]
    for candidate in ("transcript", "turn_text", "text"):
        if isinstance(turn.get(candidate), str):
            return turn[candidate]
    return ""


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or "", flags=re.UNICODE))


def turn_duration_seconds(turn: Dict[str, Any]) -> float:
    duration = safe_float(turn.get("duration"))
    if math.isfinite(duration) and duration >= 0.0:
        return duration

    start = safe_float(turn.get("startTime", turn.get("start_time")))
    end = safe_float(turn.get("endTime", turn.get("end_time")))
    if math.isfinite(start) and math.isfinite(end) and end >= start:
        return end - start
    return float("nan")


def speaker_ids_differ(pair: Dict[str, Any]) -> Optional[bool]:
    a_id = pair.get("a_turn_speaker_id")
    c_id = pair.get("c_turn_speaker_id")
    if a_id is not None and c_id is not None:
        return str(a_id) != str(c_id)

    if "same_speaker" in pair and pair.get("same_speaker") is not None:
        return not bool(pair.get("same_speaker"))

    a_id = pair.get("a_speaker_id")
    c_id = pair.get("c_speaker_id")
    if a_id is not None and c_id is not None:
        return str(a_id) != str(c_id)
    return None


def build_turn_file_index(turns_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    for path in turns_dir.rglob("*.json"):
        duplicates[path.name].append(path)

    for name, paths in duplicates.items():
        # The entailment pipeline expects episode filenames to be unique across shards.
        # Prefer the lexicographically first file if identical names are duplicated.
        index[name] = sorted(paths)[0]
    return index


def choose_earliest_match(matches: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    matches = list(matches)
    if not matches:
        return None

    def key(pair: Dict[str, Any]) -> Tuple[float, int]:
        c_time = safe_float(pair.get("c_time"), float("inf"))
        c_turn = safe_int(pair.get("c_turn_idx"), 10**12)
        return c_time, c_turn if c_turn is not None else 10**12

    return min(matches, key=key)


def collect_assumption_rows(
    pairs_dir: Path,
    turns_dir: Path,
    threshold: int,
) -> pd.DataFrame:
    turn_index = build_turn_file_index(turns_dir)
    rows: List[Dict[str, Any]] = []
    missing_turn_files = 0
    bad_turn_indices = 0

    pair_files = sorted(path for path in pairs_dir.glob("*.json") if not path.name.startswith("_"))
    if not pair_files:
        raise FileNotFoundError(f"No entailment JSON files found in {pairs_dir}")

    for pair_path in pair_files:
        obj = load_json(pair_path)
        episode_id = str(obj.get("episode_id", pair_path.stem)).strip()
        category = obj.get("category")
        pairs = obj.get("pairs", []) or []
        if not isinstance(pairs, list) or not pairs:
            continue

        raw_path = turn_index.get(f"{episode_id}.json")
        turns: Optional[List[Dict[str, Any]]] = None
        if raw_path is not None:
            try:
                turns = extract_turns(load_json(raw_path))
            except Exception:
                turns = None
        else:
            missing_turn_files += 1

        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            a_turn = safe_int(pair.get("a_turn_idx"))
            a_idx = safe_int(pair.get("a_idx_in_turn"))
            if a_turn is None or a_idx is None:
                continue
            grouped[(a_turn, a_idx)].append(pair)

        for (a_turn, a_idx), candidate_pairs in grouped.items():
            positive = [p for p in candidate_pairs if safe_float(p.get("entailment_score"), 0.0) >= threshold]
            cross_positive = [p for p in positive if speaker_ids_differ(p) is True]
            same_positive = [p for p in positive if speaker_ids_differ(p) is False]

            earliest = choose_earliest_match(positive)
            earliest_cross = choose_earliest_match(cross_positive)

            source_turn: Dict[str, Any] = {}
            if turns is not None and 0 <= a_turn < len(turns):
                source_turn = turns[a_turn]
            elif turns is not None:
                bad_turn_indices += 1

            text = extract_turn_text(source_turn) if source_turn else ""
            explicit = source_turn.get("explicit_propositions", []) if source_turn else []
            assumptions = source_turn.get("assumptions", []) if source_turn else []

            overlaps = [safe_float(p.get("overlap_score"), 0.0) for p in candidate_pairs]
            turn_gaps = []
            for p in candidate_pairs:
                c_turn = safe_int(p.get("c_turn_idx"))
                if c_turn is not None:
                    turn_gaps.append(c_turn - a_turn)

            first_pair = candidate_pairs[0]
            assumption_text = str(first_pair.get("assumption_text", ""))

            rows.append({
                "episode_id": episode_id,
                "category": category,
                "a_turn_idx": a_turn,
                "a_idx_in_turn": a_idx,
                "assumption_text": assumption_text,
                "assumption_word_count": word_count(assumption_text),
                "candidate_count": len(candidate_pairs),
                "max_overlap": max(overlaps) if overlaps else 0.0,
                "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
                "min_turn_gap": min(turn_gaps) if turn_gaps else float("nan"),
                "accommodated": int(bool(positive)),
                "cross_speaker_accommodated": int(bool(cross_positive)),
                "same_speaker_accommodated": int(bool(same_positive)),
                "earliest_match_cross_speaker": (
                    int(speaker_ids_differ(earliest) is True) if earliest is not None and speaker_ids_differ(earliest) is not None else np.nan
                ),
                "earliest_match_turn_gap": (
                    safe_int(earliest.get("c_turn_idx")) - a_turn if earliest is not None and safe_int(earliest.get("c_turn_idx")) is not None else np.nan
                ),
                "earliest_cross_turn_gap": (
                    safe_int(earliest_cross.get("c_turn_idx")) - a_turn if earliest_cross is not None and safe_int(earliest_cross.get("c_turn_idx")) is not None else np.nan
                ),
                "source_word_count": word_count(text) if source_turn else np.nan,
                "duration_sec": turn_duration_seconds(source_turn) if source_turn else np.nan,
                "explicit_count": len(explicit) if isinstance(explicit, list) else np.nan,
                "source_assumption_count": len(assumptions) if isinstance(assumptions, list) else np.nan,
            })

    print(f"Built {len(rows):,} assumption-level rows")
    if missing_turn_files:
        print(f"Warning: {missing_turn_files:,} pair files had no matching stance-labeled turn file")
    if bad_turn_indices:
        print(f"Warning: {bad_turn_indices:,} assumptions had source turn indices outside the loaded turn list")

    return pd.DataFrame(rows)


def conversion_rate(series: pd.Series) -> float:
    return float(series.astype(float).mean() * 100.0) if len(series) else float("nan")


def build_speaker_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    accommodated_df = df[df["accommodated"] == 1]
    earliest_known = accommodated_df["earliest_match_cross_speaker"].dropna()

    rows = [
        {
            "analysis": "all_future_matches",
            "denominator": total,
            "matched": int(df["accommodated"].sum()),
            "conversion_rate_percent": conversion_rate(df["accommodated"]),
        },
        {
            "analysis": "other_speaker_only",
            "denominator": total,
            "matched": int(df["cross_speaker_accommodated"].sum()),
            "conversion_rate_percent": conversion_rate(df["cross_speaker_accommodated"]),
        },
        {
            "analysis": "same_speaker_only",
            "denominator": total,
            "matched": int(df["same_speaker_accommodated"].sum()),
            "conversion_rate_percent": conversion_rate(df["same_speaker_accommodated"]),
        },
        {
            "analysis": "earliest_match_is_other_speaker_among_known_accommodated",
            "denominator": int(len(earliest_known)),
            "matched": int(earliest_known.sum()),
            "conversion_rate_percent": conversion_rate(earliest_known),
        },
        {
            "analysis": "ever_other_speaker_among_accommodated",
            "denominator": int(len(accommodated_df)),
            "matched": int(accommodated_df["cross_speaker_accommodated"].sum()),
            "conversion_rate_percent": conversion_rate(accommodated_df["cross_speaker_accommodated"]),
        },
    ]
    return pd.DataFrame(rows)


def compute_threshold_sensitivity(pairs_dir: Path, thresholds: List[int]) -> pd.DataFrame:
    assumptions_by_threshold = {threshold: [0, 0] for threshold in thresholds}

    pair_files = sorted(path for path in pairs_dir.glob("*.json") if not path.name.startswith("_"))
    for pair_path in pair_files:
        obj = load_json(pair_path)
        grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for pair in obj.get("pairs", []) or []:
            if not isinstance(pair, dict):
                continue
            a_turn = safe_int(pair.get("a_turn_idx"))
            a_idx = safe_int(pair.get("a_idx_in_turn"))
            if a_turn is None or a_idx is None:
                continue
            grouped[(a_turn, a_idx)].append(safe_float(pair.get("entailment_score"), 0.0))

        for scores in grouped.values():
            max_score = max(scores) if scores else 0.0
            for threshold in thresholds:
                assumptions_by_threshold[threshold][0] += 1
                assumptions_by_threshold[threshold][1] += int(max_score >= threshold)

    rows = []
    for threshold in thresholds:
        total, matched = assumptions_by_threshold[threshold]
        rows.append({
            "entailment_threshold": threshold,
            "total_assumptions": total,
            "accommodated": matched,
            "conversion_rate_percent": (100.0 * matched / total) if total else float("nan"),
        })
    return pd.DataFrame(rows)


def build_verbosity_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    usable = df.copy()
    usable = usable[np.isfinite(pd.to_numeric(usable["source_word_count"], errors="coerce"))]
    if usable.empty:
        return pd.DataFrame()

    usable["word_count_quartile"] = pd.qcut(
        usable["source_word_count"].rank(method="first"),
        q=4,
        labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"],
    )
    out = (
        usable.groupby("word_count_quartile", observed=True)
        .agg(
            assumptions=("accommodated", "size"),
            accommodation_rate_percent=("accommodated", lambda x: float(np.mean(x) * 100.0)),
            cross_speaker_rate_percent=("cross_speaker_accommodated", lambda x: float(np.mean(x) * 100.0)),
            median_source_words=("source_word_count", "median"),
            median_duration_sec=("duration_sec", "median"),
            median_candidate_count=("candidate_count", "median"),
        )
        .reset_index()
    )
    return out


def fit_surface_baseline(df: pd.DataFrame, seed: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    features = [
        "source_word_count",
        "duration_sec",
        "explicit_count",
        "source_assumption_count",
        "candidate_count",
        "max_overlap",
        "assumption_word_count",
        "min_turn_gap",
    ]
    model_df = df[["episode_id", "accommodated", *features]].copy()
    for col in features:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=["episode_id", "accommodated"])

    episodes = np.array(sorted(model_df["episode_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    split = max(1, int(round(len(episodes) * 0.8)))
    train_eps = set(episodes[:split])
    train_mask = model_df["episode_id"].astype(str).isin(train_eps)

    train_df = model_df[train_mask]
    test_df = model_df[~train_mask]
    if train_df.empty or test_df.empty:
        raise ValueError("Need at least two episodes for episode-held-out baseline evaluation")

    x_train = train_df[features]
    y_train = train_df["accommodated"].astype(int).to_numpy()
    x_test = test_df[features]
    y_test = test_df["accommodated"].astype(int).to_numpy()

    preprocess = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            features,
        )
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("logit", LogisticRegression(max_iter=1000, class_weight=None, random_state=seed)),
    ])
    pipeline.fit(x_train, y_train)
    p_test = pipeline.predict_proba(x_test)[:, 1]

    auroc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else float("nan")
    auprc = average_precision_score(y_test, p_test)
    prevalence = float(np.mean(y_test))

    coef = pipeline.named_steps["logit"].coef_[0]
    coef_df = pd.DataFrame({"feature": features, "standardized_logit_coefficient": coef})
    coef_df["abs_coefficient"] = coef_df["standardized_logit_coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")

    result = {
        "task": "predict_existing_RQ2_accommodation_from_surface_and_search_opportunity_features",
        "split": "80/20 held out by episode",
        "random_seed": seed,
        "train_episodes": len(train_eps),
        "test_episodes": len(set(episodes[split:])),
        "train_assumptions": int(len(train_df)),
        "test_assumptions": int(len(test_df)),
        "test_prevalence": prevalence,
        "majority_accuracy": max(prevalence, 1.0 - prevalence),
        "surface_baseline_auroc": float(auroc),
        "surface_baseline_auprc": float(auprc),
        "no_skill_auprc": prevalence,
        "features": features,
        "interpretation": (
            "This is a diagnostic baseline for whether simple verbosity and candidate-opportunity signals "
            "can recover the existing RQ2 accommodation labels. It is not independent human ground truth."
        ),
    }
    return result, coef_df


def save_plots(
    speaker_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    verbosity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    speaker_plot = speaker_df[speaker_df["analysis"].isin([
        "all_future_matches",
        "other_speaker_only",
        "same_speaker_only",
    ])].copy()
    speaker_plot["label"] = speaker_plot["analysis"].map({
        "all_future_matches": "All matches",
        "other_speaker_only": "Other-speaker only",
        "same_speaker_only": "Same-speaker only",
    })
    plt.figure(figsize=(7.2, 4.6))
    plt.bar(speaker_plot["label"], speaker_plot["conversion_rate_percent"])
    plt.ylabel("Accommodation rate (%)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "exp4_speaker_control.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "exp4_speaker_control.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.8, 4.6))
    plt.plot(threshold_df["entailment_threshold"], threshold_df["conversion_rate_percent"], marker="o")
    plt.xlabel("Entailment threshold")
    plt.ylabel("Accommodation rate (%)")
    plt.xticks(threshold_df["entailment_threshold"].tolist())
    plt.tight_layout()
    plt.savefig(output_dir / "exp4_threshold_sensitivity.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "exp4_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()

    if not verbosity_df.empty:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(
            verbosity_df["word_count_quartile"].astype(str),
            verbosity_df["accommodation_rate_percent"],
            marker="o",
            label="All matches",
        )
        plt.plot(
            verbosity_df["word_count_quartile"].astype(str),
            verbosity_df["cross_speaker_rate_percent"],
            marker="o",
            label="Other-speaker only",
        )
        plt.xlabel("Source-turn word-count quartile")
        plt.ylabel("Accommodation rate (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "exp4_verbosity_control.pdf", bbox_inches="tight")
        plt.savefig(output_dir / "exp4_verbosity_control.png", dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast RQ2 controls using only existing entailment scores and stance-labeled turns; no LLM calls."
    )
    parser.add_argument("--pairs_dir", default=DEFAULT_PAIRS_DIR)
    parser.add_argument("--turns_dir", default=DEFAULT_TURNS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=int, default=7)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    turns_dir = Path(args.turns_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("RQ2 FAST CONTROLS")
    print("No LLM inference or relabeling will be performed.")

    df = collect_assumption_rows(pairs_dir, turns_dir, args.threshold)
    df.to_csv(output_dir / "exp4_fast_control_assumption_rows.csv", index=False)

    speaker_df = build_speaker_summary(df)
    speaker_df.to_csv(output_dir / "exp4_speaker_control_summary.csv", index=False)

    threshold_df = compute_threshold_sensitivity(pairs_dir, sorted(set(args.thresholds)))
    threshold_df.to_csv(output_dir / "exp4_threshold_sensitivity.csv", index=False)

    verbosity_df = build_verbosity_quartiles(df)
    verbosity_df.to_csv(output_dir / "exp4_verbosity_control.csv", index=False)

    baseline_result, coef_df = fit_surface_baseline(df, args.seed)
    coef_df.to_csv(output_dir / "exp4_surface_baseline_coefficients.csv", index=False)
    with (output_dir / "exp4_surface_baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_result, handle, indent=2, ensure_ascii=False)

    save_plots(speaker_df, threshold_df, verbosity_df, output_dir)

    summary = {
        "threshold": args.threshold,
        "n_assumptions": int(len(df)),
        "speaker_control": speaker_df.to_dict(orient="records"),
        "threshold_sensitivity": threshold_df.to_dict(orient="records"),
        "verbosity_quartiles": verbosity_df.to_dict(orient="records"),
        "surface_baseline": baseline_result,
        "method_note": (
            "All analyses reuse the existing entailment scores. The other-speaker restriction controls for self-repetition; "
            "word-count quartiles and the episode-held-out surface baseline diagnose verbosity/search-opportunity confounds; "
            "threshold sensitivity checks dependence on the score>=7 operational boundary."
        ),
    }
    with (output_dir / "exp4_fast_controls_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nSpeaker control")
    print(speaker_df.to_string(index=False))
    print("\nThreshold sensitivity")
    print(threshold_df.to_string(index=False))
    if not verbosity_df.empty:
        print("\nVerbosity quartiles")
        print(verbosity_df.to_string(index=False))
    print("\nSurface-only diagnostic baseline")
    print(json.dumps(baseline_result, indent=2))
    print(f"\nWrote results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
