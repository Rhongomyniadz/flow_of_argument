import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = "data/maxim_violations_labeled"
DEFAULT_OUTPUT_DIR = "experiments/exp7_social_power/results"
ANNOTATED_VIOLATION_TRIGGER = "annotated_local_context"
ANNOTATED_VIOLATION_SOURCE = "maxim_violation_label"

ROLE_ORDER = ("guest", "host")
VIOLATION_ORDER = ("Manner", "Relation", "Quantity")
VIOLATION_SEVERITY = {
    "Manner": 1,
    "Relation": 2,
    "Quantity": 3,
}

REPAIR_MOVES = {
    "Clarification Request (Generic)",
    "Clarification Request (Specific)",
}
CHALLENGE_MOVES = {
    "Correction / Challenge",
}
SELF_REPAIR_MOVES = {
    "Self-Correction",
}
POLICING_MOVES = REPAIR_MOVES | CHALLENGE_MOVES
FIXED_EFFECT_COLUMNS = [
    "Intercept",
    "Role[host]",
    "Violation[Relation]",
    "Violation[Quantity]",
    "Role[host]:Violation[Relation]",
    "Role[host]:Violation[Quantity]",
]
NON_VIOLATION_LABELS = {
    "no violation",
    "none",
    "no maxim violation",
}


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        return int(float(value))
    except Exception:
        return default


def safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def normal_survival(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def finite_or_none(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def sort_turns(turns: Iterable[Dict]) -> List[Dict]:
    indexed: List[Tuple[int, int, Dict]] = []
    for pos, turn in enumerate(turns):
        indexed.append((safe_int(turn.get("turn_idx"), pos), pos, turn))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def load_turns(path: Path) -> Optional[List[Dict]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, list):
        return sort_turns(payload)
    if isinstance(payload, dict) and isinstance(payload.get("turns"), list):
        return sort_turns(payload["turns"])
    return None


def turn_text(turn: Optional[Dict]) -> str:
    if not turn:
        return ""
    return str(turn.get("turn_text") or turn.get("transcript") or "").strip()


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def speaker_id(turn: Optional[Dict]) -> str:
    if not turn:
        return ""
    return str(turn.get("speaker_id") or turn.get("speaker") or "").strip()


def same_speaker(turn_a: Optional[Dict], turn_b: Optional[Dict]) -> bool:
    speaker_a = speaker_id(turn_a)
    speaker_b = speaker_id(turn_b)
    return bool(speaker_a and speaker_b and speaker_a == speaker_b)


def normalize_role(turn: Dict) -> str:
    role = str(turn.get("speaker_role") or "").strip().lower()
    return role if role in ROLE_ORDER else "unknown"


def word_count(turn: Dict) -> int:
    explicit = safe_float(turn.get("wordCount", turn.get("word_count")))
    if math.isfinite(explicit):
        return max(0, int(round(explicit)))
    text = turn_text(turn)
    if not text:
        return 0
    return len(text.split())


def normalize_maxim_violation_label(value: object) -> Optional[str]:
    text = normalize_space(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in NON_VIOLATION_LABELS:
        return None
    for label in VIOLATION_ORDER:
        if lowered == label.lower():
            return label
    return None


def explicit_no_violation_label(value: object) -> bool:
    text = normalize_space(value)
    return bool(text) and text.lower() in NON_VIOLATION_LABELS


def annotation_violation_label(turn: Dict) -> Tuple[Optional[str], bool]:
    raw = turn.get("maxim_violation_label")
    if raw is None:
        return None, False
    label = normalize_maxim_violation_label(raw)
    if label is not None:
        return label, True
    if explicit_no_violation_label(raw):
        return None, True
    return None, False


def classify_violation(turn: Dict) -> Tuple[Optional[str], Optional[str], str]:
    annotated_label, annotation_available = annotation_violation_label(turn)
    if annotation_available:
        if annotated_label is None:
            return None, None, ANNOTATED_VIOLATION_SOURCE
        return annotated_label, ANNOTATED_VIOLATION_TRIGGER, ANNOTATED_VIOLATION_SOURCE
    return None, None, ANNOTATED_VIOLATION_SOURCE


def classify_next_turn(
    source_turn: Dict,
    next_turn: Optional[Dict],
) -> Dict[str, object]:
    source_speaker = speaker_id(source_turn)
    next_speaker = ""
    next_role = ""
    next_move = ""
    same_speaker = False
    repair = 0
    challenge = 0
    external_policing = 0
    self_repair = 0

    if next_turn:
        next_speaker = speaker_id(next_turn)
        next_role = normalize_role(next_turn)
        next_move = str(next_turn.get("conversation_move_label") or "").strip()
        same_speaker = bool(source_speaker and next_speaker and source_speaker == next_speaker)
        repair = int(next_move in REPAIR_MOVES)
        challenge = int(next_move in CHALLENGE_MOVES)
        if same_speaker and next_move in SELF_REPAIR_MOVES:
            self_repair = 1
        if (not same_speaker) and next_move in POLICING_MOVES:
            external_policing = 1

    return {
        "next_move": next_move,
        "next_role": next_role,
        "same_speaker": int(same_speaker),
        "repair": repair,
        "challenge": challenge,
        "self_repair": self_repair,
        "policed": external_policing,
    }


def build_rows(
    input_dir: Path,
    max_files: int = 0,
) -> Tuple[List[Dict], Counter, Counter, Counter]:
    nested_files = sorted(input_dir.glob("*/*.json"))
    direct_files = sorted(input_dir.glob("*.json"))
    files = nested_files if nested_files else direct_files
    if max_files > 0:
        files = files[:max_files]

    rows: List[Dict] = []
    role_turn_counts: Counter = Counter()
    violation_counts: Counter = Counter()
    violation_source_counts: Counter = Counter()

    for path in tqdm(files, desc="Scanning episodes", unit="file"):
        turns = load_turns(path)
        if not turns:
            continue

        episode_id = str(turns[0].get("episode_id") or path.stem)
        for idx, turn in enumerate(turns):
            role = normalize_role(turn)
            if role in ROLE_ORDER:
                role_turn_counts[role] += 1

            next_turn = turns[idx + 1] if idx + 1 < len(turns) else None
            violation, trigger, violation_source = classify_violation(turn)
            if violation is None or role not in ROLE_ORDER:
                continue

            next_meta = classify_next_turn(turn, next_turn)
            turn_idx = safe_int(turn.get("turn_idx"), idx)
            wc = word_count(turn)
            violation_counts[(role, violation)] += 1
            violation_source_counts[violation_source] += 1

            rows.append(
                {
                    "episode_id": episode_id,
                    "turn_idx": turn_idx,
                    "speaker_role": role,
                    "speaker_id": str(turn.get("speaker_id") or turn.get("speaker") or ""),
                    "violation_type": violation,
                    "violation_trigger": trigger,
                    "violation_source": violation_source,
                    "severity": VIOLATION_SEVERITY[violation],
                    "conversation_move_label": str(turn.get("conversation_move_label") or ""),
                    "turn_type_label": str(turn.get("turn_type_label") or ""),
                    "word_count": wc,
                    "turn_text": turn_text(turn),
                    "next_move": next_meta["next_move"],
                    "next_role": next_meta["next_role"],
                    "same_speaker_next": next_meta["same_speaker"],
                    "repair_next": next_meta["repair"],
                    "challenge_next": next_meta["challenge"],
                    "self_repair_next": next_meta["self_repair"],
                    "policed_next": next_meta["policed"],
                }
            )

    rows.sort(key=lambda row: (str(row["episode_id"]), int(row["turn_idx"])))
    return rows, role_turn_counts, violation_counts, violation_source_counts


def canonical_term_name(name: str) -> str:
    raw = str(name or "").strip()
    if raw in FIXED_EFFECT_COLUMNS:
        return raw
    if raw == "Intercept":
        return raw

    has_role = "speaker_role" in raw and "[T.host]" in raw
    has_relation = "violation_type" in raw and "[T.Relation]" in raw
    has_quantity = "violation_type" in raw and "[T.Quantity]" in raw

    if has_role and has_relation:
        return "Role[host]:Violation[Relation]"
    if has_role and has_quantity:
        return "Role[host]:Violation[Quantity]"
    if has_role:
        return "Role[host]"
    if has_relation:
        return "Violation[Relation]"
    if has_quantity:
        return "Violation[Quantity]"
    return raw


def fit_statsmodels_mixed_logit(
    rows: Sequence[Dict],
    max_iter: int,
) -> Dict[str, object]:
    tqdm.write("Fitting mixed-effects logistic model with statsmodels...")
    frame = pd.DataFrame(
        {
            "episode_id": [str(row["episode_id"]) for row in rows],
            "speaker_role": [str(row["speaker_role"]) for row in rows],
            "violation_type": [str(row["violation_type"]) for row in rows],
            "policed_next": [int(row["policed_next"]) for row in rows],
        }
    )
    frame["speaker_role"] = pd.Categorical(frame["speaker_role"], categories=list(ROLE_ORDER), ordered=True)
    frame["violation_type"] = pd.Categorical(frame["violation_type"], categories=list(VIOLATION_ORDER), ordered=True)

    formula = (
        "policed_next ~ "
        "C(speaker_role, Treatment(reference='guest')) * "
        "C(violation_type, Treatment(reference='Manner'))"
    )
    vc_formulas = {
        "episode_intercept": "0 + C(episode_id)",
    }

    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, frame)
    try:
        result = model.fit_vb(
            fit_method="BFGS",
            minim_opts={"maxiter": max_iter},
        )
        fit_mode = "statsmodels_vb"
    except Exception:
        result = model.fit_map(
            method="BFGS",
            minim_opts={"maxiter": max_iter},
        )
        fit_mode = "statsmodels_map"

    term_names = [canonical_term_name(name) for name in getattr(model, "exog_names", [])]
    fe_mean = np.asarray(getattr(result, "fe_mean"), dtype=float)
    fe_sd = np.asarray(getattr(result, "fe_sd"), dtype=float)
    term_to_idx = {name: idx for idx, name in enumerate(term_names)}

    beta = np.zeros(len(FIXED_EFFECT_COLUMNS), dtype=float)
    se = np.full(len(FIXED_EFFECT_COLUMNS), np.nan, dtype=float)
    for idx, term in enumerate(FIXED_EFFECT_COLUMNS):
        source_idx = term_to_idx.get(term)
        if source_idx is None:
            continue
        beta[idx] = float(fe_mean[source_idx])
        se[idx] = float(fe_sd[source_idx])

    predicted = result.predict()
    p_hat = np.asarray(predicted, dtype=float)

    log_lik = float(
        np.sum(
            frame["policed_next"].to_numpy(dtype=float) * np.log(np.clip(p_hat, 1e-12, 1.0))
            + (1.0 - frame["policed_next"].to_numpy(dtype=float)) * np.log(np.clip(1.0 - p_hat, 1e-12, 1.0))
        )
    )

    optim = getattr(result, "optim_retvals", None)
    converged = True
    iterations = 0
    objective = None
    if isinstance(optim, dict):
        converged = bool(optim.get("success", True))
        iterations = safe_int(optim.get("nit"), 0)
        objective = finite_or_none(optim.get("fun"))

    return {
        "beta": beta,
        "u": np.array([], dtype=float),
        "cov": np.diag(np.where(np.isfinite(se), se ** 2, np.nan)),
        "se": se,
        "p_hat": p_hat,
        "log_lik": log_lik,
        "objective": objective,
        "converged": converged,
        "iterations": iterations,
        "method": fit_mode,
    }


def coefficient_rows(
    columns: Sequence[str],
    beta: np.ndarray,
    se: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(columns):
        estimate = float(beta[idx])
        std_err = float(se[idx])
        z_score = estimate / std_err if std_err > 0 else float("nan")
        p_value = 2.0 * normal_survival(abs(z_score)) if math.isfinite(z_score) else float("nan")
        ci_low = estimate - (1.96 * std_err)
        ci_high = estimate + (1.96 * std_err)
        rows.append(
            {
                "term": name,
                "estimate": estimate,
                "std_error": std_err,
                "z_score": z_score,
                "p_value_approx": p_value,
                "odds_ratio": math.exp(estimate),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return rows


def violation_matrix_rows(
    role_turn_counts: Counter,
    violation_counts: Counter,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role in ROLE_ORDER:
        total_turns = int(role_turn_counts.get(role, 0))
        role_violation_total = sum(int(violation_counts.get((role, v), 0)) for v in VIOLATION_ORDER)
        for violation in VIOLATION_ORDER:
            count = int(violation_counts.get((role, violation), 0))
            rows.append(
                {
                    "speaker_role": role,
                    "violation_type": violation,
                    "total_turns_for_role": total_turns,
                    "violation_count": count,
                    "violation_rate_per_turn": (count / total_turns) if total_turns else 0.0,
                    "share_of_role_violations": (count / role_violation_total) if role_violation_total else 0.0,
                }
            )
    return rows


def enforcement_summary_rows(rows: Sequence[Dict]) -> List[Dict[str, object]]:
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        key = (str(row["speaker_role"]), str(row["violation_type"]))
        bucket = summary.setdefault(
            key,
            {
                "n_violations": 0.0,
                "policed_next": 0.0,
                "repair_next": 0.0,
                "challenge_next": 0.0,
                "self_repair_next": 0.0,
            },
        )
        bucket["n_violations"] += 1.0
        bucket["policed_next"] += float(row["policed_next"])
        bucket["repair_next"] += float(row["repair_next"])
        bucket["challenge_next"] += float(row["challenge_next"])
        bucket["self_repair_next"] += float(row["self_repair_next"])

    out: List[Dict[str, object]] = []
    for role in ROLE_ORDER:
        for violation in VIOLATION_ORDER:
            bucket = summary.get((role, violation), None)
            n = float(bucket["n_violations"]) if bucket else 0.0
            policed = float(bucket["policed_next"]) if bucket else 0.0
            repairs = float(bucket["repair_next"]) if bucket else 0.0
            challenges = float(bucket["challenge_next"]) if bucket else 0.0
            self_repairs = float(bucket["self_repair_next"]) if bucket else 0.0
            out.append(
                {
                    "speaker_role": role,
                    "violation_type": violation,
                    "n_violations": int(n),
                    "policed_next_count": int(policed),
                    "repair_next_count": int(repairs),
                    "challenge_next_count": int(challenges),
                    "self_repair_next_count": int(self_repairs),
                    "policed_next_rate": (policed / n) if n else 0.0,
                    "repair_next_rate": (repairs / n) if n else 0.0,
                    "challenge_next_rate": (challenges / n) if n else 0.0,
                    "self_repair_next_rate": (self_repairs / n) if n else 0.0,
                }
            )
    return out


def predicted_curve(beta: np.ndarray) -> List[Dict[str, object]]:
    curve: List[Dict[str, object]] = []
    for role in ROLE_ORDER:
        role_host = 1.0 if role == "host" else 0.0
        for violation in VIOLATION_ORDER:
            is_relation = 1.0 if violation == "Relation" else 0.0
            is_quantity = 1.0 if violation == "Quantity" else 0.0
            features = np.array(
                [
                    1.0,
                    role_host,
                    is_relation,
                    is_quantity,
                    role_host * is_relation,
                    role_host * is_quantity,
                ],
                dtype=float,
            )
            eta = float(np.dot(features, beta))
            probability = float(sigmoid(np.array([eta], dtype=float))[0])
            curve.append(
                {
                    "speaker_role": role,
                    "violation_type": violation,
                    "severity": VIOLATION_SEVERITY[violation],
                    "predicted_policed_probability": probability,
                }
            )
    curve.sort(key=lambda row: (str(row["speaker_role"]), int(row["severity"])))
    return curve


def overall_rate(rows: Sequence[Dict], role: str, field: str) -> float:
    filtered = [row for row in rows if row["speaker_role"] == role]
    if not filtered:
        return 0.0
    total = sum(float(row[field]) for row in filtered)
    return total / float(len(filtered))


def model_weighted_gap(curve_rows: Sequence[Dict], rows: Sequence[Dict]) -> float:
    violation_counts = Counter(str(row["violation_type"]) for row in rows)
    total = float(sum(violation_counts.values()))
    if total <= 0:
        return 0.0

    by_key = {
        (str(row["speaker_role"]), str(row["violation_type"])): float(row["predicted_policed_probability"])
        for row in curve_rows
    }
    guest = 0.0
    host = 0.0
    for violation in VIOLATION_ORDER:
        weight = float(violation_counts.get(violation, 0)) / total
        guest += weight * by_key.get(("guest", violation), 0.0)
        host += weight * by_key.get(("host", violation), 0.0)
    return guest - host


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_png_plot(path: Path, curve_rows: Sequence[Dict[str, object]]) -> None:
    colors = {
        "guest": "#b42318",
        "host": "#175cd3",
    }
    severity_to_label = {VIOLATION_SEVERITY[violation]: f"{VIOLATION_SEVERITY[violation]}. {violation}" for violation in VIOLATION_ORDER}
    x_values = [VIOLATION_SEVERITY[violation] for violation in VIOLATION_ORDER]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e5e7eb", linewidth=1.0)
    ax.set_axisbelow(True)

    for role in ROLE_ORDER:
        role_rows = [row for row in curve_rows if row["speaker_role"] == role]
        role_rows.sort(key=lambda row: int(row["severity"]))
        xs = [int(row["severity"]) for row in role_rows]
        ys = [float(row["predicted_policed_probability"]) for row in role_rows]
        ax.plot(xs, ys, color=colors[role], linewidth=2.8, marker="o", markersize=6, label=role.title())
        for x_value, y_value in zip(xs, ys):
            ax.text(x_value, y_value + 0.03, f"{y_value:.2f}", ha="center", va="bottom", fontsize=10, color=colors[role])

    ax.text(
        0.5,
        1.01,
        "Predicted probability that the next turn polices the violation",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475467",
    )
    ax.set_xlabel("Violation ordering used in analysis (Manner -> Relation -> Quantity)", fontsize=12)
    ax.set_ylabel("Probability of policing", fontsize=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels([severity_to_label[x_value] for x_value in x_values], fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper right")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    max_files: int,
    max_iter: int,
) -> Dict[str, object]:
    rows, role_turn_counts, violation_counts, violation_source_counts = build_rows(
        input_dir=input_dir,
        max_files=max_files,
    )
    if not rows:
        raise RuntimeError("No violation rows were constructed from the input directory.")

    groups = sorted({str(row["episode_id"]) for row in rows})
    model = fit_statsmodels_mixed_logit(
        rows=rows,
        max_iter=max_iter,
    )

    coeffs = coefficient_rows(FIXED_EFFECT_COLUMNS, model["beta"], model["se"])
    matrix_rows = violation_matrix_rows(role_turn_counts, violation_counts)
    enforcement_rows = enforcement_summary_rows(rows)
    curve_rows = predicted_curve(model["beta"])

    observed_guest = overall_rate(rows, "guest", "policed_next")
    observed_host = overall_rate(rows, "host", "policed_next")
    observed_gap = observed_guest - observed_host
    model_gap = model_weighted_gap(curve_rows, rows)

    interaction_map = {row["term"]: row["estimate"] for row in coeffs}
    interaction_effects = {
        "role_host_x_relation": float(interaction_map.get("Role[host]:Violation[Relation]", 0.0)),
        "role_host_x_quantity": float(interaction_map.get("Role[host]:Violation[Quantity]", 0.0)),
    }
    interaction_magnitude = max(abs(val) for val in interaction_effects.values()) if interaction_effects else 0.0

    summary = {
        "experiment": "exp7_social_power",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_violation_rows": len(rows),
        "n_conversations": len(groups),
        "model_formula_proxy": "P(policed_next) ~ violation_type * speaker_role + (1|episode_id)",
        "operationalization": {
            "definition": (
                "A maxim violation is a substantive source turn that, relative to the immediately "
                "preceding turn and the action it projects, creates a local problem of sufficiency, "
                "relevance, or interpretability strong enough that a cooperative recipient would be "
                "warranted in withholding straightforward uptake."
            ),
            "label_source_precedence": [
                "Use maxim_violation_label embedded on each turn in the maxim-labeled input files.",
            ],
            "Quantity": [
                "Underinformative or overinformative relative to the immediate discourse demand.",
            ],
            "Relation": [
                "Insufficiently responsive to the question, challenge, or repair request currently on the table.",
            ],
            "Manner": [
                "Too unclear, incomplete, disordered, or under-specified for current purposes.",
            ],
            "ExcludedFromViolation": [
                "Backchannel and Procedural turns.",
                "Ordinary topic management when no response is projected from the current speaker.",
                "Self-Correction as such; self-repair within the same turn is not automatically a violation.",
                "Rudeness, disagreement, or socially dispreferred but usable turns.",
            ],
            "PolicedNext": [
                "N+1 is a different speaker",
                "N+1 move is a clarification request or correction / challenge",
            ],
        },
        "violation_source_counts": {str(key): int(value) for key, value in violation_source_counts.items()},
        "role_turn_counts": {role: int(role_turn_counts.get(role, 0)) for role in ROLE_ORDER},
        "observed_policing_rate": {
            "guest": observed_guest,
            "host": observed_host,
            "guest_minus_host": observed_gap,
        },
        "model_weighted_guest_minus_host_gap": model_gap,
        "status_shield_supported": bool(model_gap > 0.0),
        "interaction_effects": interaction_effects,
        "interaction_effect_magnitude": interaction_magnitude,
        "model_fit": {
            "method": str(model.get("method") or "unknown"),
            "converged": bool(model["converged"]),
            "iterations": int(model["iterations"]),
            "log_likelihood": finite_or_none(model.get("log_lik")),
            "objective": finite_or_none(model.get("objective")),
        },
    }

    output_jobs = [
        (
            "exp7_violation_turns.csv",
            lambda: write_csv(
                output_dir / "exp7_violation_turns.csv",
                rows,
                [
                    "episode_id",
                    "turn_idx",
                    "speaker_role",
                    "speaker_id",
                    "violation_type",
                    "violation_trigger",
                    "violation_source",
                    "severity",
                    "conversation_move_label",
                    "turn_type_label",
                    "word_count",
                    "turn_text",
                    "next_move",
                    "next_role",
                    "same_speaker_next",
                    "repair_next",
                    "challenge_next",
                    "self_repair_next",
                    "policed_next",
                ],
            ),
        ),
        (
            "exp7_violation_matrix.csv",
            lambda: write_csv(
                output_dir / "exp7_violation_matrix.csv",
                matrix_rows,
                [
                    "speaker_role",
                    "violation_type",
                    "total_turns_for_role",
                    "violation_count",
                    "violation_rate_per_turn",
                    "share_of_role_violations",
                ],
            ),
        ),
        (
            "exp7_enforcement_summary.csv",
            lambda: write_csv(
                output_dir / "exp7_enforcement_summary.csv",
                enforcement_rows,
                [
                    "speaker_role",
                    "violation_type",
                    "n_violations",
                    "policed_next_count",
                    "repair_next_count",
                    "challenge_next_count",
                    "self_repair_next_count",
                    "policed_next_rate",
                    "repair_next_rate",
                    "challenge_next_rate",
                    "self_repair_next_rate",
                ],
            ),
        ),
        (
            "exp7_model_coefficients.csv",
            lambda: write_csv(
                output_dir / "exp7_model_coefficients.csv",
                coeffs,
                [
                    "term",
                    "estimate",
                    "std_error",
                    "z_score",
                    "p_value_approx",
                    "odds_ratio",
                    "ci95_low",
                    "ci95_high",
                ],
            ),
        ),
        (
            "exp7_interaction_curve.csv",
            lambda: write_csv(
                output_dir / "exp7_interaction_curve.csv",
                curve_rows,
                [
                    "speaker_role",
                    "violation_type",
                    "severity",
                    "predicted_policed_probability",
                ],
            ),
        ),
        (
            "exp7_status_shield_plot.png",
            lambda: render_png_plot(output_dir / "exp7_status_shield_plot.png", curve_rows),
        ),
        (
            "exp7_summary.json",
            lambda: (output_dir / "exp7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"),
        ),
    ]

    for label, job in tqdm(output_jobs, desc="Writing outputs", unit="file"):
        job()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 7: test whether host turns are policed less often than guest turns "
            "for comparable local maxim violations using embedded maxim_violation_label annotations."
        )
    )
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_files", type=int, default=0, help="Optional debug cap on the number of episode files.")
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    summary = run_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        max_files=max(0, int(args.max_files)),
        max_iter=max(20, int(args.max_iter)),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
