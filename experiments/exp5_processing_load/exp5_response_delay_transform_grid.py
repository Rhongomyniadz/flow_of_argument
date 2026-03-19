import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SKEW_LOG_THRESHOLD = 2.0


def finite_or_none(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def distribution_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=float)
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
        "supports_log1p": bool(min_val >= 0.0),
    }


def standardize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if not math.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (arr - mu) / sd, mu, sd


def safe_term_name(name: str, transform: str) -> str:
    return f"{name}__{transform}".replace("(", "").replace(")", "").replace("+", "plus").replace("-", "minus")


def friendly_term_name(name: str, transform: str) -> str:
    if transform == "raw":
        return name
    if transform == "log1p":
        return f"log1p({name})"
    if transform == "z_raw":
        return f"z({name})"
    if transform == "z_log1p":
        return f"z(log1p({name}))"
    raise ValueError(f"Unknown transform: {transform}")


def build_transform_options(
    name: str,
    stats: Dict[str, Optional[float]],
    allow_standardize: bool,
    include_raw: bool = True,
) -> List[str]:
    options: List[str] = []
    if include_raw:
        options.append("raw")

    supports_log1p = bool(stats.get("supports_log1p"))
    skewness = stats.get("skewness")
    should_log = supports_log1p and skewness is not None and skewness > SKEW_LOG_THRESHOLD
    if should_log:
        options.append("log1p")

    if allow_standardize and include_raw:
        options.append("z_raw")
    if allow_standardize and should_log:
        options.append("z_log1p")
    return options


def apply_transform(series: pd.Series, transform: str) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
    arr = series.to_numpy(dtype=float)
    meta: Dict[str, Optional[float]] = {
        "used_log1p": False,
        "used_standardize": False,
        "standardize_mean": None,
        "standardize_std": None,
    }

    if transform in {"log1p", "z_log1p"}:
        arr = np.log1p(arr)
        meta["used_log1p"] = True

    if transform in {"z_raw", "z_log1p"}:
        arr, mu, sd = standardize(arr)
        meta["used_standardize"] = True
        meta["standardize_mean"] = finite_or_none(mu)
        meta["standardize_std"] = finite_or_none(sd)

    return arr, meta


def coefficient_payload(result, term: str) -> Dict[str, Optional[float]]:
    estimate = float(result.params[term])
    std_error = float(result.bse[term])
    z_score = estimate / std_error if std_error > 1e-12 else float("nan")
    p_value = float(result.pvalues[term])
    ci = result.conf_int().loc[term]
    return {
        "term": term,
        "estimate": finite_or_none(estimate),
        "std_error_hc3": finite_or_none(std_error),
        "z_score": finite_or_none(z_score),
        "p_value": finite_or_none(p_value),
        "ci95_low": finite_or_none(float(ci.iloc[0])),
        "ci95_high": finite_or_none(float(ci.iloc[1])),
    }


def filter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "response_delay_at_time_n",
        "implicature_load",
        "explicit_statement_count",
        "average_response_time_0_to_n_minus_1",
    ]
    out = frame.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[
        out["response_delay_at_time_n"].notna()
        & out["implicature_load"].notna()
        & out["explicit_statement_count"].notna()
        & out["average_response_time_0_to_n_minus_1"].notna()
        & (out["response_delay_at_time_n"] >= 0.0)
        & (out["implicature_load"] >= 0.0)
        & (out["average_response_time_0_to_n_minus_1"] >= 0.0)
    ].copy()
    return out


def run_grid(input_csv: Path, output_dir: Path) -> Dict[str, object]:
    frame = pd.read_csv(input_csv)
    frame = filter_frame(frame)

    dist_checks = {
        "response_delay_at_time_n": distribution_stats(frame["response_delay_at_time_n"].to_numpy(dtype=float)),
        "implicature_load": distribution_stats(frame["implicature_load"].to_numpy(dtype=float)),
        "explicit_statement_count": distribution_stats(frame["explicit_statement_count"].to_numpy(dtype=float)),
        "average_response_time_0_to_n_minus_1": distribution_stats(
            frame["average_response_time_0_to_n_minus_1"].to_numpy(dtype=float)
        ),
    }

    outcome_options = build_transform_options(
        "response_delay_at_time_n",
        dist_checks["response_delay_at_time_n"],
        allow_standardize=False,
    )
    predictor_options = {
        "implicature_load": build_transform_options(
            "implicature_load",
            dist_checks["implicature_load"],
            allow_standardize=True,
        ),
        "explicit_statement_count": build_transform_options(
            "explicit_statement_count",
            dist_checks["explicit_statement_count"],
            allow_standardize=True,
        ),
        "average_response_time_0_to_n_minus_1": build_transform_options(
            "average_response_time_0_to_n_minus_1",
            dist_checks["average_response_time_0_to_n_minus_1"],
            allow_standardize=True,
        ),
    }

    transformed_columns: Dict[Tuple[str, str], str] = {}
    transform_metadata: Dict[str, Dict[str, object]] = {}

    all_options = {"response_delay_at_time_n": outcome_options, **predictor_options}
    for var_name, options in all_options.items():
        transform_metadata[var_name] = {}
        for transform in options:
            col_name = safe_term_name(var_name, transform)
            values, meta = apply_transform(frame[var_name], transform)
            frame[col_name] = values
            transformed_columns[(var_name, transform)] = col_name
            transform_metadata[var_name][transform] = meta

    grid_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []

    for outcome_transform in outcome_options:
        outcome_col = transformed_columns[("response_delay_at_time_n", outcome_transform)]
        outcome_name = friendly_term_name("response_delay_at_time_n", outcome_transform)
        for load_transform in predictor_options["implicature_load"]:
            for explicit_transform in predictor_options["explicit_statement_count"]:
                for avg_transform in predictor_options["average_response_time_0_to_n_minus_1"]:
                    load_col = transformed_columns[("implicature_load", load_transform)]
                    explicit_col = transformed_columns[("explicit_statement_count", explicit_transform)]
                    avg_col = transformed_columns[("average_response_time_0_to_n_minus_1", avg_transform)]

                    formula = f"{outcome_col} ~ {load_col} + {explicit_col} + {avg_col}"

                    import statsmodels.formula.api as smf

                    fit = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")

                    resid = np.asarray(fit.resid, dtype=float)
                    resid_std = float(np.std(resid))
                    resid_skew = (
                        float(np.mean(((resid - np.mean(resid)) / resid_std) ** 3))
                        if resid_std > 1e-12 and len(resid) >= 3
                        else float("nan")
                    )

                    model_id = "__".join(
                        [
                            outcome_transform,
                            load_transform,
                            explicit_transform,
                            avg_transform,
                        ]
                    )
                    formula_friendly = (
                        f"{outcome_name} ~ "
                        f"{friendly_term_name('implicature_load', load_transform)} + "
                        f"{friendly_term_name('explicit_statement_count', explicit_transform)} + "
                        f"{friendly_term_name('average_response_time_0_to_n_minus_1', avg_transform)}"
                    )

                    load_coef = coefficient_payload(fit, load_col)
                    explicit_coef = coefficient_payload(fit, explicit_col)
                    avg_coef = coefficient_payload(fit, avg_col)
                    intercept_coef = coefficient_payload(fit, "Intercept")

                    grid_rows.append(
                        {
                            "model_id": model_id,
                            "formula_friendly": formula_friendly,
                            "outcome_transform": outcome_transform,
                            "load_transform": load_transform,
                            "explicit_transform": explicit_transform,
                            "average_response_transform": avg_transform,
                            "n": int(fit.nobs),
                            "r_squared": finite_or_none(fit.rsquared),
                            "adjusted_r_squared": finite_or_none(fit.rsquared_adj),
                            "rmse": finite_or_none(math.sqrt(float(np.mean(resid ** 2)))),
                            "residual_skewness": finite_or_none(resid_skew),
                            "load_term": friendly_term_name("implicature_load", load_transform),
                            "load_estimate": load_coef["estimate"],
                            "load_std_error_hc3": load_coef["std_error_hc3"],
                            "load_z_score": load_coef["z_score"],
                            "load_p_value": load_coef["p_value"],
                            "load_ci95_low": load_coef["ci95_low"],
                            "load_ci95_high": load_coef["ci95_high"],
                        }
                    )

                    coefficient_rows.extend(
                        [
                            {
                                "model_id": model_id,
                                "formula_friendly": formula_friendly,
                                "term": "Intercept",
                                **{k: v for k, v in intercept_coef.items() if k != "term"},
                            },
                            {
                                "model_id": model_id,
                                "formula_friendly": formula_friendly,
                                "term": friendly_term_name("implicature_load", load_transform),
                                **{k: v for k, v in load_coef.items() if k != "term"},
                            },
                            {
                                "model_id": model_id,
                                "formula_friendly": formula_friendly,
                                "term": friendly_term_name("explicit_statement_count", explicit_transform),
                                **{k: v for k, v in explicit_coef.items() if k != "term"},
                            },
                            {
                                "model_id": model_id,
                                "formula_friendly": formula_friendly,
                                "term": friendly_term_name(
                                    "average_response_time_0_to_n_minus_1",
                                    avg_transform,
                                ),
                                **{k: v for k, v in avg_coef.items() if k != "term"},
                            },
                        ]
                    )

    def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    output_dir.mkdir(parents=True, exist_ok=True)

    grid_rows_sorted = sorted(
        grid_rows,
        key=lambda row: (
            str(row["outcome_transform"]),
            -(float(row["adjusted_r_squared"]) if row["adjusted_r_squared"] is not None else float("-inf")),
            float(row["load_p_value"]) if row["load_p_value"] is not None else float("inf"),
        ),
    )

    write_csv(
        output_dir / "exp5_response_delay_transform_grid.csv",
        grid_rows_sorted,
        [
            "model_id",
            "formula_friendly",
            "outcome_transform",
            "load_transform",
            "explicit_transform",
            "average_response_transform",
            "n",
            "r_squared",
            "adjusted_r_squared",
            "rmse",
            "residual_skewness",
            "load_term",
            "load_estimate",
            "load_std_error_hc3",
            "load_z_score",
            "load_p_value",
            "load_ci95_low",
            "load_ci95_high",
        ],
    )
    write_csv(
        output_dir / "exp5_response_delay_transform_grid_coefficients.csv",
        coefficient_rows,
        [
            "model_id",
            "formula_friendly",
            "term",
            "estimate",
            "std_error_hc3",
            "z_score",
            "p_value",
            "ci95_low",
            "ci95_high",
        ],
    )

    summary = {
        "requested_formula": (
            "response_delay_at_time_n ~ implicature_load + explicit_statement_count + "
            "average_response_time_0_to_n_minus_1"
        ),
        "n": int(len(frame)),
        "distribution_checks": dist_checks,
        "allowed_transforms": {
            "response_delay_at_time_n": outcome_options,
            **predictor_options,
        },
        "transform_metadata": transform_metadata,
        "num_models_fit": int(len(grid_rows)),
        "top_models_by_adjusted_r_squared": sorted(
            grid_rows,
            key=lambda row: float(row["adjusted_r_squared"]) if row["adjusted_r_squared"] is not None else float("-inf"),
            reverse=True,
        )[:10],
        "top_models_by_smallest_positive_load_p": sorted(
            [
                row
                for row in grid_rows
                if row["load_estimate"] is not None and row["load_estimate"] > 0.0 and row["load_p_value"] is not None
            ],
            key=lambda row: (float(row["load_p_value"]), -float(row["load_estimate"])),
        )[:10],
    }

    (output_dir / "exp5_response_delay_transform_grid_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit all justified log/std variants of the Exp 5 response-delay regression.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="experiments/exp5_processing_load/results/exp5_turn_level_features.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/exp5_processing_load/results/transform_grid",
    )
    args = parser.parse_args()

    summary = run_grid(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
