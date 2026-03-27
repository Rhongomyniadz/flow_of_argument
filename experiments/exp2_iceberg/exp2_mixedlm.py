import argparse
import json
import logging
import math
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm


warnings.filterwarnings("ignore", category=ConvergenceWarning)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = "data/stance_labeled/512"
DEFAULT_CATEGORY_DATA_SUBDIR = "parsed"
DEFAULT_OUTPUT_DIR = "experiments/exp2_iceberg/results_mixedlm"
MODEL_FORMULA = "other_iceberg_t_shift ~ stance_t + previous_source_stance + previous_other_iceberg"


def compute_iceberg_log_ratio(explicit: int, implicit: int) -> float:
    """Positive values mean more explicit than implicit content."""
    return float(math.log((explicit + 1.0) / (implicit + 1.0)))

def preprocess_episode(data: List[dict], min_turns: int) -> Optional[pd.DataFrame]:
    """Extracts substantive turns with stance and the plotted explicit/implicit log-ratio metric."""
    features = []

    for turn in data:
        if turn.get("turn_type_label") != "Substantive":
            continue

        stance = turn.get("stance_5pt")
        if stance is None:
            continue

        raw_start_time = turn.get("start_time", turn.get("startTime", 0.0))
        start_time = float(raw_start_time if raw_start_time is not None else 0.0)

        explicit_count = len(turn.get("explicit_propositions", []) or [])
        implicit_count = len(turn.get("assumptions", []) or [])

        features.append(
            {
                "speaker_id": str(turn.get("speaker_id") or ""),
                "stance_t": float(stance),
                "iceberg_log_ratio": compute_iceberg_log_ratio(explicit_count, implicit_count),
                "start_time": start_time,
            }
        )

    if len(features) < min_turns:
        return None

    df = pd.DataFrame(features).sort_values("start_time").reset_index(drop=True)
    if df["speaker_id"].nunique() != 2:
        return None
    df["previous_same_speaker_iceberg"] = df.groupby("speaker_id")["iceberg_log_ratio"].shift(1)
    df["previous_same_speaker_stance"] = df.groupby("speaker_id")["stance_t"].shift(1)
    return df


def discover_category_names(data_path: Path, category_data_subdir: str) -> List[str]:
    categories: List[str] = []
    try:
        children = sorted(data_path.iterdir())
    except OSError:
        return categories

    for child in children:
        try:
            if not child.is_dir():
                continue
            if (child / category_data_subdir).is_dir():
                categories.append(child.name)
        except OSError:
            continue
    return categories


def normalize_categories(categories: Optional[Sequence[str]]) -> List[str]:
    if not categories:
        return []

    normalized: List[str] = []
    seen = set()
    for category in categories:
        category_name = str(category).strip()
        if not category_name:
            continue
        key = category_name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(category_name)
    return normalized


def resolve_requested_categories(requested: Sequence[str], available: Sequence[str]) -> List[str]:
    available_lookup = {category.lower(): category for category in available}

    if not requested or any(category.lower() == "all" for category in requested):
        return list(available)

    resolved: List[str] = []
    missing: List[str] = []
    for category in requested:
        match = available_lookup.get(category.lower())
        if match is None:
            missing.append(category)
            continue
        resolved.append(match)

    if missing:
        raise ValueError(
            f"Unknown categories: {', '.join(sorted(missing))}. "
            f"Available categories: {', '.join(available)}."
        )

    return resolved


def find_category_lookup_root(data_path: Path, category_data_subdir: str) -> Optional[Path]:
    for candidate in [data_path.parent.parent, data_path.parent]:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if discover_category_names(candidate, category_data_subdir):
            return candidate
    return None


def collect_episode_files(
    data_path: Path, categories: Optional[Sequence[str]], category_data_subdir: str
) -> List[Tuple[Optional[str], Path]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {data_path}")

    requested_categories = normalize_categories(categories)
    available_categories = discover_category_names(data_path, category_data_subdir)

    if available_categories:
        selected_categories = resolve_requested_categories(requested_categories, available_categories)
        logger.info("Running categories: %s", ", ".join(selected_categories))

        episode_files: List[Tuple[Optional[str], Path]] = []
        for category in selected_categories:
            category_path = data_path / category / category_data_subdir
            category_files = sorted(category_path.glob("*.json"))
            logger.info("Category=%s | files=%d | path=%s", category, len(category_files), category_path)
            episode_files.extend((category, file_path) for file_path in category_files)
        return episode_files

    single_category_path = data_path / category_data_subdir
    if single_category_path.is_dir():
        category_name = data_path.name
        category_files = sorted(single_category_path.glob("*.json"))
        logger.info("Single category=%s | files=%d | path=%s", category_name, len(category_files), single_category_path)
        return [(category_name, file_path) for file_path in category_files]

    direct_json_files = sorted(data_path.glob("*.json"))
    if direct_json_files:
        inferred_category = data_path.parent.name if data_path.name == category_data_subdir else None
        category_lookup_root = find_category_lookup_root(data_path, category_data_subdir)

        if requested_categories and inferred_category is None and category_lookup_root is not None:
            available_categories = discover_category_names(category_lookup_root, category_data_subdir)
            selected_categories = resolve_requested_categories(requested_categories, available_categories)

            filtered_files: List[Tuple[Optional[str], Path]] = []
            missing_files = 0
            for category in selected_categories:
                category_files = sorted((category_lookup_root / category / category_data_subdir).glob("*.json"))
                for category_file in category_files:
                    candidate = data_path / category_file.name
                    if candidate.exists():
                        filtered_files.append((category, candidate))
                    else:
                        missing_files += 1

            logger.info(
                "Direct JSON category selection | categories=%s | matched_files=%d | missing_matches=%d | lookup_root=%s",
                ", ".join(selected_categories),
                len(filtered_files),
                missing_files,
                category_lookup_root,
            )
            return filtered_files

        logger.info(
            "Direct JSON input | inferred_category=%s | files=%d | path=%s",
            inferred_category or "none",
            len(direct_json_files),
            data_path,
        )
        return [(inferred_category, file_path) for file_path in direct_json_files]

    raise FileNotFoundError(
        f"No episode JSON files found under {data_path}. Expected either JSON files directly, "
        f"a single '{category_data_subdir}' subdirectory, or category directories containing "
        f"'{category_data_subdir}'."
    )


def infer_category_from_episode(raw_data: List[dict]) -> Optional[str]:
    for turn in raw_data:
        category = str(turn.get("category") or "").strip()
        if category:
            return category
    return None


def build_episode_id(category: Optional[str], file_path: Path) -> str:
    if category:
        return f"{category}/{file_path.stem}"
    return file_path.stem


def iter_offsets(max_shift: int, odd_offsets_only: bool) -> List[int]:
    if odd_offsets_only:
        offsets: List[int] = []
        for k in range(1, max_shift + 1, 2):
            offsets.extend([-k, k])
        return offsets
    return list(range(-max_shift, max_shift + 1))


def build_cross_turn_rows(
    df: pd.DataFrame,
    episode_id: str,
    category: str,
    offsets: Sequence[int],
) -> List[dict]:
    rows: List[dict] = []
    n = len(df)

    for shift in offsets:
        for idx in range(n):
            target_idx = idx + shift
            if target_idx < 0 or target_idx >= n:
                continue

            curr = df.iloc[idx]
            target = df.iloc[target_idx]
            if curr["speaker_id"] == target["speaker_id"]:
                continue
            if pd.isna(target["previous_same_speaker_iceberg"]):
                continue
            if pd.isna(curr["previous_same_speaker_stance"]):
                continue

            rows.append(
                {
                    "category": category,
                    "episode": episode_id,
                    "offset": shift,
                    "source_speaker_id": str(curr["speaker_id"]),
                    "target_speaker_id": str(target["speaker_id"]),
                    "stance_t": float(curr["stance_t"]),
                    "previous_source_stance": float(curr["previous_same_speaker_stance"]),
                    "other_iceberg_t_shift": float(target["iceberg_log_ratio"]),
                    "previous_other_iceberg": float(target["previous_same_speaker_iceberg"]),
                    "source_start_time": float(curr["start_time"]),
                    "target_start_time": float(target["start_time"]),
                }
            )

    return rows


def fit_mixedlm_for_offset(df_offset: pd.DataFrame, offset: int) -> Tuple[dict, str]:
    result_row = {
        "offset": offset,
        "n_rows": int(len(df_offset)),
        "n_episodes": int(df_offset["episode"].nunique()),
        "coef_intercept": None,
        "coef_stance_t": None,
        "std_err_stance_t": None,
        "z_stance_t": None,
        "p_value_stance_t": None,
        "ci_lower_stance_t": None,
        "ci_upper_stance_t": None,
        "coef_previous_source_stance": None,
        "std_err_previous_source_stance": None,
        "z_previous_source_stance": None,
        "p_value_previous_source_stance": None,
        "ci_lower_previous_source_stance": None,
        "ci_upper_previous_source_stance": None,
        "coef_previous_other_iceberg": None,
        "std_err_previous_other_iceberg": None,
        "z_previous_other_iceberg": None,
        "p_value_previous_other_iceberg": None,
        "ci_lower_previous_other_iceberg": None,
        "ci_upper_previous_other_iceberg": None,
        "converged": False,
        "negative_effect": None,
        "fit_error": "",
    }

    if len(df_offset) < 10:
        result_row["fit_error"] = "too_few_rows"
        return result_row, f"Offset {offset}\nstatus: too_few_rows\nn_rows={len(df_offset)}\n"
    if df_offset["episode"].nunique() < 2:
        result_row["fit_error"] = "too_few_episodes"
        return result_row, (
            f"Offset {offset}\nstatus: too_few_episodes\n"
            f"n_rows={len(df_offset)}\nn_episodes={df_offset['episode'].nunique()}\n"
        )

    try:
        model = smf.mixedlm(
            MODEL_FORMULA,
            data=df_offset,
            groups=df_offset["episode"],
        )
        fit = model.fit(reml=False, method="lbfgs", disp=False)
    except Exception as e:
        result_row["fit_error"] = str(e)[:300]
        return result_row, f"Offset {offset}\nstatus: fit_error\nerror: {result_row['fit_error']}\n"

    params = fit.params
    bse = fit.bse
    pvalues = fit.pvalues
    tvalues = fit.tvalues
    conf_int = fit.conf_int()

    stance_coef = params.get("stance_t")
    previous_source_coef = params.get("previous_source_stance")
    previous_coef = params.get("previous_other_iceberg")
    result_row.update(
        {
            "coef_intercept": float(params["Intercept"]) if "Intercept" in params else None,
            "coef_stance_t": float(stance_coef) if stance_coef is not None and np.isfinite(stance_coef) else None,
            "std_err_stance_t": float(bse["stance_t"]) if "stance_t" in bse and np.isfinite(bse["stance_t"]) else None,
            "z_stance_t": float(tvalues["stance_t"]) if "stance_t" in tvalues and np.isfinite(tvalues["stance_t"]) else None,
            "p_value_stance_t": float(pvalues["stance_t"])
            if "stance_t" in pvalues and np.isfinite(pvalues["stance_t"])
            else None,
            "ci_lower_stance_t": float(conf_int.loc["stance_t", 0])
            if "stance_t" in conf_int.index and np.isfinite(conf_int.loc["stance_t", 0])
            else None,
            "ci_upper_stance_t": float(conf_int.loc["stance_t", 1])
            if "stance_t" in conf_int.index and np.isfinite(conf_int.loc["stance_t", 1])
            else None,
            "coef_previous_source_stance": float(previous_source_coef)
            if previous_source_coef is not None and np.isfinite(previous_source_coef)
            else None,
            "std_err_previous_source_stance": float(bse["previous_source_stance"])
            if "previous_source_stance" in bse and np.isfinite(bse["previous_source_stance"])
            else None,
            "z_previous_source_stance": float(tvalues["previous_source_stance"])
            if "previous_source_stance" in tvalues and np.isfinite(tvalues["previous_source_stance"])
            else None,
            "p_value_previous_source_stance": float(pvalues["previous_source_stance"])
            if "previous_source_stance" in pvalues and np.isfinite(pvalues["previous_source_stance"])
            else None,
            "ci_lower_previous_source_stance": float(conf_int.loc["previous_source_stance", 0])
            if "previous_source_stance" in conf_int.index
            and np.isfinite(conf_int.loc["previous_source_stance", 0])
            else None,
            "ci_upper_previous_source_stance": float(conf_int.loc["previous_source_stance", 1])
            if "previous_source_stance" in conf_int.index
            and np.isfinite(conf_int.loc["previous_source_stance", 1])
            else None,
            "coef_previous_other_iceberg": float(previous_coef)
            if previous_coef is not None and np.isfinite(previous_coef)
            else None,
            "std_err_previous_other_iceberg": float(bse["previous_other_iceberg"])
            if "previous_other_iceberg" in bse and np.isfinite(bse["previous_other_iceberg"])
            else None,
            "z_previous_other_iceberg": float(tvalues["previous_other_iceberg"])
            if "previous_other_iceberg" in tvalues and np.isfinite(tvalues["previous_other_iceberg"])
            else None,
            "p_value_previous_other_iceberg": float(pvalues["previous_other_iceberg"])
            if "previous_other_iceberg" in pvalues and np.isfinite(pvalues["previous_other_iceberg"])
            else None,
            "ci_lower_previous_other_iceberg": float(conf_int.loc["previous_other_iceberg", 0])
            if "previous_other_iceberg" in conf_int.index
            and np.isfinite(conf_int.loc["previous_other_iceberg", 0])
            else None,
            "ci_upper_previous_other_iceberg": float(conf_int.loc["previous_other_iceberg", 1])
            if "previous_other_iceberg" in conf_int.index
            and np.isfinite(conf_int.loc["previous_other_iceberg", 1])
            else None,
            "converged": bool(getattr(fit, "converged", False)),
            "negative_effect": bool(stance_coef < 0) if stance_coef is not None and np.isfinite(stance_coef) else None,
        }
    )
    summary_text = fit.summary().as_text()
    return result_row, summary_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit baseline-adjusted mixed-effects models of "
            f"{MODEL_FORMULA} + (1 | episode)."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--category_data_subdir", type=str, default=DEFAULT_CATEGORY_DATA_SUBDIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_shift", type=int, default=3, help="Maximum turn offset (both +/-).")
    parser.add_argument(
        "--odd_offsets_only",
        action="store_true",
        help="Only compute odd offsets (recommended for strict cross-turn comparisons).",
    )
    parser.add_argument("--min_turns", type=int, default=30)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        episode_files = collect_episode_files(
            data_path=data_path,
            categories=args.categories,
            category_data_subdir=args.category_data_subdir,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    offsets = iter_offsets(args.max_shift, args.odd_offsets_only)
    logger.info("Using offsets: %s", offsets)

    all_rows: List[dict] = []
    requested_categories = normalize_categories(args.categories)
    requested_category_keys = {category.lower() for category in requested_categories if category.lower() != "all"}

    for category, file_path in tqdm(episode_files, desc="Processing Episodes"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            file_category = category or infer_category_from_episode(raw_data) or ""
            if requested_category_keys and file_category.lower() not in requested_category_keys:
                continue

            df_episode = preprocess_episode(raw_data, min_turns=args.min_turns)
            if df_episode is None:
                continue

            episode_id = build_episode_id(file_category or None, file_path)
            rows = build_cross_turn_rows(
                df=df_episode,
                episode_id=episode_id,
                category=file_category,
                offsets=offsets,
            )
            all_rows.extend(rows)
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path.name, e)

    if not all_rows:
        logger.warning("No valid rows produced. Check min_turns or data format.")
        return

    df_long = pd.DataFrame(all_rows)
    obs_csv = out_path / "mixedlm_observations_log_ratio.csv"
    df_long.to_csv(obs_csv, index=False)
    logger.info("Saved mixed-model observation rows to %s", obs_csv)

    summary_rows: List[dict] = []
    summary_text_blocks: List[str] = []
    for offset in sorted(df_long["offset"].unique()):
        df_offset = df_long[df_long["offset"] == offset].copy()
        summary_row, summary_text = fit_mixedlm_for_offset(df_offset, int(offset))
        summary_rows.append(summary_row)
        summary_text_blocks.append(
            "\n".join(
                [
                    "=" * 80,
                    f"Offset {int(offset)}",
                    f"Formula: {MODEL_FORMULA}",
                    f"Rows: {len(df_offset)}",
                    f"Episodes: {df_offset['episode'].nunique()}",
                    "=" * 80,
                    summary_text.strip(),
                    "",
                ]
            )
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_path / "mixedlm_by_offset_log_ratio.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info("Saved mixed-model summary to %s", summary_csv)

    summary_txt = out_path / "mixedlm_model_summaries_log_ratio.txt"
    summary_txt.write_text("\n".join(summary_text_blocks), encoding="utf-8")
    logger.info("Saved mixed-model text summaries to %s", summary_txt)


if __name__ == "__main__":
    main()
