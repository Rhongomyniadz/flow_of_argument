import argparse
import inspect
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sporc import SPORCDataset


DEFAULT_CATEGORIES = ["sports", "commentary", "news", "religion", "business"]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("export-sporc-turns-by-category")


def init_dataset(sporc_dir: str, use_auth_token: Optional[str] = None) -> SPORCDataset:
    sig = inspect.signature(SPORCDataset.__init__)
    params = sig.parameters

    if "parquet_dir" in params:
        return SPORCDataset(parquet_dir=sporc_dir, use_auth_token=use_auth_token)

    if "local_data_dir" in params:
        # Legacy fallback for older sporc versions.
        return SPORCDataset(local_data_dir=sporc_dir, use_auth_token=use_auth_token, streaming=True)

    # Last-resort fallback.
    return SPORCDataset(sporc_dir, use_auth_token=use_auth_token)


def extract_raw_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    for name in ["raw", "_raw", "json", "_json", "record", "_record", "data", "_data", "source", "_source"]:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if isinstance(value, dict):
                return value

    for method_name in ["to_dict", "as_dict", "dict"]:
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, dict):
                return value

    if hasattr(obj, "__dict__") and isinstance(obj.__dict__, dict):
        fallback: Dict[str, Any] = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                fallback[k] = v
            else:
                fallback[k] = str(v)
        if fallback:
            return fallback

    return {"repr": repr(obj)}


def first_nonempty(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def get_episode_id_or_name(episode: Any) -> str:
    episode_raw = extract_raw_dict(episode)

    episode_id = first_nonempty(
        [
            getattr(episode, "episode_id", None),
            getattr(episode, "id", None),
            episode_raw.get("episode_id"),
            episode_raw.get("episodeId"),
            episode_raw.get("id"),
            episode_raw.get("guid"),
        ]
    )
    if episode_id:
        return episode_id

    episode_name = first_nonempty(
        [
            getattr(episode, "title", None),
            getattr(episode, "episode_name", None),
            getattr(episode, "name", None),
            episode_raw.get("title"),
            episode_raw.get("episode_name"),
            episode_raw.get("name"),
        ]
    )
    if episode_name:
        return episode_name

    return "unknown_episode"


def slugify_filename(name: str, max_len: int = 160) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        text = "unknown_episode"
    return text[:max_len]


def unique_output_path(out_dir: Path, file_stem: str) -> Path:
    candidate = out_dir / f"{file_stem}.json"
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = out_dir / f"{file_stem}_{idx}.json"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SPORC turn data by category to raw/{category}/{episode_id_or_name}.json"
    )
    parser.add_argument("--sporc_dir", type=str, default="/shared/6/projects/sporc/v1")
    parser.add_argument("--out_root", type=str, default="raw")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--limit_per_category",
        type=int,
        default=0,
        help="0 means no limit. Positive values cap exported episodes per category.",
    )
    parser.add_argument("--use_auth_token", type=str, default=None)
    args = parser.parse_args()

    log = setup_logger()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading SPORC dataset from %s", args.sporc_dir)
    ds = init_dataset(args.sporc_dir, use_auth_token=args.use_auth_token)

    total_written = 0
    for category in args.categories:
        out_dir = out_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Processing category=%s -> %s", category, out_dir)

        try:
            episodes = ds.search_episodes(category=category)
        except Exception as e:
            log.error("Failed to search episodes for category=%s: %s", category, e)
            continue

        written_in_category = 0
        for episode in episodes:
            if args.limit_per_category > 0 and written_in_category >= args.limit_per_category:
                break

            try:
                turns = episode.get_all_turns()
            except Exception as e:
                log.warning("Skip episode in category=%s due to turn loading error: %s", category, e)
                continue

            if not turns:
                continue

            turns_payload: List[Dict[str, Any]] = [extract_raw_dict(t) for t in turns]

            base_name = slugify_filename(get_episode_id_or_name(episode))
            out_path = unique_output_path(out_dir, base_name)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(turns_payload, f, ensure_ascii=False, indent=2)

            written_in_category += 1
            total_written += 1

        log.info("Finished category=%s, episodes_written=%d", category, written_in_category)

    log.info("Done. Total exported episode files: %d", total_written)


if __name__ == "__main__":
    main()
