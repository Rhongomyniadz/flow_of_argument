import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, TextIO


DEFAULT_CATEGORIES = ["business", "commentary", "news", "religion", "sports"]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("pack-raw-json-by-category")


def iter_category_names(raw_root: Path, categories: Optional[List[str]]) -> List[str]:
    if categories:
        return categories
    return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])


def iter_json_files(category_dir: Path) -> List[Path]:
    # Use recursive glob in case category folders include nested subfolders.
    return sorted(p for p in category_dir.rglob("*.json") if p.is_file())


def iter_records(payload: Any) -> Iterator[Any]:
    if isinstance(payload, list):
        for item in payload:
            yield item
        return
    yield payload


def open_output(path: Path, gzip_output: bool) -> TextIO:
    if gzip_output:
        return gzip.open(path, mode="wt", encoding="utf-8")
    return path.open(mode="w", encoding="utf-8")


def dump_category(
    category: str,
    category_dir: Path,
    out_dir: Path,
    gzip_output: bool,
    log: logging.Logger,
) -> tuple[int, int, int]:
    json_files = iter_json_files(category_dir)
    if not json_files:
        log.warning("No JSON files found for category=%s in %s", category, category_dir)
        return 0, 0, 0

    suffix = ".jsonl.gz" if gzip_output else ".jsonl"
    out_path = out_dir / f"{category}{suffix}"

    files_seen = 0
    files_failed = 0
    records_written = 0

    with open_output(out_path, gzip_output=gzip_output) as out_f:
        for json_file in json_files:
            files_seen += 1
            try:
                with json_file.open("r", encoding="utf-8") as in_f:
                    payload = json.load(in_f)
            except Exception as exc:
                files_failed += 1
                log.warning("Skipping unreadable JSON file=%s error=%s", json_file, exc)
                continue

            for record in iter_records(payload):
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    log.info(
        "category=%s files_seen=%d files_failed=%d records_written=%d output=%s",
        category,
        files_seen,
        files_failed,
        records_written,
        out_path,
    )
    return files_seen, files_failed, records_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle raw/category/*.json files into one JSONL (or JSONL.GZ) per category."
    )
    parser.add_argument("--raw-root", type=str, default="raw", help="Root directory containing category subfolders.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for bundled files.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Category folders to process. Set to empty by omitting this flag and passing --auto-categories.",
    )
    parser.add_argument(
        "--auto-categories",
        action="store_true",
        help="Discover all category subfolders under --raw-root. Overrides --categories.",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Write .jsonl files instead of .jsonl.gz files.",
    )
    args = parser.parse_args()

    log = setup_logger()

    raw_root = Path(args.raw_root)
    if not raw_root.exists() or not raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist or is not a directory: {raw_root}")

    out_dir = Path(args.out_dir) if args.out_dir else raw_root
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = iter_category_names(raw_root, None if args.auto_categories else args.categories)
    if not categories:
        log.warning("No categories to process under raw_root=%s", raw_root)
        return

    gzip_output = not args.no_gzip

    total_files = 0
    total_failed = 0
    total_records = 0

    for category in categories:
        category_dir = raw_root / category
        if not category_dir.exists() or not category_dir.is_dir():
            log.warning("Skipping missing category directory: %s", category_dir)
            continue

        files_seen, files_failed, records_written = dump_category(
            category=category,
            category_dir=category_dir,
            out_dir=out_dir,
            gzip_output=gzip_output,
            log=log,
        )
        total_files += files_seen
        total_failed += files_failed
        total_records += records_written

    log.info(
        "Done. categories=%d files_seen=%d files_failed=%d records_written=%d out_dir=%s",
        len(categories),
        total_files,
        total_failed,
        total_records,
        out_dir,
    )


if __name__ == "__main__":
    main()
