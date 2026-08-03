from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from .common.utils import list_episode_paths, read_jsonl


PACKAGE = "experiments.exp8_assumption_embedding_pilot"
STAGES = (
    "prepare",
    "exp01",
    "exp02",
    "exp03",
    "exp04",
    "exp06-sample",
    "exp06-summarize",
    "pilot-summary",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one CPU-only Exp8 stage locally, including parallel workers and merge."
    )
    value.add_argument("stage", choices=STAGES)
    value.add_argument("--root", type=Path, default=Path("experiments/exp8_assumption_embedding_pilot"))
    value.add_argument("--input-dir", type=Path, default=Path("data/conversation_moves_labeled"))
    value.add_argument("--data-dir", type=Path)
    value.add_argument("--cache-dir", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--pairs-csv", type=Path)
    value.add_argument("--audit-csv", type=Path)
    value.add_argument("--show-map", type=Path)
    value.add_argument("--allow-episode-fallback", action="store_true")
    value.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    value.add_argument("--episodes-per-task", type=int, default=250)
    value.add_argument("--anchors-per-task", type=int, default=1000)
    value.add_argument("--candidate-count", type=int, default=25)
    value.add_argument("--development-limit", type=int, default=10000)
    value.add_argument("--bootstrap-draws", type=int, default=1000)
    value.add_argument("--feature-dim", type=int, default=256)
    value.add_argument("--max-train-anchors", type=int, default=50000)
    value.add_argument("--ridge-alpha", type=float, default=10.0)
    value.add_argument("--sample-size", type=int, default=100)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--force", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", f"{PACKAGE}.{module}", *[str(value) for value in arguments]]


def show(command: Sequence[str]) -> None:
    print(shlex.join(command), flush=True)


def run_one(command: list[str], dry_run: bool) -> None:
    show(command)
    if not dry_run:
        subprocess.run(command, check=True)


def run_workers(commands: list[list[str]], jobs: int, dry_run: bool) -> None:
    if not commands:
        raise RuntimeError("The selected CPU stage has no work units")
    print(f"WORKER_COUNT={len(commands)}", flush=True)
    print(f"LOCAL_CONCURRENCY={min(jobs, len(commands))}", flush=True)
    if dry_run:
        for command in commands:
            show(command)
        return
    with ThreadPoolExecutor(max_workers=min(jobs, len(commands))) as executor:
        futures = [executor.submit(run_one, command, False) for command in commands]
        for future in futures:
            future.result()


def common_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return args.data_dir or (args.root / "shared_data"), args.cache_dir or (args.root / "shared_cache")


def force_flag(args: argparse.Namespace) -> list[str]:
    return ["--force"] if args.force else []


def prepare(args: argparse.Namespace) -> None:
    count = len(list_episode_paths(args.input_dir))
    patches = math.ceil(count / args.episodes_per_task) if count else 0
    if patches < 1:
        raise RuntimeError(f"No episode JSON files found under {args.input_dir}")
    data_dir, _ = common_paths(args)
    shared = [
        "--input-dir", args.input_dir,
        "--output-dir", data_dir,
        "--episodes-per-task", args.episodes_per_task,
        "--candidate-count", args.candidate_count,
        "--development-limit", args.development_limit,
        "--seed", args.seed,
        "--num-patches", patches,
    ]
    if args.show_map:
        shared.extend(["--show-map", args.show_map])
    if args.allow_episode_fallback:
        shared.append("--allow-episode-fallback")
    shared.extend(force_flag(args))
    workers = [
        module_command("prepare.build_pilot_dataset", "--mode", "worker", *shared, "--patch-index", index)
        for index in range(patches)
    ]
    print(f"TOTAL_EPISODES={count}\nNUM_PATCHES={patches}\nOUTPUT_DIR={data_dir}")
    run_workers(workers, args.jobs, args.dry_run)
    run_one(module_command("prepare.build_pilot_dataset", "--mode", "merge", *shared), args.dry_run)


def exp01(args: argparse.Namespace) -> None:
    if args.pairs_csv is None:
        raise ValueError("exp01 requires --pairs-csv")
    output = args.output_dir or (args.root / "exp01_results")
    command = module_command(
        "exp01_existing_result_audit.run",
        "--pairs-csv", args.pairs_csv,
        "--output-dir", output,
        "--seed", args.seed,
        "--bootstrap-draws", args.bootstrap_draws,
        *force_flag(args),
    )
    run_one(command, args.dry_run)


def exp02(args: argparse.Namespace) -> None:
    data_dir, cache_dir = common_paths(args)
    output = args.output_dir or (args.root / "exp02_results")
    count = sum(1 for _ in read_jsonl(data_dir / "development_anchors.jsonl"))
    patches = math.ceil(count / args.anchors_per_task) if count else 0
    shared = [
        "--data-dir", data_dir,
        "--cache-dir", cache_dir,
        "--output-dir", output,
        "--anchors-per-task", args.anchors_per_task,
        "--bootstrap-draws", args.bootstrap_draws,
        "--seed", args.seed,
        "--num-patches", patches,
        *force_flag(args),
    ]
    workers = [
        module_command("exp02_frozen_retrieval.run", "--mode", "worker", *shared, "--patch-index", index)
        for index in range(patches)
    ]
    print(f"TOTAL_ANCHORS={count}\nNUM_PATCHES={patches}\nOUTPUT_DIR={output}")
    run_workers(workers, args.jobs, args.dry_run)
    run_one(module_command("exp02_frozen_retrieval.run", "--mode", "merge", *shared), args.dry_run)


def exp03(args: argparse.Namespace) -> None:
    data_dir, cache_dir = common_paths(args)
    output = args.output_dir or (args.root / "exp03_results")
    shared = [
        "--data-dir", data_dir,
        "--cache-dir", cache_dir,
        "--output-dir", output,
        "--num-patches", 3,
        "--feature-dim", args.feature_dim,
        "--max-train-anchors", args.max_train_anchors,
        "--ridge-alpha", args.ridge_alpha,
        "--seed", args.seed,
        *force_flag(args),
    ]
    workers = [
        module_command("exp03_linear_residual.run", "--mode", "worker", *shared, "--patch-index", index)
        for index in range(3)
    ]
    print(f"NUM_CONDITIONS=3\nOUTPUT_DIR={output}")
    run_workers(workers, args.jobs, args.dry_run)
    run_one(module_command("exp03_linear_residual.run", "--mode", "merge", *shared), args.dry_run)


def exp04(args: argparse.Namespace) -> None:
    data_dir, cache_dir = common_paths(args)
    output = args.output_dir or (args.root / "exp04_results")
    anchors = sum(1 for _ in read_jsonl(data_dir / "development_anchors.jsonl"))
    anchor_shards = math.ceil(anchors / args.anchors_per_task) if anchors else 0
    patches = anchor_shards * 3
    shared = [
        "--data-dir", data_dir,
        "--cache-dir", cache_dir,
        "--output-dir", output,
        "--anchors-per-task", args.anchors_per_task,
        "--bootstrap-draws", args.bootstrap_draws,
        "--seed", args.seed,
        "--num-patches", patches,
        *force_flag(args),
    ]
    workers = [
        module_command("exp04_counterfactual_controls.run", "--mode", "worker", *shared, "--patch-index", index)
        for index in range(patches)
    ]
    print(f"TOTAL_ANCHORS={anchors}\nANCHOR_SHARDS={anchor_shards}\nNUM_PATCHES={patches}\nOUTPUT_DIR={output}")
    run_workers(workers, args.jobs, args.dry_run)
    run_one(module_command("exp04_counterfactual_controls.run", "--mode", "merge", *shared), args.dry_run)


def exp06_sample(args: argparse.Namespace) -> None:
    data_dir, _ = common_paths(args)
    output = args.output_dir or (args.root / "exp06_results")
    run_one(
        module_command(
            "exp06_human_audit.sample",
            "--data-dir", data_dir,
            "--output-dir", output,
            "--sample-size", args.sample_size,
            "--seed", args.seed,
            *force_flag(args),
        ),
        args.dry_run,
    )


def exp06_summarize(args: argparse.Namespace) -> None:
    output = args.output_dir or (args.root / "exp06_results")
    audit = args.audit_csv or (output / "audit_sample.csv")
    run_one(
        module_command(
            "exp06_human_audit.summarize",
            "--audit-csv", audit,
            "--output-dir", output,
        ),
        args.dry_run,
    )


def pilot_summary(args: argparse.Namespace) -> None:
    run_one(module_command("summarize_pilot", "--root", args.root), args.dry_run)


RUNNERS = {
    "prepare": prepare,
    "exp01": exp01,
    "exp02": exp02,
    "exp03": exp03,
    "exp04": exp04,
    "exp06-sample": exp06_sample,
    "exp06-summarize": exp06_summarize,
    "pilot-summary": pilot_summary,
}


def main() -> None:
    args = parser().parse_args()
    if args.jobs < 1 or args.episodes_per_task < 1 or args.anchors_per_task < 1:
        raise ValueError("--jobs and per-task shard sizes must be positive")
    RUNNERS[args.stage](args)


if __name__ == "__main__":
    main()

