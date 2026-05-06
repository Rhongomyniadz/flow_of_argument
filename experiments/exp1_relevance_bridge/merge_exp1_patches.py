import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp1_relevance_bridge.exp1_relevance_bridge import (
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    main as exp1_main,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Exp 1 LLM pointwise-ranking patch outputs."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max_episodes_per_category", type=int, default=None)
    parser.add_argument("--num_patches", type=int, required=True)
    parser.add_argument("--episodes_per_patch", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--download_dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forwarded_args = [
        "exp1_relevance_bridge.py",
        "--input_dir",
        str(args.input_dir),
        "--output_dir",
        str(args.output_dir),
        "--num_patches",
        str(args.num_patches),
        "--model_name",
        args.model_name,
        "--download_dir",
        str(args.download_dir),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--prompt_batch_size",
        str(args.prompt_batch_size),
        "--max_tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--bootstrap_draws",
        str(args.bootstrap_draws),
        "--merge_patches_only",
    ]
    if args.categories is not None:
        forwarded_args.append("--categories")
        forwarded_args.extend(args.categories)
    if args.max_episodes_per_category is not None:
        forwarded_args.extend(["--max_episodes_per_category", str(args.max_episodes_per_category)])
    if args.episodes_per_patch is not None:
        forwarded_args.extend(["--episodes_per_patch", str(args.episodes_per_patch)])
    if args.no_tqdm:
        forwarded_args.append("--no_tqdm")
    if args.dry_run:
        forwarded_args.append("--dry_run")
    sys.argv = forwarded_args
    exp1_main()


if __name__ == "__main__":
    main()
