from __future__ import annotations

import sys

from exp1_representation_baselines import main as run_experiment


def main(argv: list[str]) -> None:
    run_experiment(["--prepare_only", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
