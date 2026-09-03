from __future__ import annotations

import sys

from exp1_representation_baselines import parse_args, pending_patch_indices, validate_args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    validate_args(args)
    print(",".join(str(index) for index in pending_patch_indices(args)))


if __name__ == "__main__":
    main(sys.argv[1:])
