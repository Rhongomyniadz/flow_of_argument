from __future__ import annotations

import sys

from exp1_representation_baselines import main


if __name__ == "__main__":
    main(["--merge_patches_only", *sys.argv[1:]])
