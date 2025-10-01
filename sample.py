import os
import shutil
from pathlib import Path

# Paths
speaker_turn_path = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes")

# Output dir
outdir = Path("./sampled_outputs")
outdir.mkdir(exist_ok=True)

# ---- 1. JSONL file ----
jsonl_out = outdir / "appearance_samples.jsonl"
with open(speaker_turn_path, "r") as f:
    lines = f.readlines()

first_lines = lines[:5]
with open(jsonl_out, "w") as f:
    f.writelines(first_lines)

print(f"Saved {len(first_lines)} samples to {jsonl_out}")


