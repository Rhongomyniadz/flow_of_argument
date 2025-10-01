import os
from pathlib import Path

# Base directory (adjust if you want a different subfolder)
speaker_turn_path = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes")

# Output dir
outdir = Path("./sampled_outputs")
outdir.mkdir(exist_ok=True)

# ---- Find one JSONL file ----
# For example: go into subfolder 4/2 and pick the first file
target_dir = speaker_turn_path / "4" / "2"
files = sorted(target_dir.glob("*.jsonl"))
if not files:
    raise FileNotFoundError(f"No .jsonl files found in {target_dir}")

target_file = files[0]
print(f"Sampling from {target_file}")

# ---- Sample lines ----
with open(target_file, "r") as f:
    lines = f.readlines()

first_lines = lines[:5]  # first 5 lines

# ---- Write output ----
jsonl_out = outdir / "appearance_samples.jsonl"
with open(jsonl_out, "w") as f:
    f.writelines(first_lines)

print(f"Saved {len(first_lines)} samples from {target_file.name} to {jsonl_out}")

