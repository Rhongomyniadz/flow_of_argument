import json
from pathlib import Path

# Base directory (adjust if you want a different subfolder)
speaker_turn_path = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes")

# Output dir
outdir = Path("./sampled_outputs")
outdir.mkdir(exist_ok=True)

# ---- Find one JSONL file ----
target_dir = speaker_turn_path / "4" / "2"  # adjust to the subdir you want
files = sorted(target_dir.glob("*.jsonl"))
if not files:
    raise FileNotFoundError(f"No .jsonl files found in {target_dir}")

target_file = files[0]
print(f"Sampling from {target_file}")

# ---- Sample and parse lines ----
with open(target_file, "r") as f:
    lines = [json.loads(line) for line in f][:5]  # parse JSON objects

# ---- Save as JSON array ----
json_out = outdir / "appearance_samples.json"
with open(json_out, "w") as f:
    json.dump(lines, f, indent=2)

print(f"Saved {len(lines)} samples from {target_file.name} to {json_out}")
