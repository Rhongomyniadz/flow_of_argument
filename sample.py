import os
import shutil
from pathlib import Path

# Paths
appearance_path = Path("/shared/3/projects/podcastPoliticians/polAppearanceData/polEpsDataClean_interviews2infHG.jsonl")
transcript_path = Path("/shared/3/projects/podcasts/transcriptionQueue/transcripts/pol_appearance_episodes")
prosody_path   = Path("/shared/3/projects/podcasts/transcriptionQueue/prosodyMerged/pol_appearance_episodes")

# Output dir
outdir = Path("./sampled_outputs")
outdir.mkdir(exist_ok=True)

# ---- 1. JSONL file ----
jsonl_out = outdir / "appearance_samples.jsonl"
with open(appearance_path, "r") as f:
    lines = f.readlines()

first_lines = lines[:5]
with open(jsonl_out, "w") as f:
    f.writelines(first_lines)

print(f"Saved {len(first_lines)} samples to {jsonl_out}")

# ---- 2. Helper for directories ----
def copy_first_files(src_dir: Path, dst_dir: Path, n=5):
    """Copy the first n files from src_dir (recursively) to dst_dir"""
    files = sorted([p for p in src_dir.rglob("*") if p.is_file()])
    dst_dir.mkdir(parents=True, exist_ok=True)
    selected = files[:n]
    for f in selected:
        rel = f.relative_to(src_dir)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, target)
    return [str(f.relative_to(src_dir)) for f in selected]

# ---- 3. Transcript directory ----
transcript_out = outdir / "transcripts"
transcripts_first = copy_first_files(transcript_path, transcript_out, 5)
print(f"Saved transcript samples: {transcripts_first}")

# ---- 4. Prosody directory ----
prosody_out = outdir / "prosody"
prosody_first = copy_first_files(prosody_path, prosody_out, 5)
print(f"Saved prosody samples: {prosody_first}")