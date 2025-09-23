import os
import json
import random
import shutil

# Paths
appearance_path = "/shared/3/projects/podcastPoliticians/polAppearanceData/polEpsDataClean_interviews2infHG.jsonl"
transcript_path = "/shared/3/projects/podcasts/transcriptionQueue/transcripts/pol_appearance_episodes"
prosody_path = "/shared/3/projects/podcasts/transcriptionQueue/prosodyMerged/pol_appearance_episodes"

# Output dir
outdir = "./sampled_outputs"
os.makedirs(outdir, exist_ok=True)

# ---- 1. JSONL file ----
jsonl_out = os.path.join(outdir, "appearance_samples.jsonl")
with open(appearance_path, "r") as f:
    lines = f.readlines()

sampled_lines = random.sample(lines, min(5, len(lines)))
with open(jsonl_out, "w") as f:
    f.writelines(sampled_lines)

print(f"Saved {len(sampled_lines)} samples to {jsonl_out}")

# ---- 2. Transcript directory ----
def sample_files(src_dir, out_dir, n=5):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    sampled = random.sample(files, min(n, len(files)))
    os.makedirs(out_dir, exist_ok=True)
    for fname in sampled:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(out_dir, fname))
    return sampled

transcript_out = os.path.join(outdir, "transcripts")
prosody_out = os.path.join(outdir, "prosody")

transcripts_sampled = sample_files(transcript_path, transcript_out, 5)
prosody_sampled = sample_files(prosody_path, prosody_out, 5)

print(f"Saved transcript samples: {transcripts_sampled}")
print(f"Saved prosody samples: {prosody_sampled}")