import json
from pathlib import Path

# -------- Paths --------
ROOT = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes")
OUTDIR = Path("./sampled_outputs")
OUTDIR.mkdir(exist_ok=True)
OUTFILE = OUTDIR / "appearance_30episodes_max50.json"

# -------- Config --------
MAX_WORDS = 50
TARGET_EPISODES = 30

def turn_word_count(obj):
    if isinstance(obj, dict) and "wordCount" in obj and isinstance(obj["wordCount"], (int, float)):
        try:
            return int(obj["wordCount"])
        except Exception:
            pass
    tx = (obj.get("transcript") or "").strip()
    return len(tx.split()) if tx else 0

def iter_episode_files(root: Path):
    # Deterministic ordering across nested dirs (e.g., .../4/2/*.jsonl etc.)
    return sorted(root.rglob("*.jsonl"))

def collect_episode(fpath: Path):
    """Return filtered list of turns (≤ MAX_WORDS). Empty if none qualify or file unreadable."""
    turns = []
    try:
        with fpath.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wc = turn_word_count(obj)
                if wc <= MAX_WORDS:
                    turns.append({**obj, "_computedWordCount": wc})
    except Exception as e:
        print(f"[warn] Skipping unreadable {fpath}: {e}")
        return []
    return turns

def main():
    episodes_out = []
    for ep_path in iter_episode_files(ROOT):
        if len(episodes_out) >= TARGET_EPISODES:
            break
        filtered_turns = collect_episode(ep_path)
        if not filtered_turns:
            continue  # require at least one qualifying turn to keep this episode
        episodes_out.append({
            "episode_file": str(ep_path),
            "episode_id": ep_path.stem,
            "num_kept_turns": len(filtered_turns),
            "turns": filtered_turns
        })

    with OUTFILE.open("w") as f:
        json.dump(episodes_out, f, indent=2)

    print(f"Wrote {len(episodes_out)} episodes (each with turns ≤ {MAX_WORDS} words) to {OUTFILE}")

if __name__ == "__main__":
    main()