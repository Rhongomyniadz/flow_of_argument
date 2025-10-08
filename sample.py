import json
from pathlib import Path

# -------- Paths --------
ROOT = Path("/shared/3/projects/podcasts/transcriptionQueue/turns/pol_appearance_episodes")
OUTDIR = Path("./sampled_outputs")
OUTDIR.mkdir(exist_ok=True)

# -------- Config --------
MIN_WORDS = 50
TARGET_EPISODES = 30


def iter_episode_files(root: Path):
    """Yield all *.jsonl files under root in sorted order."""
    return sorted(root.rglob("*.jsonl"))


def collect_episode(fpath: Path):
    """Return all turns if every turn > MIN_WORDS, else return None to skip."""
    turns = []
    try:
        with fpath.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                wc = int(obj.get("wordCount", 0))
                if wc <= MIN_WORDS:
                    # reject entire episode immediately
                    return None
                turns.append(obj)
    except Exception as e:
        print(f"[warn] Skipping unreadable {fpath}: {e}")
        return None
    return turns


def main():
    files = list(iter_episode_files(ROOT))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {ROOT}")

    kept = 0
    for ep_path in files:
        if kept >= TARGET_EPISODES:
            break

        turns = collect_episode(ep_path)
        if turns is None:
            continue  # skip episodes with any short turns

        out_path = OUTDIR / f"{ep_path.stem}.json"
        with out_path.open("w") as f:
            json.dump(turns, f, indent=2)

        kept += 1
        print(f"[{kept:02d}/{TARGET_EPISODES}] Saved {len(turns)} turns to {out_path}")

    print(f"✅ Done. Wrote {kept} episodes (all turns > {MIN_WORDS} words) to {OUTDIR}")


if __name__ == "__main__":
    main()