import argparse
import json
from pathlib import Path


def verify_alternation(json_path: Path) -> tuple[bool, list]:
    """检查输出是否满足 ABAB 交替"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    speakers = [t.get("speaker_id") for t in data]
    violations = [(i, speakers[i], speakers[i + 1]) for i in range(len(speakers) - 1) if speakers[i] == speakers[i + 1]]
    return len(violations) == 0, violations


def load_latest_manifest_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out = set()
    for rec in manifest.get("episodes_written", []):
        eid = rec.get("episode_id")
        if eid is not None:
            out.add(str(eid))
    return out


def main():
    ap = argparse.ArgumentParser(description="Verify speaker alternation in parsed episode files.")
    ap.add_argument("--parsed_dir", type=str, default="data/political/parsed")
    ap.add_argument("--manifest", type=str, default="data/political/manifest_by_episode.json")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Verify all files in parsed_dir. Default verifies only episodes in manifest_by_episode.json.",
    )
    args = ap.parse_args()

    parsed_dir = Path(args.parsed_dir)
    manifest_path = Path(args.manifest)
    manifest_ids = load_latest_manifest_ids(manifest_path)

    files = sorted(parsed_dir.glob("*.json"))
    if not args.all and manifest_ids:
        files = [p for p in files if p.stem in manifest_ids]

    for p in files:
        ok, viols = verify_alternation(p)
        status = "✅" if ok else f"❌ violations: {viols}"
        print(f"{p.name}: {status}")


if __name__ == "__main__":
    main()
