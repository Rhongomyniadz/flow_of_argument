import json
import pathlib
import uuid

def main():
    # Input: folder with your raw JSON list files
    in_dir = pathlib.Path("results/covid")

    # Output: single jsonl file inside your assumption_detection project
    out_dir = pathlib.Path("annotation/project-hub/assumption_detection/data_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "covid_expanded.jsonl"

    with out_path.open("w", encoding="utf-8") as fout:
        for fp in sorted(in_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text())
            except Exception as e:
                print(f"⚠️ Could not read {fp}: {e}")
                continue

            # If the file is one object, wrap in list
            if isinstance(data, dict):
                data = [data]

            for rec in data:
                turn = rec.get("turn_text", "")
                assumptions = rec.get("assumptions", []) or []

                for a in assumptions:
                    row = {
                        "id": str(uuid.uuid4()),
                        "turn_text": turn,
                        "speaker_id": rec.get("speaker_id", ""),
                        "inferred_speaker_name": rec.get("inferred_speaker_name", ""),
                        "inferred_speaker_role": rec.get("inferred_speaker_role", ""),
                        "assumption_text": a,
                        "source_file": fp.name,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Wrote flattened data to: {out_path}")

if __name__ == "__main__":
    main()