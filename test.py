import json
from pathlib import Path

def verify_alternation(json_path: Path) -> tuple[bool, list]:
    """检查输出是否满足 ABAB 交替"""
    with open(json_path) as f:
        data = json.load(f)
    speakers = [t.get("speaker_id") for t in data]
    violations = [(i, speakers[i], speakers[i+1]) for i in range(len(speakers)-1) if speakers[i] == speakers[i+1]]
    return len(violations) == 0, violations

# 批量验证
parsed_dir = Path("data/political/parsed")
for p in sorted(parsed_dir.glob("*.json")):
    ok, viols = verify_alternation(p)
    status = "✅" if ok else f"❌ violations: {viols}"
    print(f"{p.name}: {status}")
