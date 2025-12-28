# python scripts/jsonl2json.py --dirname DIR
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", required=True)
args = parser.parse_args()
dirpath = Path(args.dirname)
out = dirpath / "transcription.json"
merged = {}
for p in sorted(dirpath.glob("transcription.*.jsonl")):
    for line in p.open():
        if line.strip():
            merged.update(json.loads(line))
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"Merged {len(merged)} entries into: {out}")
