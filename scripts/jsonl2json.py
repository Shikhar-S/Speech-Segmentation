# MODELS: ctag lv60 xlsr53 powsm powsm_ctc zipactc zipactc_ns
# DATA: edacc uaspeech vaanigeo
# doreco gmuaccent l2arctic_perceived timit tusom2021 voxangeles
# for mp in $(seq 0 0.1 0.9); do python scripts/jsonl2json.py --dirname exp/runs/inf_timit_pr_powsm_ctc/masked_pr_${mp}; done
# python scripts/jsonl2json.py --dirname exp/runs/inf_cmul2arcticl1_qweni/1jobArr
# for d in cmul2arcticl1 easycall edacc fleurs speechocean uaspeech ultrasuite_child vaanigeo; do python scripts/jsonl2json.py --dirname exp/runs/infzs_${d}_qweni/1jobArr; done
# for m in ctag lv60 xlsr53 powsm powsm_ctc zipactc zipactc_ns; do python scripts/jsonl2json.py --dirname exp/runs/inf_cmul2arcticl1_$m/8jobARR; done
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", required=True)
args = parser.parse_args()
dirpath = Path(args.dirname)
out = dirpath / "transcription.json"
merged = {}
for p in sorted(dirpath.glob("*jsonl")):
    for line in p.open():
        if line.strip():
            merged.update(json.loads(line))
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"Merged {len(merged)} entries into: {out}")
