"""Seen-vs-unseen micro per-phone PFER on the panphon-unrestricted oracle dump.

Seen = ref phone whose canonical_ipa form is in the model's fitted IPA-view
inventory. Silence and panphon-unrepresentable refs are excluded.

Usage:
    python scripts/eval_seen_unseen.py "<dir>/phonvec_oracle*.jsonl" \
        [--model juice500/wavlm-24-phonemodel]
"""

import argparse
import glob
import json

import panphon
import panphon.distance
from phone_metrics import canonical_ipa
from phonological_posteriogram import PhoneModel

DEFAULT_MODEL = "juice500/wavlm-24-phonemodel"


def base_phone(key: str) -> str:
    """Strip the closure/release state marker from a featmap key."""
    return key[:-3] if key.endswith(("_cl", "_rl")) else key


def seen_inventory(model_name: str) -> set:
    """Base-phone inventory the model was fit on, from its IPA-view featmap."""
    model = PhoneModel.from_pretrained(model_name, device="cpu")
    featmap = model.posteriogram.views["ipa"].featmap
    bases = {base_phone(k) for k in featmap if k != "_"}
    return {canonical_ipa(p) or p for p in bases}


def iter_pairs(files):
    """Yield (pred_label, ref_phone) for every oracle 1:1 position."""
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                _, payload = next(iter(json.loads(line).items()))
                pred = [u["label"] for u in (payload.get("pred") or [])]
                ref = payload.get("passthrough", {}).get("phones") or []
                if len(pred) == len(ref):
                    yield from zip(pred, ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="oracle JSONL shard(s) or glob.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    seen = seen_inventory(args.model)
    ft = panphon.FeatureTable()
    dist = panphon.distance.Distance()
    files = [f for p in args.files for f in (sorted(glob.glob(p)) or [p])]

    cost = {"seen": 0.0, "unseen": 0.0}
    n = {"seen": 0, "unseen": 0}
    for pred, ref in iter_pairs(files):
        rf = canonical_ipa(ref) or ref
        if pred == "_" or rf == "_":      # silence
            continue
        if not ft.seg_known(rf):          # unproducible: diphthong / exotic
            continue
        tag = "seen" if rf in seen else "unseen"
        cost[tag] += float(dist.feature_edit_distance(pred, rf))
        n[tag] += 1

    pfer = {t: 100 * cost[t] / n[t] for t in ("seen", "unseen")}
    print(f"seen-set: {len(seen)} base phones from {args.model}")
    print(f"seen   PFER {pfer['seen']:.2f}  (n={n['seen']})")
    print(f"unseen PFER {pfer['unseen']:.2f}  (n={n['unseen']})")
    print(f"gap         {pfer['unseen'] - pfer['seen']:.2f}")


if __name__ == "__main__":
    main()
