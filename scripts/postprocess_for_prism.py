"""Normalize recognition predictions into the prism IPA convention for scoring.

The map converts the training alphabet only (it is derived from GT-vs-GT
alignment), so it cannot mask genuine model errors.

Usage:
    python scripts/postprocess_for_prism.py <input.jsonl> <output.jsonl> \
        [map.json] [--keep-ids ids.txt]
"""

import argparse
import ast
import json

MAP_FILE = "configs/inference/seg2prism_ipa_map.json"


def map_units(units, drop, split, relabel):
    """Apply drop -> split -> relabel to a list of {label, ...} units."""
    out = []
    for u in units:
        lab = u.get("label")
        if lab is None:
            out.append(u)
        elif lab in drop:
            continue
        elif lab in split:
            out.extend({**u, "label": x} for x in split[lab])
        else:
            out.append({**u, "label": relabel.get(lab, lab)})
    return out


def map_pred(pred, drop, split, relabel):
    """Rewrite a prediction payload (flat list or {head: {utt_id: units}})."""
    if isinstance(pred, list):
        return map_units(pred, drop, split, relabel)
    if isinstance(pred, dict):
        if "error" in pred:
            return pred
        return {
            head: {k: map_units(v, drop, split, relabel) for k, v in d.items()}
            for head, d in pred.items()
        }
    return pred


def utt_id(payload):
    """Normalized (``/`` -> ``-``) utt_id of a record payload, or None."""
    pt = payload.get("passthrough", {})
    if isinstance(pt, str):
        pt = ast.literal_eval(pt)
    uid = pt.get("utt_id") or pt.get("key")
    return uid.replace("/", "-") if uid else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="recognition jsonl to transform")
    ap.add_argument("output", help="output jsonl")
    ap.add_argument(
        "map_file",
        nargs="?",
        default=MAP_FILE,
        help=f"alphabet convention map (default: {MAP_FILE})",
    )
    ap.add_argument(
        "--keep-ids",
        help="optional file of utt_ids (one per line); records whose utt_id is "
        "not listed are dropped. Ids matched after normalizing '/' -> '-'.",
    )
    args = ap.parse_args()

    with open(args.map_file) as f:
        m = json.load(f)
    drop, split, relabel = set(m["drop"]), m["split"], m["relabel"]

    keep_ids = None
    if args.keep_ids:
        with open(args.keep_ids) as f:
            keep_ids = {
                line.strip().replace("/", "-") for line in f if line.strip()
            }

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if keep_ids is not None:
                if utt_id(next(iter(rec.values()))) not in keep_ids:
                    continue
            for payload in rec.values():
                payload["pred"] = map_pred(
                    payload.get("pred"), drop, split, relabel
                )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
