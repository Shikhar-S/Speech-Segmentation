"""
Usage:
    python -m scripts.create_train_splits \
        --wav_scp /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/train_fixed_all/wav.scp \
        --text /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/train_fixed_all/text \
        --accent_jsonl exp/data/tags/train_fixed_all.accent.shard0.jsonl \
        --out_dir exp/data/train_fixed_all_partitions
"""

import argparse, json, re
from pathlib import Path

LANG_RE = re.compile(r"<([^>]+)>")
CV_RE = re.compile(r"_cv_")


def read_wav_scp(path: Path):
    d = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            utt, rest = line.split(maxsplit=1)
            d[utt] = rest
    return d


def read_text_lang(path: Path):
    d = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            utt = line.split(maxsplit=1)[0]
            m = LANG_RE.search(line)
            if m:
                d[utt] = m.group(1)
    return d


def read_accent_jsonl(path: Path):
    d = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            d[j["utt_id"]] = j.get("tag", "").strip()
    return d


def write_subset(out: Path, utts, wav_scp_map):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fo:
        for u in utts:
            w = wav_scp_map.get(u)
            if w is not None:
                fo.write(f"{u} {w}\n")


def isprutt(utt: str) -> bool:
    return utt.endswith("_pr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument(
        "--accent_jsonl",
        required=True,
        help="e.g. exp/data/tags/accent.score.shard0.jsonl",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--mid_min_utts",
        type=int,
        default=4000,
        help=">= this => mid-resource, else low-resource",
    )
    args = ap.parse_args()

    wav = read_wav_scp(Path(args.wav_scp))
    lang = read_text_lang(Path(args.text))
    acc = read_accent_jsonl(Path(args.accent_jsonl))

    outd = Path(args.out_dir)

    # ---------- (1) English: US vs Other (CommonVoice only, using accent tag) ----------
    eng_utts = [u for u in wav.keys() if lang.get(u) == "eng" and isprutt(u)]
    eng_cv = [
        u for u in eng_utts if CV_RE.search(u) and isprutt(u)
    ]  # only those we expect accent labels for

    eng_us = [
        u for u in eng_utts if acc.get(u, "us") == "us" and isprutt(u)
    ]  # default to "us" if no tag (non-CV data)
    eng_ot = [u for u in eng_cv if (u in acc and acc.get(u) != "us") and isprutt(u)]

    write_subset(outd / "wav.eng_us.scp", eng_us, wav)
    write_subset(outd / "wav.eng_other.scp", eng_ot, wav)

    # ---------- (2) Multilingual: mid-resource vs low-resource (exclude English) ----------
    # Count per-language (over utts that exist in wav.scp)
    counts = {}
    for u in wav.keys():
        if not isprutt(u):
            continue
        l = lang.get(u)
        if not l or l == "eng":  # keep English out of the multilingual buckets
            continue
        counts[l] = counts.get(l, 0) + 1

    mid_langs = {l for l, c in counts.items() if c >= args.mid_min_utts}
    low_langs = {l for l, c in counts.items() if c < args.mid_min_utts}

    mid_utts = [u for u in wav.keys() if lang.get(u) in mid_langs and isprutt(u)]
    low_utts = [u for u in wav.keys() if lang.get(u) in low_langs and isprutt(u)]

    write_subset(outd / "wav.multi_mid.scp", mid_utts, wav)
    write_subset(outd / "wav.multi_low.scp", low_utts, wav)

    # quick summary
    def n(x):
        return len(x)

    print("Wrote:")
    print(f"  eng_us     {n(eng_us)}")
    print(f"  eng_other  {n(eng_ot)}")
    print(f"  multi_mid  {n(mid_utts)}  (langs={len(mid_langs)})")
    print(f"  multi_low  {n(low_utts)}  (langs={len(low_langs)})")


if __name__ == "__main__":
    main()
