"""Per-language attested-phone vocab from a *segmentation* HF dataset.

Mirrors ``scripts/build_phonvec_vocab.py`` but sources phones from the GT
``phones`` of a ``changelinglab/*-segment`` dataset (post
``HF_REPO_TRANSFORMS``), grouped by the row ``language``. Silence labels are
dropped and tokens panphon does not recognize are filtered out (the runtime
``panphon_featmap`` rejects them anyway). The output JSON shape::

    {"eng": ["b", "d", ...], "abk": ["a", "d͡ʒ", ...], ...}

is consumed unchanged by
``src.model.phonvec.inference._load_vocab_recognizers`` for oracle-boundary
recognition with per-language vocab routing.

Usage::

    PYTHONPATH=. python scripts/build_phonvec_vocab_segment.py \
        changelinglab/timit-segment configs/inference/phonvec_vocab/timit_segment.json
    PYTHONPATH=. python scripts/build_phonvec_vocab_segment.py \
        changelinglab/voxangeles-segment configs/inference/phonvec_vocab/voxangeles_segment.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import panphon

from src.core.ipa_utils import IPA_SILENCE_LABELS
from src.data.segmentation.dataset_processing_transforms import (
    transform_for_repo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hf_repo", help="changelinglab/*-segment repo id.")
    parser.add_argument("out_file", help="Output JSON path.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--cache_dir", default="exp/cache/hf", help="HF datasets cache dir."
    )
    args = parser.parse_args()

    import datasets

    transform = transform_for_repo(args.hf_repo)
    ds = datasets.load_dataset(
        args.hf_repo, split=args.split, cache_dir=args.cache_dir
    )
    # Drop the audio column so iteration does not decode waveforms.
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")

    ft = panphon.FeatureTable()
    lang_phones: dict[str, set[str]] = defaultdict(set)
    dropped: dict[str, set[str]] = defaultdict(set)
    for row in ds:
        phones = list(row["phones"])
        if transform is not None:
            timestamps = list(zip(row["phone_starts"], row["phone_ends"]))
            _, phones = transform(timestamps, phones)
        lang = row["language"]
        for phone in phones:
            if phone in IPA_SILENCE_LABELS:
                continue
            if ft.seg_known(phone):
                lang_phones[lang].add(phone)
            else:
                dropped[lang].add(phone)

    out = {lg: sorted(ps) for lg, ps in sorted(lang_phones.items())}
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {args.out_file} for {len(out)} language(s).")
    for lg, phs in out.items():
        drop = sorted(dropped.get(lg, set()))
        drop_msg = (
            f"  [dropped {len(drop)} non-panphon: {drop!r}]" if drop else ""
        )
        print(f"  {lg}: {len(phs)} phones{drop_msg}")


if __name__ == "__main__":
    main()
