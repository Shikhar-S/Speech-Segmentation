"""Pre-compute per-language attested phone vocab for phonvec recognition eval.

Iterates a prism_preval test set, tokenizes each reference transcript with the
*same* tokenizer ``scripts/eval_recognition.py`` uses (``parse_ref_labels``),
and writes the per-language union of phone tokens as a JSON dict.

The output JSON shape is::

    {
        "eng": ["b", "d", "i", "ʃ", "ɜ", "˞", ...],
        "deu": ["a", "ç", "ə", ...],
        ...
    }

which is consumed by ``src.model.phonvec.inference.PhoneModelInference`` to
build one per-language ``Recognizer`` whose featmap is restricted to that
language's attested phones.

Usage::

    PYTHONPATH=. python scripts/build_phonvec_vocab.py timit configs/inference/phonvec_vocab/timit.json
    PYTHONPATH=. python scripts/build_phonvec_vocab.py voxangeles configs/inference/phonvec_vocab/voxangeles.json
"""

from __future__ import annotations

import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

import panphon
from phone_metrics import canonical_ipa, tokenize_ipa

from src.data.recognition.prism_preval import build_prism_preval_dataset


def derive_lang(key: str, lang_sym: str) -> str:
    """Routing key for the per-language Recognizer.

    Some prism_preval datasets (e.g. voxangeles) set ``lang_sym`` to ``"unk"``
    because their lang file uses placeholder tags; the real ISO-639-3 code
    lives in the key prefix. Fall back to that when ``lang_sym`` is unknown.
    """
    if lang_sym and lang_sym != "unk":
        return lang_sym
    return key.split("-", 1)[0]


# Mirror scripts/eval_recognition.py:parse_ref_labels verbatim so the vocab
# tokenization matches what the eval will see on the reference side.
def parse_ref_labels(text) -> list[str]:
    if not text:
        return []
    s = str(text).strip()
    if " " in s:
        return s.split()
    return tokenize_ipa(canonical_ipa(s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_name", help="prism_preval dataset name (e.g. timit, voxangeles).")
    parser.add_argument("out_file", help="Output JSON path.")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("PRISM_DUMP_DIR", "exp/data/prism"),
        help="prism_preval data root.",
    )
    parser.add_argument(
        "--dataset_config_path",
        default="configs/data/powsm_evalset_index.yaml",
        help="prism_preval dataset index config.",
    )
    args = parser.parse_args()

    dataset = build_prism_preval_dataset(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        dataset_config_path=args.dataset_config_path,
        portable_wavscp=True,
    )

    lang_phones: dict[str, set[str]] = defaultdict(set)
    dropped: dict[str, set[str]] = defaultdict(set)
    n_utts = 0
    ft = panphon.FeatureTable()
    for key in dataset.keys:
        text = dataset.text[key]
        lang = derive_lang(key, dataset.key2lang[key])
        n_utts += 1
        for tok in parse_ref_labels(text):
            # Filter to panphon-known phones: the runtime ``panphon_featmap``
            # call inside ``PhoneModelInference`` rejects unknown tokens, so
            # they would have no feature vector to drive recognition anyway.
            # Ref-side tokenization in ``eval_recognition.py`` is unaffected;
            # unknown phones still count as reference tokens (and as errors).
            if ft.seg_known(tok):
                lang_phones[lang].add(tok)
            else:
                dropped[lang].add(tok)

    out = {lg: sorted(ps) for lg, ps in sorted(lang_phones.items())}
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {args.out_file} for {n_utts} utts, {len(out)} language(s).")
    for lg, phs in out.items():
        preview = phs[:12]
        suffix = "..." if len(phs) > 12 else ""
        drop = sorted(dropped.get(lg, set()))
        drop_msg = f"  [dropped {len(drop)} non-panphon: {drop!r}]" if drop else ""
        print(f"  {lg}: {len(phs)} phones (e.g. {preview!r}{suffix}){drop_msg}")


if __name__ == "__main__":
    main()
