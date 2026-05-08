"""Per-row GT preprocessing for SegmentationDataset.

``HF_REPO_TRANSFORMS`` maps HF repo id -> per-row cleanup applied inside
``SegmentationDataset.__getitem__`` to ``(phone_timestamps, phones)``.
Must be idempotent.

DatasetDict-level split reshaping lives in ``dataset_splitting_transforms``.
# TODO(shikhar,stephen): Check this to ensure match with notebook.
"""

from typing import Callable, Dict, List, Tuple

from src.core.ipa_utils import ARPABET_TO_IPA, IPA_SILENCE_LABELS

# TIMIT closure -> stop merge tables.
# Keys are IPA (post ARPABET→IPA conversion).
_IPA_CLOSURE_TO_STOP = {
    "b̚": "b",
    "d̚": "d",
    "ɡ̚": "ɡ",
    "p̚": "p",
    "t̚": "t",
    "k̚": "k",
}
_IPA_AFFRICATE_PAIRS = {
    ("d̚", "d͡ʒ"),
    ("t̚", "t͡ʃ"),
}


def process_timit_symbols(
    phone_timestamps: List[Tuple[float, float]],
    phones: List[str],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """TIMIT GT pipeline: ARPABET→IPA + closure->stop merge + silence collapse.

    All three steps are idempotent. Labels already in IPA pass through the
    ARPABET map unchanged via the lowercase fallback.

    Args:
        phone_timestamps: list of ``(start, end)`` tuples (seconds).
        phones: parallel list of phone labels (ARPABET or IPA).

    Returns:
        ``(merged_timestamps, merged_phones)`` with phones in IPA, closure-
        stop pairs merged into the stop, and adjacent silence runs
        collapsed into a single span.
    """
    ipa_phones = [ARPABET_TO_IPA.get(p.lower(), p.lower()) for p in phones]
    segs = [(s, e, p) for (s, e), p in zip(phone_timestamps, ipa_phones)]

    merged = []
    i = 0
    while i < len(segs):
        s, e, p = segs[i]
        if p in _IPA_CLOSURE_TO_STOP and i + 1 < len(segs):
            ns, ne, np_ = segs[i + 1]
            if (
                np_ == _IPA_CLOSURE_TO_STOP[p]
                or (p, np_) in _IPA_AFFRICATE_PAIRS
            ):
                merged.append((s, ne, np_))
                i += 2
                continue
        merged.append((s, e, p))
        i += 1

    collapsed = []
    for s, e, p in merged:
        if (
            p in IPA_SILENCE_LABELS
            and collapsed
            and collapsed[-1][2] in IPA_SILENCE_LABELS
        ):
            ps, _, pp = collapsed[-1]
            collapsed[-1] = (ps, e, pp)
            continue
        collapsed.append((s, e, p))

    return (
        [(s, e) for s, e, _ in collapsed],
        [p for _, _, p in collapsed],
    )


def arpabet_phones_to_ipa(
    phone_timestamps: List[Tuple[float, float]],
    phones: List[str],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Map ARPABET phone labels to IPA, leaving timestamps untouched.

    Idempotent: labels already in IPA pass through via the lowercase
    fallback in ``ARPABET_TO_IPA.get``.
    """
    return (
        list(phone_timestamps),
        [ARPABET_TO_IPA.get(p.lower(), p.lower()) for p in phones],
    )


def process_gtimit_arpabet_symbols(
    phone_timestamps: List[Tuple[float, float]],
    phones: List[str],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """GTIMIT English ARPABET → IPA + closure->stop merge + silence collapse.

    Handles both lowercase-no-stress (l2simple) and uppercase-with-stress
    (l2tbnk, l1simple, l1tbnk) variants in a single pass: lowercase, strip
    trailing stress digits, ARPABET→IPA, then merge closure+stop pairs and
    collapse adjacent silences. Idempotent on already-clean IPA input.
    """
    ipa_phones = [
        ARPABET_TO_IPA.get(stripped, stripped)
        for stripped in (p.lower().rstrip("0123456789") for p in phones)
    ]
    segs = [(s, e, p) for (s, e), p in zip(phone_timestamps, ipa_phones)]

    merged = []
    i = 0
    while i < len(segs):
        s, e, p = segs[i]
        if p in _IPA_CLOSURE_TO_STOP and i + 1 < len(segs):
            ns, ne, np_ = segs[i + 1]
            if (
                np_ == _IPA_CLOSURE_TO_STOP[p]
                or (p, np_) in _IPA_AFFRICATE_PAIRS
            ):
                merged.append((s, ne, np_))
                i += 2
                continue
        merged.append((s, e, p))
        i += 1

    collapsed = []
    for s, e, p in merged:
        if (
            p in IPA_SILENCE_LABELS
            and collapsed
            and collapsed[-1][2] in IPA_SILENCE_LABELS
        ):
            ps, _, pp = collapsed[-1]
            collapsed[-1] = (ps, e, pp)
            continue
        collapsed.append((s, e, p))

    return (
        [(s, e) for s, e, _ in collapsed],
        [p for _, _, p in collapsed],
    )


def process_gtimit_thai_symbols(
    phone_timestamps: List[Tuple[float, float]],
    phones: List[str],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """GTIMIT Thai pass-through with case normalization.

    Thai phones use a custom romanization that is not ARPABET; no closure
    merging applies. Tones live in a separate field excluded from ``phones``.
    Idempotent.
    """
    return list(phone_timestamps), [p.lower() for p in phones]


# SSNCE Tamil romanization -> IPA. Vowel doubles encode length (aa -> aː);
# *x consonants are retroflex (tx -> ʈ); `eu` is the Tamil centralized high
# vowel; `aɪ` and `n̪d̪` are not in xeuspr ipa_vocab.json and tokenize to
# <unk> until the vocab is extended.
_TAMIL_TO_IPA: Dict[str, str] = {
    "a": "a",   "aa": "aː",
    "i": "i",   "ii": "iː",
    "u": "u",   "uu": "uː",
    "e": "e",   "ee": "eː",
    "o": "o",   "oo": "oː",
    "ai": "aɪ", "eu": "ɨ",
    "k": "k",   "g": "ɡ",
    "c": "t͡ɕ", "j": "d͡ʒ",
    "t": "t̪",  "d": "d̪",
    "tx": "ʈ",  "dx": "ɖ",
    "p": "p",   "b": "b",
    "m": "m",   "n": "n̪",
    "nx": "ɳ",  "nj": "ɲ",  "ng": "ŋ",
    "nd": "n̪d̪",
    "l": "l",   "lx": "ɭ",
    "r": "r",   "rx": "ɽ",  "zh": "ɻ",
    "s": "s",   "sx": "ʂ",  "h": "h",
    "w": "ʋ",   "y": "j",
}


def process_ssnce_symbols(
    phone_timestamps: List[Tuple[float, float]],
    phones: List[str],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """SSNCE Tamil romanization -> IPA via ``_TAMIL_TO_IPA``.

    No closure-stop merge, no silence collapse — SSNCE GT has no silence
    labels and no closures. Idempotent: tokens already in IPA pass through
    via the lowercase fallback.

    Args:
        phone_timestamps: list of ``(start, end)`` tuples (seconds).
        phones: parallel list of Tamil-romanization phone labels.

    Returns:
        Unmodified timestamps; phones mapped to IPA.
    """
    ipa = [_TAMIL_TO_IPA.get(p.lower(), p.lower()) for p in phones]
    return list(phone_timestamps), ipa


# Registry: HuggingFace repo id -> default per-row GT transform.
HF_REPO_TRANSFORMS: Dict[
    str,
    Callable[
        [List[Tuple[float, float]], List[str]],
        Tuple[List[Tuple[float, float]], List[str]],
    ],
] = {
    "changelinglab/timit-segment": process_timit_symbols,
    "changelinglab/buckeye-segment": arpabet_phones_to_ipa,
    "changelinglab/gtimit-l2simple-segment": process_gtimit_arpabet_symbols,
    "changelinglab/gtimit-l2tbnk-segment": process_gtimit_arpabet_symbols,
    "changelinglab/gtimit-l1simple-segment": process_gtimit_arpabet_symbols,
    "changelinglab/gtimit-l1tbnk-segment": process_gtimit_arpabet_symbols,
    "changelinglab/gtimit-tha-segment": process_gtimit_thai_symbols,
    "changelinglab/torgo-segment": process_gtimit_arpabet_symbols,
    "changelinglab/ssnce-segment": process_ssnce_symbols,
}
