"""Per-row GT preprocessing for SegmentationDataset.

``HF_REPO_TRANSFORMS`` maps dataset name -> per-row cleanup applied inside
``SegmentationDataset.__getitem__`` to ``(phone_timestamps, phones)``.
Must be idempotent.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

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


def _segs_from_zip(
    phone_timestamps: List[Tuple[float, float]],
    phones: List[str],
) -> List[Tuple[float, float, str]]:
    return [(s, e, p) for (s, e), p in zip(phone_timestamps, phones)]


def _split_segs(
    segs: List[Tuple[float, float, str]],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    return (
        [(s, e) for s, e, _ in segs],
        [p for _, _, p in segs],
    )


def _row_segs(row: dict) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Return (phone_timestamps, phones)."""
    return (
        list(zip(row["phone_starts"], row["phone_ends"])),
        list(row["phones"]),
    )


def _merge_closures_to_stops(
    segs: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """Fold each IPA closure into the following stop/affricate release.

    Standalone closures are relabeled to their bare stop.
    """
    merged: List[Tuple[float, float, str]] = []
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
        merged.append((s, e, _IPA_CLOSURE_TO_STOP.get(p, p)))
        i += 1
    return merged


def _collapse_adjacent_silence(
    segs: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """Coalesce consecutive ``IPA_SILENCE_LABELS`` segments into one interval."""
    collapsed: List[Tuple[float, float, str]] = []
    for s, e, p in segs:
        if (
            p in IPA_SILENCE_LABELS
            and collapsed
            and collapsed[-1][2] in IPA_SILENCE_LABELS
        ):
            ps, _, pp = collapsed[-1]
            collapsed[-1] = (ps, e, pp)
            continue
        collapsed.append((s, e, p))
    return collapsed


def process_timit_symbols(
    row: dict,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """TIMIT GT pipeline: ARPABET→IPA + closure->stop merge + silence collapse.

    All three steps are idempotent. Labels already in IPA pass through the
    ARPABET map unchanged via the lowercase fallback.
    """
    phone_timestamps, phones = _row_segs(row)
    ipa = [ARPABET_TO_IPA.get(p.lower(), p.lower()) for p in phones]
    segs = _segs_from_zip(phone_timestamps, ipa)
    segs = _merge_closures_to_stops(segs)
    segs = _collapse_adjacent_silence(segs)
    return _split_segs(segs)


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
    row: dict,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """GTIMIT English ARPABET → IPA + closure->stop merge + silence collapse.

    Handles both lowercase-no-stress (l2simple) and uppercase-with-stress
    (l2tbnk, l1simple, l1tbnk) variants in a single pass: lowercase, strip
    trailing stress digits, ARPABET→IPA, then closure-stop merge and
    adjacent-silence collapse. Idempotent on already-clean IPA input.
    """
    phone_timestamps, phones = _row_segs(row)
    ipa = [
        ARPABET_TO_IPA.get(stripped, stripped)
        for stripped in (p.lower().rstrip("0123456789") for p in phones)
    ]
    segs = _segs_from_zip(phone_timestamps, ipa)
    segs = _merge_closures_to_stops(segs)
    segs = _collapse_adjacent_silence(segs)
    return _split_segs(segs)


# --- Buckeye ---------------------------------------------------------------
# Non-speech tokens that appear in Buckeye GT (case-preserving).
_BUCKEYE_NONSPEECH: frozenset = frozenset(
    {"VOCNOISE", "IVER", "UNKNOWN", "LAUGH", "NOISE"}
)
# Buckeye allophone pre-remap (ARPABET-space): glottalized /t/ -> 'q' (-> ʔ).
_BUCKEYE_PRE_REMAP: Dict[str, str] = {"tq": "q"}
# ARPABET vowels that Buckeye nasalization-suffixes with trailing 'n'
# (e.g., 'ihn' = nasalized IH). We drop the 'n' and score against bare vowel.
_BUCKEYE_BARE_VOWELS: frozenset = frozenset(
    {
        "aa",
        "ae",
        "ah",
        "ao",
        "aw",
        "ay",
        "eh",
        "ey",
        "ih",
        "iy",
        "ow",
        "oy",
        "uh",
        "uw",
    }
)


def _buckeye_pre_remap(phone: str) -> str:
    """Map Buckeye-specific tokens before standard ARPABET→IPA.

    - Non-speech markers (VOCNOISE/IVER/UNKNOWN/LAUGH/NOISE and any
      ``{...}`` or ``<...>`` wrapped tag) -> ``"sil"``.
    - Glottalized /t/ (``tq``) -> ARPABET ``q`` (mapped to IPA ʔ downstream).
    - Buckeye nasalized vowels (e.g., ``ihn``, ``ahn``) -> bare ARPABET vowel.
    """
    if phone in _BUCKEYE_NONSPEECH:
        return "sil"
    if phone.startswith("{") and phone.endswith("}"):
        return "sil"
    if phone.startswith("<") and phone.endswith(">"):
        return "sil"
    lower = phone.lower()
    if lower in _BUCKEYE_PRE_REMAP:
        return _BUCKEYE_PRE_REMAP[lower]
    if lower.endswith("n") and lower[:-1] in _BUCKEYE_BARE_VOWELS:
        return lower[:-1]
    return phone


def process_buckeye_symbols(
    row: dict,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Buckeye: non-speech -> sil, glottalized /t/ -> ʔ, nasal vowels ->
    bare vowel, then ARPABET → IPA, then adjacent-silence collapse.

    No closure-stop merge: Buckeye GT does not annotate closures.
    """
    phone_timestamps, phones = _row_segs(row)
    pre = [_buckeye_pre_remap(p) for p in phones]
    ipa = [ARPABET_TO_IPA.get(p.lower(), p.lower()) for p in pre]
    segs = _segs_from_zip(phone_timestamps, ipa)
    segs = _collapse_adjacent_silence(segs)
    return _split_segs(segs)


# --- gTIMIT Thai -----------------------------------------------------------
# Thai romanization → IPA. Case-preserving — Thai uses case to distinguish
# phonemes (``N`` = ŋ, ``n`` = n; ``W`` = ɯ, ``w`` = w; etc.). Long vowels
# use ``ː``; aspiration uses ``ʰ``; the palatal affricate uses the tie-bar.
_THAI_TO_IPA: Dict[str, str] = {
    # vowels — short
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    "E": "ɛ",
    "O": "ɔ",
    "W": "ɯ",
    "@": "ə",
    # vowels — long
    "aa": "aː",
    "ii": "iː",
    "uu": "uː",
    "ee": "eː",
    "oo": "oː",
    "EE": "ɛː",
    "OO": "ɔː",
    "WW": "ɯː",
    "@@": "əː",
    # nasals
    "m": "m",
    "n": "n",
    "N": "ŋ",
    # stops — voiceless unaspirated
    "p": "p",
    "t": "t",
    "k": "k",
    # stops — voiceless aspirated
    "ph": "pʰ",
    "th": "tʰ",
    "kh": "kʰ",
    # stops — voiced
    "b": "b",
    "d": "d",
    # affricates
    "c": "t͡ɕ",
    "chh": "t͡ɕʰ",
    # fricatives
    "f": "f",
    "s": "s",
    "h": "h",
    # approximants
    "w": "w",
    "j": "j",
    "l": "l",
    "r": "r",
}


def process_gtimit_thai_symbols(
    row: dict,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """GTIMIT Thai: Thai romanization → IPA via ``_THAI_TO_IPA`` + silence
    collapse. Case-preserving lookup; unmapped tokens (incl. ``sil``) pass
    through and are resolved downstream by ``_canonicalize_phone``.
    """
    phone_timestamps, phones = _row_segs(row)
    ipa = [_THAI_TO_IPA.get(p, p) for p in phones]
    segs = _segs_from_zip(phone_timestamps, ipa)
    segs = _collapse_adjacent_silence(segs)
    return _split_segs(segs)


_THAI_TONE_DIACRITIC: Dict[str, str] = {
    "Tone_M": "˧",
    "Tone_L": "˨˩",
    "Tone_F": "˥˩",
    "Tone_R": "˩˩˦",
    "Tone_H": "˦˥",
}
# Above tones mappings are only appended to the vowels below
_THAI_VOWELS = frozenset(
    _THAI_TO_IPA[k]
    for k in (
        "a",
        "aa",
        "i",
        "ii",
        "u",
        "uu",
        "e",
        "ee",
        "o",
        "oo",
        "E",
        "EE",
        "O",
        "OO",
        "W",
        "WW",
        "@",
        "@@",
    )
)


def process_gtimit_thai_with_tones(
    row: dict,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """GTIMIT Thai with tone fusion: Thai romanization → tone-marked IPA.

    Same as ``process_gtimit_thai_symbols`` but appends the syllable tone
    diacritic to each vowel using the row's parallel ``tones`` field (one
    label per phone). Consonants and silence are left toneless.
    """
    phone_timestamps, phones = _row_segs(row)
    ipa = []
    for p, t in zip(phones, row["tones"]):
        x = _THAI_TO_IPA.get(p, p)
        if x in _THAI_VOWELS and t in _THAI_TONE_DIACRITIC:
            x += _THAI_TONE_DIACRITIC[t]
        ipa.append(x)
    segs = _segs_from_zip(phone_timestamps, ipa)
    segs = _collapse_adjacent_silence(segs)
    return _split_segs(segs)


# --- SSNCE Tamil -----------------------------------------------------------
# Vowel doubles encode length (aa -> aː); *x consonants are retroflex (tx
# -> ʈ); ``eu`` is the Tamil centralized high vowel; ``aɪ`` and ``n̪d̪`` are
# not in xeuspr ipa_vocab.json and tokenize to <unk> until the vocab is
# extended.
_TAMIL_TO_IPA: Dict[str, str] = {
    "a": "a",
    "aa": "aː",
    "i": "i",
    "ii": "iː",
    "u": "u",
    "uu": "uː",
    "e": "e",
    "ee": "eː",
    "o": "o",
    "oo": "oː",
    "ai": "aɪ",
    "eu": "ɨ",
    "k": "k",
    "g": "ɡ",
    "c": "t͡ɕ",
    "j": "d͡ʒ",
    "t": "t̪",
    "d": "d̪",
    "tx": "ʈ",
    "dx": "ɖ",
    "p": "p",
    "b": "b",
    "m": "m",
    "n": "n̪",
    "nx": "ɳ",
    "nj": "ɲ",
    "ng": "ŋ",
    "nd": "n̪d̪",
    "l": "l",
    "lx": "ɭ",
    "r": "r",
    "rx": "ɽ",
    "zh": "ɻ",
    "s": "s",
    "sx": "ʂ",
    "h": "h",
    "w": "ʋ",
    "y": "j",
}


def process_ssnce_symbols(
    row: dict,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """SSNCE Tamil romanization -> IPA via ``_TAMIL_TO_IPA``.

    No closure-stop merge, no silence collapse — SSNCE GT has no silence
    labels and no closures. Idempotent: tokens already in IPA pass through
    via the lowercase fallback.
    """
    phone_timestamps, phones = _row_segs(row)
    ipa = [_TAMIL_TO_IPA.get(p.lower(), p.lower()) for p in phones]
    return list(phone_timestamps), ipa


# Registry: HuggingFace repo id -> default per-row GT transform.
# Each transform takes a dataset ``row`` and returns ``(phone_timestamps,
# phones)``; it reads whatever fields it needs (the THA tone transform also
# reads ``row["tones"]``).
HF_REPO_TRANSFORMS: Dict[
    str,
    Callable[[dict], Tuple[List[Tuple[float, float]], List[str]]],
] = {
    "timit-segment": process_timit_symbols,
    "buckeye-segment": process_buckeye_symbols,
    "gtimit-l2simple-segment": process_gtimit_arpabet_symbols,
    "gtimit-l2tbnk-segment": process_gtimit_arpabet_symbols,
    "gtimit-l1simple-segment": process_gtimit_arpabet_symbols,
    "gtimit-l1tbnk-segment": process_gtimit_arpabet_symbols,
    "gtimit-tha-segment": process_gtimit_thai_symbols,
    "torgo-segment": process_gtimit_arpabet_symbols,
    "ssnce-segment": process_ssnce_symbols,
}



def env_var_for(name: str) -> str:
    """``timit-segment`` -> ``SEG_REPO_TIMIT``: the env var holding its Hub repo id."""
    return "SEG_REPO_" + name.removesuffix("-segment").replace("-", "_").upper()


def repo_for(name: str) -> str:
    """Hub repo id for a registered dataset name, honouring ``SEG_REPO_<NAME>``."""
    return os.environ.get(env_var_for(name), f"changelinglab/{name}")


def transform_for_repo(hf_repo: str) -> Optional[Callable]:
    """Transform for an HF repo id: match the ``SEG_REPO_*`` override first,
    then the canonical ``<org>/<name>`` form."""
    for name, fn in HF_REPO_TRANSFORMS.items():
        if hf_repo == repo_for(name) or hf_repo.rsplit("/", 1)[-1] == name:
            return fn
    return None
