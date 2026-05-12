"""Shared utilities for MFA inference modules."""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import filelock
import torch
import torchaudio

from src.core.ipa_utils import ARPABET_TO_IPA
from src.metrics.segmentation_evaluator import SegmentationUnit

# Phones to exclude from MFA transcripts: silence labels + Buckeye non-speech
# markers that survive arpabet_phones_to_ipa as lowercase pass-throughs.
# Membership is case-insensitive; compare via ``is_mfa_silence`` below.
MFA_SILENCE_PHONES: frozenset[str] = frozenset(
    {
        "sil", "sp", "spn", "pau", "h#", "ʔ̞", "epi",
        "vocnoise", "laugh", "noise", "unknown", "iver", "<exclude-name>",
        # Buckeye transcription endpoint markers
        "{b_trans}", "{e_trans}",
    }
)


def is_mfa_silence(phone: str) -> bool:
    """Case-insensitive membership test against ``MFA_SILENCE_PHONES``."""
    return phone.lower() in MFA_SILENCE_PHONES

# IPA conventions in our phone set that differ from MFA english_mfa's phone set.
_IPA_TO_MFA_ENGLISH: dict[str, str] = {
    # Diphthongs / r-colored vowels
    "aɪ": "aj", "oʊ": "ow", "eɪ": "ej", "aʊ": "aw", "ɔɪ": "ɔj",
    "ɜ˞": "ɝ", "ə˞": "ɚ", "ʌ": "ɐ",
    # TIMIT unreleased stops → released equivalents
    "b̚": "b", "d̚": "d", "ɡ̚": "ɡ", "k̚": "k", "p̚": "p", "t̚": "t",
    # Latin 'g' (U+0067) → IPA 'ɡ' (U+0261)
    "g": "ɡ",
    # TIMIT-style glottal stop variant
    "q": "ʔ",
    # Affricates (tie-bar vs plain)
    "d͡ʒ": "dʒ", "t͡ʃ": "tʃ",
    # Koel-specific: r-trill → English approximant
    "r": "ɹ",
    # Koel-specific: voiced h → h
    "ɦ": "h",
    # Koel-specific: British diphthong
    "əʊ": "ow",
    # Koel-specific: syllabic sonorants
    "l̩": "ɫ̩", "ŋ̍": "ŋ",
    # Koel-specific: central vowel
    "ɨ": "ɪ",
    # Koel-specific: devoiced schwa
    "ə̥": "ə",
    # Koel-specific: nasalized vowels → base vowel
    "aɪ̃": "aj", "oʊ̃": "ow", "ĩ": "i", "æ̃": "æ",
    "ɑ̃": "ɑ", "ə̃": "ə", "ɛ̃": "ɛ", "ʊ̃": "ʊ",
    # Koel-specific: aspirated/rare consonants
    "sʰ": "s", "θʰ": "θ", "x": "h", "ɣ": "ɡ", "β": "v",
    # Buckeye glottalized t
    "tq": "t",
    # Buckeye nasalized vowels (n-suffixed ARPABET extensions)
    "ihn": "ɪ", "ahn": "ɐ", "aen": "æ", "ehn": "ɛ", "iyn": "i",
    "aan": "ɑ", "ayn": "aj", "eyn": "ej", "awn": "aw", "aon": "ɔ",
    "oyn": "ɔj", "own": "ow", "uhn": "ʊ", "uwn": "u",
    # Buckeye surface-realization compound tokens (underlying phoneme)
    "ah ix": "ɐ", "ah l": "ɐ", "ah r": "ɐ", "ih l": "ɪ",
    # Bare lowercase ARPABET passthroughs
    "a": "ɐ", "e": "ɛ",
}

# Trailing stress digit on ARPABET vowels (e.g. AH0, EY1).
_ARPABET_STRESS_RE = re.compile(r"[012]$")


def normalize_phones_for_mfa_english(phones: list[str]) -> list[str]:
    """Map our IPA phones to the MFA english_mfa acoustic model phone set.

    Applies after the per-dataset transform (which converts ARPABET to our IPA
    convention). Remaps diphthongs, r-colored vowels, and the strut vowel to
    the notation used by the MFA english_mfa acoustic model.

    Args:
        phones: Phone labels in our IPA convention.

    Returns:
        Phone labels in the MFA english_mfa phone set.
    """
    return [_IPA_TO_MFA_ENGLISH.get(p, p) for p in phones]


def normalize_phones_for_koel(phones: list[str]) -> list[str]:
    """Map raw Koel ARPABET tokens to the MFA english_mfa acoustic model phone set.

    Koel's ``predicted_transcript`` contains raw CTC-collapsed tokens including
    word-boundary markers (``|``) and special tokens (``<PAD>``, ``<UNK>`` …).
    This function filters those out, strips stress digits, converts ARPABET to
    IPA via ``ARPABET_TO_IPA``, and then applies the same IPA→MFA remapping as
    ``normalize_phones_for_mfa_english``.

    Args:
        phones: Raw token strings from Koel's ``predicted_transcript``.

    Returns:
        Phone labels in the MFA english_mfa phone set, with non-speech tokens
        removed.
    """
    result = []
    for p in phones:
        if not p or p.startswith("<") or p == "|":
            continue
        p_clean = _ARPABET_STRESS_RE.sub("", p)
        ipa = ARPABET_TO_IPA.get(p_clean.lower(), p_clean.lower())
        result.append(_IPA_TO_MFA_ENGLISH.get(ipa, ipa))
    return result


def mfa_env(cache_dir: str | None = None) -> dict[str, str]:
    """Return os.environ with MFA_ROOT_DIR set to cache_dir.

    Keeps pretrained models in the project cache instead of ~/Documents/MFA.

    Args:
        cache_dir: MFA model directory. If None, MFA uses its default.

    Returns:
        A copy of os.environ, optionally with MFA_ROOT_DIR added.
    """
    env = os.environ.copy()
    if cache_dir is not None:
        env["MFA_ROOT_DIR"] = str(Path(cache_dir).resolve())
    return env


def _mfa_model_present(model_type: str, name: str, env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["mfa", "model", "list", model_type],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return name in result.stdout


def ensure_mfa_model(
    acoustic_model: str,
    dictionary: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Download acoustic_model and (optionally) dictionary if absent.

    Pass dictionary=None when supplying a custom phone-to-phone dictionary
    file at runtime (units="phones" mode).

    Args:
        acoustic_model: MFA acoustic model name (e.g. ``english_mfa``).
        dictionary: MFA dictionary name, or None to skip download.
        env: Environment dict from ``mfa_env``; defaults to os.environ.
    """
    if env is None:
        env = os.environ.copy()
    targets = [("acoustic", acoustic_model)]
    if dictionary is not None:
        targets.append(("dictionary", dictionary))
    for model_type, name in targets:
        if not _mfa_model_present(model_type, name, env):
            print(f"Downloading MFA {model_type} model: {name}", flush=True)
            subprocess.run(
                ["mfa", "model", "download", model_type, name],
                env=env,
                check=True,
            )


def mfa_extracted_path(
    acoustic_model: str,
    env: dict[str, str] | None = None,
) -> Path:
    """Return the extracted acoustic model directory, extracting if needed.

    Uses a per-model file lock so concurrent workers never race to extract the
    same zip. MFA's Archive.__init__ re-extracts every time when given a model
    name; passing a directory path bypasses that.

    Args:
        acoustic_model: MFA acoustic model name (e.g. ``english_mfa``).
        env: Environment dict from ``mfa_env``; used to locate MFA_ROOT_DIR.

    Returns:
        Path to the extracted acoustic model directory.
    """
    if env is None:
        env = os.environ.copy()
    mfa_root = Path(env.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
    extract_dir = (
        mfa_root / "extracted_models" / "acoustic" / f"{acoustic_model}_acoustic"
    )
    lock_path = mfa_root / f".{acoustic_model}.extract.lock"
    with filelock.FileLock(str(lock_path)):
        if not extract_dir.exists():
            zip_path = (
                mfa_root / "pretrained_models" / "acoustic" / f"{acoustic_model}.zip"
            )
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"MFA model zip not found: {zip_path}. "
                    "Run ensure_mfa_model() first."
                )
            extract_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(zip_path), str(extract_dir))
            files = sorted(extract_dir.iterdir())
            if len(files) == 1 and files[0].is_dir():
                inner = files[0]
                for f in list(inner.iterdir()):
                    shutil.move(str(f), str(extract_dir / f.name))
                inner.rmdir()
    return extract_dir


def build_phone_dict(phones_iter: Iterable[list[str]], dict_path: Path) -> None:
    """Write a phone-to-phone MFA pronunciation dictionary.

    Each unique non-empty phone is written as a one-phone "word" mapping to
    itself (e.g. ``ʃ\\tʃ``), bypassing word-level dictionary lookup. Phone
    symbols must belong to the acoustic model's phone set.

    Args:
        phones_iter: Iterable of phone lists, one per utterance.
        dict_path: Destination path for the dictionary file.
    """
    unique = sorted({p for phones in phones_iter for p in phones if p})
    with open(dict_path, "w", encoding="utf-8") as f:
        for phone in unique:
            f.write(f"{phone}\t{phone}\n")


def _phones_from_mfa_json(json_path: Path) -> list[SegmentationUnit]:
    """Parse an MFA JSON output file into phone-level SegmentationUnits.

    Args:
        json_path: MFA-produced JSON file for a single utterance.

    Returns:
        Phone-level segments with start/end in seconds.
    """
    with open(json_path) as f:
        data = json.load(f)
    entries = data["tiers"]["phones"]["entries"]
    return [
        SegmentationUnit(start=s, end=e, label=label if label else None)
        for s, e, label in entries
    ]


def _save_utterance(
    speech: torch.Tensor, text: str, wav_path: Path, sr: int
) -> None:
    """Write a waveform and its transcript to disk.

    Args:
        speech: 1D float waveform tensor.
        text: Utterance transcript (words or space-joined phones).
        wav_path: Destination WAV path; .lab is written alongside it.
        sr: Sample rate.
    """
    os.makedirs(wav_path.parent, exist_ok=True)
    torchaudio.save(str(wav_path), speech.unsqueeze(0), sr)
    wav_path.with_suffix(".lab").write_text(text + "\n")
