"""Shared utilities for MFA inference modules."""

import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

import torch
import torchaudio

from src.metrics.segmentation_evaluator import SegmentationUnit


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
