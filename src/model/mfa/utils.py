"""Shared utilities for MFA inference modules."""

import json
from pathlib import Path
from typing import List

import torch
import torchaudio

from src.metrics.segmentation_evaluator import SegmentationUnit


def _phones_from_mfa_json(json_path: Path) -> List[SegmentationUnit]:
    """Parse MFA JSON output for one utterance into phone-level SegmentationUnits.

    Args:
        json_path: Path to the MFA-produced JSON file for a single utterance.

    Returns:
        Phone-level segments with start/end in seconds and label.
        Silence intervals (empty/sil/spn labels) are included; stripping is
        left to eval_segmentation.py --strip-outer-silences.
    """
    with open(json_path) as f:
        data = json.load(f)
    tiers = data["tiers"]
    # tiers may be a list or dict depending on MFA version
    tier_iter = tiers.values() if isinstance(tiers, dict) else tiers
    phones_tier = next(t for t in tier_iter if t["name"].endswith(" - phones"))
    return [
        SegmentationUnit(start=s, end=e, label=label if label else None)
        for s, e, label in phones_tier["entries"]
    ]


def _save_utterance(
    speech: torch.Tensor, text: str, wav_path: Path, sr: int
) -> None:
    """Write a waveform and its transcript to disk.

    Args:
        speech: 1D float waveform tensor.
        text: Utterance transcript.
        wav_path: Destination WAV path; the .lab file is written alongside it.
        sr: Sample rate.
    """
    torchaudio.save(str(wav_path), speech.unsqueeze(0), sr)
    wav_path.with_suffix(".lab").write_text(text + "\n")
