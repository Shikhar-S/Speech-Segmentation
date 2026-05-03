"""Shared utilities for MFA inference modules."""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

from src.metrics.segmentation_evaluator import SegmentationUnit


def mfa_env(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Return an os.environ copy with MFA_ROOT_DIR set to cache_dir.

    MFA reads and writes pretrained models from MFA_ROOT_DIR.  Setting it
    explicitly keeps models out of ~/Documents/MFA and in the project cache.

    Args:
        cache_dir: Directory for MFA models. If None, MFA uses its default
            (~~/Documents/MFA).

    Returns:
        A copy of os.environ, optionally with MFA_ROOT_DIR added.
    """
    env = os.environ.copy()
    if cache_dir is not None:
        env["MFA_ROOT_DIR"] = str(Path(cache_dir).resolve())
    return env


def ensure_mfa_model(
    acoustic_model: str,
    dictionary: str,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Download acoustic_model and dictionary into MFA_ROOT_DIR if absent.

    Uses ``mfa model list`` to check before downloading so already-present
    models are not re-fetched.

    Args:
        acoustic_model: MFA acoustic model name (e.g. ``english_mfa``).
        dictionary: MFA dictionary name (e.g. ``english_mfa``).
        env: Environment dict (from ``mfa_env``).  Uses os.environ if None.
    """
    if env is None:
        env = os.environ.copy()

    def _is_present(model_type: str, name: str) -> bool:
        result = subprocess.run(
            ["mfa", "model", "list", model_type],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        return name in result.stdout

    for model_type, name in [("acoustic", acoustic_model), ("dictionary", dictionary)]:
        if not _is_present(model_type, name):
            print(f"Downloading MFA {model_type} model: {name}", flush=True)
            subprocess.run(
                ["mfa", "model", "download", model_type, name],
                env=env,
                check=True,
            )


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
