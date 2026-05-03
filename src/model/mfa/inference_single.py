"""MFA single-utterance inference via mfa align_one.

Plugs into the distributed_inference harness (one __call__ per item).
Speaker adaptation is skipped — mfa align_one uses utterance-level CMVN only.
Use src.model.mfa.inference (batch) when per-speaker fMLLR quality matters.

Usage (via distributed_inference harness):
    python src/main.py experiment=inference/mfa_single \\
        data.hf_repo=changelinglab/timit-segment
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.metrics.segmentation_evaluator import SegmentationUnit
from src.model.mfa.utils import _phones_from_mfa_json, _save_utterance


class MFASingleInference:
    """Per-utterance forced alignment using mfa align_one."""

    def __init__(
        self,
        dictionary: str = "english_mfa",
        acoustic_model: str = "english_mfa",
        sr: int = 16000,
    ):
        """Args:
        dictionary: MFA dictionary name or path.
        acoustic_model: MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        """
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.sr = sr

    def __call__(
        self, speech, speech_length, text: str, **kwargs
    ) -> List[SegmentationUnit]:
        """Align one utterance and return phone-level boundaries.

        Args:
            speech: 1D waveform tensor or numpy array.
            speech_length: Number of valid samples.
            text: Utterance transcript.
            **kwargs: Ignored extras forwarded by the dataset item.

        Returns:
            Phone-level SegmentationUnit list with start/end in seconds.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        sp = speech[: int(speech_length)].float()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _save_utterance(sp, text, tmp / "item.wav", self.sr)
            subprocess.run(
                [
                    "mfa", "align_one",
                    "item.wav", "item.lab",
                    self.dictionary, self.acoustic_model,
                    "item.json",
                    "--output_format", "json",
                    "--overwrite",
                ],
                cwd=tmp,
                check=True,
            )
            return _phones_from_mfa_json(tmp / "item.json")


def build_mfa_single_inference(
    dictionary: str = "english_mfa",
    acoustic_model: str = "english_mfa",
    sr: int = 16000,
) -> MFASingleInference:
    """Hydra entry point: instantiate MFASingleInference.

    Args:
        dictionary: MFA dictionary name or path.
        acoustic_model: MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
    """
    return MFASingleInference(
        dictionary=dictionary, acoustic_model=acoustic_model, sr=sr
    )
