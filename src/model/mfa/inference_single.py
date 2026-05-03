"""MFA single-utterance inference via mfa align_one.

Plugs into the distributed_inference harness (one __call__ per item).
Speaker adaptation is skipped — mfa align_one uses utterance-level CMVN only.
Use src.model.mfa.inference (batch) when per-speaker fMLLR quality matters.

Usage (via distributed_inference harness):
    python src/main.py experiment=inference/mfa_en \\
        data.hf_repo=changelinglab/timit-segment
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.metrics.segmentation_evaluator import SegmentationUnit
from src.model.mfa.utils import (
    _phones_from_mfa_json,
    _save_utterance,
    ensure_mfa_model,
    mfa_env,
)

# Type for one entry in language_model_map: [dictionary, acoustic_model]
_ModelEntry = Union[Tuple[str, str], Sequence[str]]


class MFASingleInference:
    """Per-utterance forced alignment using mfa align_one.

    For monolingual datasets, set ``dictionary`` and ``acoustic_model``
    directly.  For multilingual datasets (e.g. VoxAngeles), pass a
    ``language_model_map`` keyed by ISO 639-3 code; utterances whose
    language is not in the map are skipped (return ``[]``).
    """

    def __init__(
        self,
        dictionary: str = "english_mfa",
        acoustic_model: str = "english_mfa",
        sr: int = 16000,
        language_model_map: Optional[Dict[str, _ModelEntry]] = None,
        cache_dir: Optional[str] = None,
    ):
        """Args:
        dictionary: Default MFA dictionary name or path (used when
            language_model_map is None or language is absent from kwargs).
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: Optional mapping from ISO 639-3 language code to
            [dictionary, acoustic_model].  When provided, the model is
            selected per utterance from kwargs["language"].  Utterances
            whose language is absent from the map are skipped.
        cache_dir: Directory for MFA pretrained models (sets MFA_ROOT_DIR).
            Models are downloaded here if not already present.
        """
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.sr = sr
        self.language_model_map = language_model_map
        self._env = mfa_env(cache_dir)
        self._download_all_models()

    def _download_all_models(self) -> None:
        """Ensure every model referenced by this instance is downloaded."""
        pairs: List[Tuple[str, str]] = [(self.dictionary, self.acoustic_model)]
        if self.language_model_map:
            pairs += [(v[0], v[1]) for v in self.language_model_map.values()]
        # deduplicate while preserving order
        for dictionary, acoustic_model in dict.fromkeys(pairs):
            ensure_mfa_model(acoustic_model, dictionary, self._env)

    def _resolve_model(self, language: Optional[str]) -> Optional[Tuple[str, str]]:
        """Return (dictionary, acoustic_model) for this utterance, or None to skip."""
        if self.language_model_map is None:
            return self.dictionary, self.acoustic_model
        if language is None or language not in self.language_model_map:
            return None
        entry = self.language_model_map[language]
        return entry[0], entry[1]

    def __call__(
        self, speech, speech_length, text: str, **kwargs
    ) -> List[SegmentationUnit]:
        """Align one utterance and return phone-level boundaries.

        Args:
            speech: 1D waveform tensor or numpy array.
            speech_length: Number of valid samples.
            text: Utterance transcript.
            **kwargs: Dataset item extras; ``language`` (ISO 639-3) is used
                when language_model_map is set.

        Returns:
            Phone-level SegmentationUnit list with start/end in seconds.
            Empty list if no MFA model is available for this language.
        """
        model = self._resolve_model(kwargs.get("language"))
        if model is None:
            return []
        dictionary, acoustic_model = model

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
                    dictionary, acoustic_model,
                    "item.json",
                    "--output_format", "json",
                    "--overwrite",
                ],
                cwd=tmp,
                env=self._env,
                check=True,
            )
            return _phones_from_mfa_json(tmp / "item.json")


def build_mfa_single_inference(
    dictionary: str = "english_mfa",
    acoustic_model: str = "english_mfa",
    sr: int = 16000,
    language_model_map: Optional[Dict[str, _ModelEntry]] = None,
    cache_dir: Optional[str] = None,
) -> MFASingleInference:
    """Hydra entry point: instantiate MFASingleInference.

    Args:
        dictionary: Default MFA dictionary name or path.
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: Optional ISO 639-3 → [dictionary, acoustic_model]
            mapping for multilingual datasets.
        cache_dir: Directory for MFA pretrained models.  Downloaded here if absent.
    """
    return MFASingleInference(
        dictionary=dictionary,
        acoustic_model=acoustic_model,
        sr=sr,
        language_model_map=language_model_map,
        cache_dir=cache_dir,
    )
