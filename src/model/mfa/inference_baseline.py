"""MFA baseline.

This implementation handles multiple cases:
1. MFA baseline for text-independent alignment.
    1A) Phone recognition + alignment with unit=phones
    1B) ASR + alignment with unit=words and a standard dictionary/acoustic model pair
2. MFA topline for text-dependent alignment (forced alignment) without speaker adaptation.

Usage:
    python src/main.py experiment=inference/mfa_en \
        data.hf_repo=changelinglab/timit-segment
"""

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from src.metrics.segmentation_evaluator import SegmentationUnit
from src.model.mfa.utils import (
    _phones_from_mfa_json,
    _save_utterance,
    build_phone_dict,
    ensure_mfa_model,
    mfa_env,
)

_ModelEntry: TypeAlias = tuple[str, str] | Sequence[str]


class MFASingleInference:
    """Per-utterance forced alignment using mfa align_one.

    For monolingual datasets, set ``dictionary`` and ``acoustic_model``
    directly. For multilingual datasets (e.g. VoxAngeles), pass a
    ``language_model_map`` keyed by ISO 639-3 code; utterances whose
    language is not in the map are skipped (return ``[]``).

    Attributes:
        dictionary: Default MFA dictionary name or path.
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: ISO 639-3 → [dictionary, acoustic_model] mapping.
        units: The units to align, either "phones" or "words".
        recognizer: Optional callable for producing transcripts from speech.
    """

    def __init__(
        self,
        dictionary: str = "english_mfa",
        acoustic_model: str = "english_mfa",
        sr: int = 16000,
        language_model_map: dict[str, _ModelEntry] | None = None,
        cache_dir: str | None = None,
        units: str = "phones",
        recognizer: Any | None = None,
    ):
        """Args:
        dictionary: Default MFA dictionary name or path. Ignored when
            units="phones".
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: Optional mapping from ISO 639-3 language code to
            [dictionary, acoustic_model]. Utterances whose language is
            absent from the map are skipped (return []).
        cache_dir: Directory for MFA pretrained models (sets MFA_ROOT_DIR).
            Models are downloaded here if not already present.
        units: The units to align, either "phones" or "words".
        recognizer: Optional callable (PR or ASR model). When provided,
            the waveform is passed through it first and its output replaces
            the dataset's phones/text. Expected to return List[Dict] with
            "predicted_transcript" (slash-separated phones) and
            symbols to be in the acoustic model's phone set.
        """
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.sr = sr
        self.language_model_map = language_model_map
        self.units = units
        self.recognizer = recognizer
        self._env = mfa_env(cache_dir)
        self._download_all_models()

    def _download_all_models(self) -> None:
        """Download all required MFA models if not already present."""
        if self.units == "phones":
            acoustic_models = [self.acoustic_model]
            if self.language_model_map:
                acoustic_models += [
                    v[1] for v in self.language_model_map.values()
                ]
            for am in dict.fromkeys(acoustic_models):
                ensure_mfa_model(am, dictionary=None, env=self._env)
        else:
            pairs: list[tuple[str, str]] = [
                (self.dictionary, self.acoustic_model)
            ]
            if self.language_model_map:
                pairs += [
                    (v[0], v[1]) for v in self.language_model_map.values()
                ]
            for dict_name, model_name in dict.fromkeys(pairs):
                ensure_mfa_model(model_name, dict_name, env=self._env)

    def _run_recognizer(self, sp: torch.Tensor) -> list[str] | str:
        """Run self.recognizer on sp and return phones or word transcript.

        Args:
            sp: 1D float waveform (already sliced to valid length).

        Returns:
            List[str] of phones when units="phones"; word transcript str
            when units="words".
        """
        record = self.recognizer(sp)[0]
        if self.units == "phones":
            raw = record["predicted_transcript"]
            return [p for p in raw.split("/") if p]
        return record["processed_transcript"]

    def _resolve_model(self, language: str | None) -> tuple[str, str] | None:
        """Resolve the model pair for this utterance, or None if unsupported."""
        if self.language_model_map is None:
            return self.dictionary, self.acoustic_model
        if language is None or language not in self.language_model_map:
            return None
        entry = self.language_model_map[language]
        return entry[0], entry[1]

    def __call__(
        self, speech, speech_length, text: str, **kwargs
    ) -> list[SegmentationUnit]:
        """Align one utterance and return phone-level boundaries.

        Args:
            speech: 1D waveform tensor or numpy array.
            speech_length: Number of valid samples.
            text: Utterance word transcript (used when units="words").
            **kwargs: Dataset item extras. ``language`` (ISO 639-3) is used
                when language_model_map is set; ``phones`` (list of phone
                strings) is used when units="phones" and recognizer is None.

        Returns:
            Phone-level SegmentationUnit list with start/end in seconds.
            Empty list if no MFA model is available for this language.
        """
        model = self._resolve_model(kwargs.get("language"))
        if model is None:
            return []
        pretrained_dictionary, acoustic_model = model

        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        sp = speech[: int(speech_length)].float()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            if self.units == "phones":
                phones: list[str] = (
                    self._run_recognizer(sp)
                    if self.recognizer is not None
                    else kwargs.get("phones", [])
                )
                transcript = " ".join(phones)
                build_phone_dict([phones], tmp / "phone_dict.txt")
                dictionary = str(tmp / "phone_dict.txt")
            else:
                transcript = (
                    self._run_recognizer(sp)
                    if self.recognizer is not None
                    else text
                )
                dictionary = pretrained_dictionary

            _save_utterance(sp, transcript, tmp / "item.wav", self.sr)
            subprocess.run(
                [
                    "mfa",
                    "align_one",
                    "item.wav",
                    "item.lab",
                    dictionary,
                    acoustic_model,
                    "item.json",
                    "--output_format",
                    "json",
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
    language_model_map: dict[str, _ModelEntry] | None = None,
    cache_dir: str | None = None,
    units: str = "phones",
    recognizer: Any | None = None,
) -> MFASingleInference:
    """Hydra entry point: instantiate MFASingleInference.

    Args:
        dictionary: Default MFA dictionary name or path.
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: Optional ISO 639-3 → [dictionary, acoustic_model]
            mapping for multilingual datasets.
        cache_dir: Directory for MFA pretrained models. Downloaded if absent.
        units: The units to align, either "phones" or "words".
        recognizer: Optional PR or ASR callable. When set, its output
            transcript replaces the dataset's phones/text field.
    """
    return MFASingleInference(
        dictionary=dictionary,
        acoustic_model=acoustic_model,
        sr=sr,
        language_model_map=language_model_map,
        cache_dir=cache_dir,
        units=units,
        recognizer=recognizer,
    )
