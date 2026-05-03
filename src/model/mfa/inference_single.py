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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
        use_phones: bool = True,
        recognizer: Optional[Any] = None,
    ):
        """Args:
        dictionary: Default MFA dictionary name or path.  Ignored when
            use_phones=True.
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: Optional mapping from ISO 639-3 language code to
            [dictionary, acoustic_model].  When provided, the model is
            selected per utterance from kwargs["language"].  Utterances
            whose language is absent from the map are skipped.
        cache_dir: Directory for MFA pretrained models (sets MFA_ROOT_DIR).
            Models are downloaded here if not already present.
        use_phones: If True (default), write the dataset's phone sequence to
            .lab and generate a per-utterance phone-to-phone dictionary,
            bypassing word-level dictionary lookup.  Requires the phone
            symbols to be in the acoustic model's phone set.  If False, use
            the word transcript and the named dictionary.
        recognizer: Optional callable (PR or ASR model).  When provided,
            the waveform is passed through it first and its output transcript
            replaces the dataset's phones/text.  Expected to return
            List[Dict] with "predicted_transcript" (slash-separated phones)
            and "processed_transcript" (word text); use_phones controls
            which field is used.
        """
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.sr = sr
        self.language_model_map = language_model_map
        self.use_phones = use_phones
        self.recognizer = recognizer
        self._env = mfa_env(cache_dir)
        self._download_all_models()

    def _download_all_models(self) -> None:
        """Ensure every acoustic model (and dictionary if not use_phones) is downloaded."""
        if self.use_phones:
            # Provide our own dictionary file at runtime; only acoustic models needed.
            acoustic_models = [self.acoustic_model]
            if self.language_model_map:
                acoustic_models += [v[1] for v in self.language_model_map.values()]
            for am in dict.fromkeys(acoustic_models):
                ensure_mfa_model(am, dictionary=None, env=self._env)
        else:
            pairs: List[Tuple[str, str]] = [(self.dictionary, self.acoustic_model)]
            if self.language_model_map:
                pairs += [(v[0], v[1]) for v in self.language_model_map.values()]
            for dictionary, acoustic_model in dict.fromkeys(pairs):
                ensure_mfa_model(acoustic_model, dictionary, env=self._env)

    def _run_recognizer(
        self, sp: torch.Tensor
    ) -> Union[List[str], str]:
        """Run self.recognizer on sp and return phones list or word transcript.

        Args:
            sp: 1D float waveform (already sliced to valid length).

        Returns:
            List[str] of phones when use_phones=True; word transcript str
            when use_phones=False.
        """
        record = self.recognizer(sp)[0]
        if self.use_phones:
            raw = record["predicted_transcript"]
            return [p for p in raw.split("/") if p]
        return record["processed_transcript"]

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
            text: Utterance word transcript (used when use_phones=False).
            **kwargs: Dataset item extras.  ``language`` (ISO 639-3) is used
                when language_model_map is set; ``phones`` (list of phone
                strings) is used when use_phones=True.

        Returns:
            Phone-level SegmentationUnit list with start/end in seconds.
            Empty list if no MFA model is available for this language.
        """
        model = self._resolve_model(kwargs.get("language"))
        if model is None:
            return []
        _, acoustic_model = model
        # When use_phones=False, use the pretrained dictionary from the model pair.
        pretrained_dictionary = model[0]

        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        sp = speech[: int(speech_length)].float()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            if self.use_phones:
                if self.recognizer is not None:
                    phones: List[str] = self._run_recognizer(sp)
                else:
                    phones = kwargs.get("phones", [])
                transcript = " ".join(phones)
                build_phone_dict([phones], tmp / "phone_dict.txt")
                dictionary = str(tmp / "phone_dict.txt")
            else:
                if self.recognizer is not None:
                    transcript = self._run_recognizer(sp)
                else:
                    transcript = text
                dictionary = pretrained_dictionary

            _save_utterance(sp, transcript, tmp / "item.wav", self.sr)
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
    use_phones: bool = True,
    recognizer: Optional[Any] = None,
) -> MFASingleInference:
    """Hydra entry point: instantiate MFASingleInference.

    Args:
        dictionary: Default MFA dictionary name or path.
        acoustic_model: Default MFA acoustic model name or path.
        sr: Expected sample rate of incoming waveforms.
        language_model_map: Optional ISO 639-3 → [dictionary, acoustic_model]
            mapping for multilingual datasets.
        cache_dir: Directory for MFA pretrained models.  Downloaded if absent.
        use_phones: If True (default), align phone sequences directly instead
            of word transcripts.
        recognizer: Optional PR or ASR callable.  When set, its output
            transcript replaces the dataset's phones/text field.
    """
    return MFASingleInference(
        dictionary=dictionary,
        acoustic_model=acoustic_model,
        sr=sr,
        language_model_map=language_model_map,
        cache_dir=cache_dir,
        use_phones=use_phones,
        recognizer=recognizer,
    )
