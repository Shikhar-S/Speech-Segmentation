"""Inference with a PhoneModel (phonological_posteriogram) segmenter+recognizer.

Wraps :class:`phonological_posteriogram.PhoneModel` for the
``distributed_inference`` harness. The model is fully self-contained: it loads
its own SSL encoder and runs encode -> segment -> recognize end to end, so no
external ``net`` is needed.

Usage:
    python main.py experiment=inference/phonvec \
       inference.inference_runner.model_name_or_path=<HF repo id or local path>
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
from phonological_posteriogram import PhoneModel

from src.metrics.segmentation_evaluator import SegmentationUnit


class PhoneModelInference:
    """Per-utterance inference: ``PhoneModel.recognize`` over a waveform."""

    def __init__(
        self,
        model: PhoneModel,
        *,
        lang: Optional[str] = None,
        phoible_id: Optional[int] = None,
        phoneme: bool = False,
        vocab: Optional[Sequence[str]] = None,
        dedup: bool = False,
    ):
        """Args:
        model: a loaded ``PhoneModel`` (encoder + posteriogram + hparams).
        lang / phoible_id / phoneme: optional Phoible-inventory vocab
            constraint forwarded to ``PhoneModel.recognize``.
        vocab: optional explicit phone vocabulary (mutually exclusive with the
            Phoible constraint).
        dedup: merge consecutive segments sharing a label (default False).
        """
        self.model = model
        self.lang = lang
        self.phoible_id = phoible_id
        self.phoneme = phoneme
        self.vocab = vocab
        self.dedup = dedup

    @torch.no_grad()
    def __call__(
        self, speech, speech_length, **kwargs
    ) -> List[SegmentationUnit]:
        """Args:
        speech: 1D waveform tensor (or numpy array of float32 samples) at the
            model's expected sample rate.
        speech_length: int valid sample count.
        **kwargs: ignored extras forwarded by the dataset item.
        """
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()
        waveform = np.asarray(speech, dtype=np.float32)[: int(speech_length)]
        units = self.model.recognize(
            waveform,
            lang=self.lang,
            phoible_id=self.phoible_id,
            phoneme=self.phoneme,
            vocab=self.vocab,
            dedup=self.dedup,
        )
        return [SegmentationUnit(u.start, u.end, u.label) for u in units]


def build_phonvec_inference(
    model_name_or_path: str,
    device: str = "cuda",
    *,
    filename: str = "model.pt",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lang: Optional[str] = None,
    phoible_id: Optional[int] = None,
    phoneme: bool = False,
    vocab: Optional[Sequence[str]] = None,
    dedup: bool = False,
) -> PhoneModelInference:
    """Hydra entry point: load a PhoneModel via ``from_pretrained`` and wrap it.

    ``model_name_or_path`` is a HuggingFace repo id, a local directory holding
    ``filename``, or a path to the artifact file itself.
    """
    model = PhoneModel.from_pretrained(
        model_name_or_path,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
    )
    return PhoneModelInference(
        model,
        lang=lang,
        phoible_id=phoible_id,
        phoneme=phoneme,
        vocab=vocab,
        dedup=dedup,
    )
