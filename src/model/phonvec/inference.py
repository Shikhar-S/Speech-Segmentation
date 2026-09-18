"""Inference with a PhoneModel (phonological_posteriogram) segmenter+recognizer.

Mirrors the upstream reproduction pipeline (``scripts/reproduce_phone_classification.py``
in the phonological-posteriogram repo): extract SSL features → project to the
IPA-view sigmoid posteriogram → segment with the model's hparams → recognize
each segment with a ``Recognizer`` whose ``featmap`` is either (a) the model's
fitted IPA-view featmap (default, when no ``vocab_file`` is provided) or
(b) a per-language ``panphon_featmap`` built at construction time from a
precomputed per-language phone vocabulary (matches upstream ``evaluate_vox``).

``PhoneModel.recognize`` is *not* used because, in phonological_posteriogram@0d50cb2 of the
upstream library, it forwards a stale ``dedup`` kwarg to
``Recognizer.recognize`` and crashes.

Usage:
    python main.py experiment=inference/phonvec \
       inference.inference_runner.model_name_or_path=<HF repo id or local path> \
       inference.inference_runner.vocab_file=configs/inference/phonvec_vocab/<ds>.json
"""

import json
from typing import Dict, List, Optional

import numpy as np
import torch
from phonological_posteriogram import PhoneModel
from phonological_posteriogram.recognizer import Recognizer, panphon_featmap

from src.metrics.types import SegmentationUnit

MEL_SVF_PATCH = {"kwargs": {"left": 2, "right": 1}, "shift": 0}
SILENCE_THRESHOLD = 0.4  # threshold value for predicting a silence frame
SILENCE_GAP_FRAMES = 5  # combine at most these many silence frames if
# they are surrounded by other silence frames


def patch_mel_svf(signals: List[dict]) -> List[dict]:
    """Return ``signals`` with any mel_svf entry's alignment patched.
    A no-op for recipes without mel_svf.
    """
    out = []
    for s in signals:
        s = dict(s)
        if s["name"] == "mel_svf":
            s.update(MEL_SVF_PATCH)
        out.append(s)
    return out


def derive_lang(key: str, lang_sym: Optional[str]) -> str:
    """Routing key for the per-language Recognizer.

    Some prism_preval datasets (e.g. voxangeles) set ``lang_sym`` to ``"unk"``
    because their lang file uses placeholder tags; the real ISO-639-3 code
    lives in the utterance key prefix. Fall back to that when ``lang_sym`` is
    missing or ``"unk"``.
    """
    if lang_sym and lang_sym != "unk":
        return lang_sym
    return (key or "").split("-", 1)[0]


def _load_vocab_recognizers(
    vocab_file: str, model: PhoneModel
) -> Dict[str, Recognizer]:
    """Build one ``Recognizer`` per language listed in the vocab JSON.

    The JSON is the output of ``scripts/build_phonvec_vocab.py``: a dict
    ``{lang: [phone, ...]}``. Each phone list is filtered against panphon's
    inventory (``panphon_featmap`` skips unknown phones internally) and the
    silence token ``"_"`` is always appended so the recognizer can emit it.
    """
    with open(vocab_file) as f:
        lang_phones: dict[str, list[str]] = json.load(f)
    featnames = model.posteriogram.featnames
    return {
        lg: Recognizer(
            featnames=featnames,
            featmap=panphon_featmap([*sorted(ps), "_"], featnames),
        )
        for lg, ps in lang_phones.items()
    }


def _boundary_units(bt_times) -> List[SegmentationUnit]:
    bt = [float(t) for t in bt_times]
    if len(bt) >= 2:
        return [SegmentationUnit(bt[i], bt[i + 1]) for i in range(len(bt) - 1)]
    if len(bt) == 1:
        return [SegmentationUnit(bt[0], bt[0])]
    return []


class PhoneModelInference:
    """Per-utterance inference: explicit upstream pipeline over a waveform."""

    def __init__(
        self,
        model: PhoneModel,
        *,
        recognizers: Optional[Dict[str, Recognizer]] = None,
        fixed_recognizer: Optional[Recognizer] = None,
        oracle: bool = False,
        boundaries_only: bool = False,
    ):
        """Args:
        model: a loaded ``PhoneModel`` (encoder + posteriogram + hparams).
        recognizers: optional ``{lang: Recognizer}`` map. When provided, each
            utterance is routed to the recognizer for its derived language.
            When ``None``, every utterance is decoded with the model's own
            recognizer (its IPA-view featmap).
        fixed_recognizer: optional single ``Recognizer`` used for every
            utterance, overriding both ``recognizers`` routing and the model's
            own recognizer. Used for the panphon-unrestricted decode (full
            PanPhon inventory, no featmap).
        oracle: when True, recognize each ground-truth segment using the GT
            boundaries (``phone_timestamps`` kwarg) instead of the predicted
            segments.
        boundaries_only: when True (segmentation), emit just the segmenter's
            boundaries — skip the recognizer and the label-based outer-silence
            trimming so the predicted boundary set never depends on recognition
            labels. Ignored in oracle mode.
        """
        self.model = model
        self.recognizers = recognizers
        self.fixed_recognizer = fixed_recognizer
        self.oracle = oracle
        self.boundaries_only = boundaries_only
        self.segmenter = model.segmenter(
            {
                "combined_signals": patch_mel_svf(
                    model.hparams["combined_signals"]
                ),
                "silence_threshold": SILENCE_THRESHOLD,
                "silence_gap_frames": SILENCE_GAP_FRAMES,
            }
        )
        self.sr = model.net_spec["sr"]
        self.frame_shift = model.encoder.stride_size

    @torch.no_grad()
    def __call__(
        self, speech, speech_length, **kwargs
    ) -> List[SegmentationUnit]:
        """Args:
        speech: 1D waveform tensor (or numpy array of float32 samples) at the
            model's expected sample rate.
        speech_length: int valid sample count.
        **kwargs: passthrough fields from the dataset.
            ``key``/``utt_id``, ``lang_sym``/``language`` : read for per-language recognizer routing
            ``phone_timestamps``: supplies the GT boundaries in oracle mode
        """
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()
        waveform = np.asarray(speech, dtype=np.float32)[: int(speech_length)]

        feats = self.model.extract_features(waveform)
        post = self.model.posteriogram.project(feats, view="ipa", act="sigmoid")

        if self.oracle:
            # Oracle segmentation: pool each GT segment at its own center frame.
            # Pass every GT boundary so the recognizer brackets each segment
            # exactly; it always adds an outer [0, end_t] bracket, which yields
            # a leading and/or trailing extra segment whenever the GT does not
            # span the whole utterance (silence-free datasets such as
            # VoxAngeles). Those extras are dropped after recognition so the
            # labels stay 1:1 with the GT segments (contiguous GT assumed).
            timestamps = kwargs.get("phone_timestamps")
            if not timestamps:
                return []
            end_t = len(post) * self.frame_shift / self.sr
            edges = [s for s, _ in timestamps] + [
                timestamps[-1][1]
            ]  # edges = boundaries between tokens
            bt_times = np.asarray(
                [b for b in edges if 0.0 < b < end_t], dtype=float
            )
        else:
            bt_frames = np.asarray(self.segmenter.segment(feats, waveform))
            bt_times = self.model.encoder.frame_to_time(bt_frames)
            if self.boundaries_only:
                return _boundary_units(bt_times)

        if self.fixed_recognizer is not None:
            recognizer = self.fixed_recognizer
        elif self.recognizers is None:
            recognizer = self.model.recognizer
        else:
            # need two gets to handle conventions for segmentation style
            #  and prism recog datasets
            lang = derive_lang(
                kwargs.get("key") or kwargs.get("utt_id"),
                kwargs.get("lang_sym") or kwargs.get("language"),
            )
            recognizer = self.recognizers.get(lang)
            if recognizer is None:
                # Unknown language at inference time: emit nothing rather than
                # routing to an arbitrary other-language recognizer.
                return []

        labels = recognizer.recognize(
            post,
            bt_times,
            sr=self.sr,
            frame_shift=self.frame_shift,
        )
        if not labels:
            return []

        if self.oracle:
            # Drop the leading bracket ([0, first GT start]) when the GT does
            # not begin at 0, then keep one label per GT segment, positionally
            # aligned so scripts/eval_recognition.py can score against
            # ``phones``. The trailing extra (if any) falls off the n-slice.
            lead = 1 if timestamps[0][0] > 0.0 else 0
            labels = labels[lead : lead + len(timestamps)]
            assert len(labels) == len(timestamps)
            return [
                SegmentationUnit(float(s), float(e), labels[i])
                for i, (s, e) in enumerate(timestamps)
            ]

        end_t = len(waveform) / self.sr
        bt_full = np.concatenate([[0.0], bt_times, [end_t]])
        units = [
            SegmentationUnit(
                float(bt_full[i]), float(bt_full[i + 1]), labels[i]
            )
            for i in range(len(labels))
        ]
        # Drop the [0, bt_times[0]] / [bt_times[-1], end_t] silence brackets
        # the recognizer needs internally for label alignment. Without this,
        # downstream segmentation eval counts 0.0 and end_t as predicted
        # boundaries, inflating pred count by 2 per utterance.
        while units and units[0].label == "_":
            units.pop(0)
        while units and units[-1].label == "_":
            units.pop()
        return units


def build_phonvec_inference(
    model_name_or_path: str,
    device: str = "cuda",
    *,
    filename: str = "model.pt",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    vocab_file: Optional[str] = None,
    panphon_unrestricted: bool = False,
    oracle: bool = False,
    boundaries_only: bool = False,
) -> PhoneModelInference:
    """Hydra entry point: load a PhoneModel via ``from_pretrained`` and wrap it.

    ``model_name_or_path`` is a HuggingFace repo id, a local directory holding
    ``filename``, or a path to the artifact file itself. When ``vocab_file`` is
    set, per-language Recognizers are built from its JSON contents and routed
    per utterance. When ``panphon_unrestricted`` is set, a single Recognizer
    over the full PanPhon inventory (no featmap) decodes every utterance,
    overriding ``vocab_file``; pass ``vocab_file=null`` with it. When ``oracle``
    is set, ground-truth boundaries drive recognition instead of the model
    segmenter. When ``boundaries_only`` is set (segmentation), only the
    segmenter's boundaries are emitted (no recognizer).
    """
    model = PhoneModel.from_pretrained(
        model_name_or_path,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
    )
    recognizers = (
        _load_vocab_recognizers(vocab_file, model) if vocab_file else None
    )
    fixed_recognizer = (
        Recognizer(featnames=model.posteriogram.featnames)
        if panphon_unrestricted
        else None
    )
    return PhoneModelInference(
        model,
        recognizers=recognizers,
        fixed_recognizer=fixed_recognizer,
        oracle=oracle,
        boundaries_only=boundaries_only,
    )
