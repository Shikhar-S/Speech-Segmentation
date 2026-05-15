"""Inference with Phonvec segmentation using distributed_inference harness.

Usage:
    python main.py experiment=inference/phonvec \
       inference.inference_runner.artifact_path=exp/runs/phonvec_tune/timit/phonvec_timit.pt
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.segmentation_evaluator import SegmentationUnit
from src.model.phonvec.model import PhoneClassifier, Segmenter


def _pred_frames_to_units(
    pred_frames, vlen, frame_shift, sr, labels=None
) -> List[SegmentationUnit]:
    """Convert phonvec frame indices to SegmentationUnit time spans."""
    if labels is None:
        times = sorted({int(p) for p in pred_frames if 0 <= int(p) < vlen})
        if len(times) < 2:
            return []
    else:
        peaks = sorted({int(p) for p in pred_frames if 0 < int(p) < vlen})
        times = [0, *peaks, vlen]
    return [
        SegmentationUnit(
            start=times[i] * frame_shift / sr,
            end=times[i + 1] * frame_shift / sr,
            label=None if labels is None else str(labels[i]),
        )
        for i in range(len(times) - 1)
    ]


class PhonvecInference:
    """Per-utterance inference: encoder forward + Segmenter.segment.
    When recognize=True, we also run phone classifier on each predicted segment.
    """

    def __init__(
        self,
        artifact: dict,
        net: nn.Module,
        device: str = "cpu",
        *,
        frame_shift: Optional[int] = None,
        sr: Optional[int] = None,
        recognize: bool = False,
    ):
        """Args:
        artifact: dict produced by ``Segmenter.to_artifact``; must
            contain a ``net`` spec with ``frame_shift`` and ``sr``.
        net: encoder module exposing ``encode(speech, lengths)``.
        device: torch device for the encoder forward.
        frame_shift: optional override of the artifact's saved hop.
        sr: optional override of the artifact's saved sample rate.
        recognize: if True, also classify each segment into a phone label.
        """
        self.net = net.to(device).eval()
        self.device = device
        self.segmenter = Segmenter.from_artifact(artifact)
        self.classifier = (
            PhoneClassifier(self.segmenter.pv_ipa) if recognize else None
        )
        net_spec = artifact["net"]
        self.frame_shift = int(
            frame_shift if frame_shift is not None else net_spec["frame_shift"]
        )
        self.sr = int(sr if sr is not None else net_spec["sr"])

    @torch.no_grad()
    def __call__(
        self, speech, speech_length, **kwargs
    ) -> List[SegmentationUnit]:
        """Args:
        speech: 1D waveform tensor (or numpy array of float32 samples).
        speech_length: int valid sample count.
        **kwargs: ignored extras forwarded by the dataset item.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        sp = speech[: int(speech_length)].float()

        pad = (400 - self.frame_shift) // 2
        sp_padded = F.pad(sp, (pad, pad)).unsqueeze(0).to(self.device)
        feats, feat_lens = self.net.encode(
            sp_padded,
            torch.tensor([sp_padded.shape[-1]], device=self.device),
        )
        vlen = int(feat_lens[0])
        if vlen <= 0:
            return []
        net_feats = feats[0, :vlen].cpu().numpy()
        waveform_np = sp.cpu().numpy()

        pred_frames = self.segmenter.segment(net_feats, waveform_np)
        if self.classifier is None:
            return _pred_frames_to_units(
                pred_frames, vlen, self.frame_shift, self.sr
            )

        peaks = sorted({int(p) for p in pred_frames if 0 < int(p) < vlen})
        seg_bounds = [0, *peaks, vlen]
        silence_mask = self.segmenter.silence_handler.predict_silence_mask(
            net_feats
        )
        labels = self.classifier.classify(net_feats, seg_bounds, silence_mask)
        return _pred_frames_to_units(
            pred_frames, vlen, self.frame_shift, self.sr, labels=labels
        )


def build_phonvec_inference(
    artifact_path: str,
    net: nn.Module,
    device: str = "cuda",
    *,
    frame_shift: Optional[int] = None,
    sr: Optional[int] = None,
    recognize: bool = False,
) -> PhonvecInference:
    """Hydra entry point: load artifact + bind to an encoder net."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    return PhonvecInference(
        artifact,
        net,
        device=device,
        frame_shift=frame_shift,
        sr=sr,
        recognize=recognize,
    )
