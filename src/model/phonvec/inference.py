"""Inference with Phonvec segmentation using distributed_inference harness.

# TODO(shikhar): keep inference.py at the same place, baselines are at recipe level?
# where should mfa inference be?
To use this, put the following in config:
    inference_runner:
      _target_: src.model.phonvec.inference.build_phonvec_inference
      artifact_path: path/to/phonvec_artifact.pt
      device: cuda
      # Optional overrides; default to the artifact's saved values:
      # frame_shift: 320
      # sr: 16000
      net:
        _target_: src.model.wavlm.builders.build_wavlm_model
        hf_repo: microsoft/wavlm-large
        encoder_layer: 24
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.segmentation_evaluator import SegmentationUnit
from src.model.phonvec.model import Segmenter


def _pred_frames_to_units(
    pred_frames, vlen, frame_shift, sr
) -> List[SegmentationUnit]:
    """Convert phonvec frame indices to SegmentationUnit time spans."""
    times = sorted({int(p) for p in pred_frames if 0 <= int(p) < vlen})
    if len(times) < 2:
        return []
    return [
        SegmentationUnit(
            start=times[i] * frame_shift / sr,
            end=times[i + 1] * frame_shift / sr,
        )
        for i in range(len(times) - 1)
    ]


class PhonvecInference:
    """Per-utterance inference: encoder forward + Segmenter.segment."""

    def __init__(
        self,
        artifact: dict,
        net: nn.Module,
        device: str = "cpu",
        *,
        frame_shift: Optional[int] = None,
        sr: Optional[int] = None,
    ):
        """Args:
        artifact: dict produced by ``Segmenter.to_artifact``; must
            contain a ``net`` spec with ``frame_shift`` and ``sr``.
        net: encoder module exposing ``encode(speech, lengths)``.
        device: torch device for the encoder forward.
        frame_shift: optional override of the artifact's saved hop.
        sr: optional override of the artifact's saved sample rate.
        """
        self.net = net.to(device).eval()
        self.device = device
        self.segmenter = Segmenter.from_artifact(artifact)
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
        net_feats = feats[0, :vlen].cpu().numpy()
        waveform_np = sp.cpu().numpy()

        pred_frames = self.segmenter.segment(net_feats, waveform_np)
        return _pred_frames_to_units(
            pred_frames, vlen, self.frame_shift, self.sr
        )


def build_phonvec_inference(
    artifact_path: str,
    net: nn.Module,
    device: str = "cuda",
    *,
    frame_shift: Optional[int] = None,
    sr: Optional[int] = None,
) -> PhonvecInference:
    """Hydra entry point: load artifact + bind to an encoder net.

    Passes ``frame_shift`` / ``sr`` to override the artifact's saved values
    """
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    return PhonvecInference(
        artifact, net, device=device, frame_shift=frame_shift, sr=sr
    )
