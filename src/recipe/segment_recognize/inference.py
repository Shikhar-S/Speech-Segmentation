"""Inference wrapper for SegmentRecognizeModel.

Supports src/core/distributed_inference.py

Usage:
    python -m src.recipe.segment_recognize.inference
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.recipe.segment_recognize.model_module import (
    SegmentRecognizeModel,
)
from src.recipe.segment_recognize.weight_tying import BoundaryFromBinary


class SegmentRecognizeInference:
    def __init__(
        self,
        model: SegmentRecognizeModel,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        speech: torch.Tensor,
        speech_length: Any,
        target: torch.Tensor = None,
        target_length: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Decode a single utterance across all registered heads."""
        sp = speech[: int(speech_length)].unsqueeze(0).to(self.device)
        sl = torch.as_tensor([int(speech_length)], device=self.device)
        assert 'utt_id' in kwargs, "Make sure the predict dataset returns utt_id!"
        utt_id = kwargs['utt_id']
        batch: Dict[str, Any] = {"speech": sp, "speech_length": sl, 'utt_id': [utt_id]}
        
        if target is not None and target_length is not None:
            batch["target"] = (
                target[: int(target_length)].unsqueeze(0).to(self.device)
            )
            batch["target_length"] = torch.as_tensor(
                [int(target_length)], device=self.device,
            )

        return self.model.predict_step(batch, 0)


def build_segment_recognize_inference(
    net: nn.Module,
    ckpt_path: str,
    device: str = "cuda",
    tied_weights: bool = False,
    count_ctc_boundary_mode: str = None,
    boundary_threshold: float | None = None,
) -> SegmentRecognizeInference:
    """Build an inference wrapper from a Lightning checkpoint.

    Args:
        tied_weights: Set ``True`` for checkpoints trained with
            ``TieBCECountCTCProj``. Such checkpoints lack
            ``seg_losses.bce.boundary_head.*`` keys (the head was replaced
            with a parameter-less ``BoundaryFromBinary`` wrapper); we load
            with ``strict=False`` and re-apply the same tying so the BCE
            branch reads its log-odds from ``count_ctc.proj``.
    """
    model = SegmentRecognizeModel.load_from_checkpoint(
        ckpt_path,
        net=net,
        map_location="cpu",
        strict=not tied_weights,
        weights_only=False,
    )
    if tied_weights:
        if "bce" not in model.seg_losses:
            raise RuntimeError("tied_weights=True but 'bce' missing from seg_losses.")
        if "count_ctc" in model.pr_losses:
            cc = model.pr_losses["count_ctc"]
        elif "count_ctc" in model.seg_losses:
            cc = model.seg_losses["count_ctc"]
        else:
            raise RuntimeError("tied_weights=True but 'count_ctc' missing from heads.")
        model.seg_losses["bce"].boundary_head = BoundaryFromBinary(cc.proj)
    if count_ctc_boundary_mode is not None:
        cc = model.pr_losses["count_ctc"] if "count_ctc" in model.pr_losses else model.seg_losses["count_ctc"]
        cc.boundary_mode = count_ctc_boundary_mode
    if boundary_threshold is not None and "bce" in model.seg_losses:
        model.seg_losses["bce"].boundary_threshold = boundary_threshold
    return SegmentRecognizeInference(model=model, device=device)
