"""Segmentation inference module.

Usage:
    python -m src.recipe.segmentation.inference
"""
from typing import List

import torch
import torch.nn as nn
from src.metrics.segmentation_evaluator import SegmentationUnit
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SegmentationInference:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        greedy: bool = True,
    ) -> None:
        self.net = model
        self.device = device
        self.greedy = greedy
        self.net.to(self.device)

    @staticmethod
    def post_process_alignments(net, labels) -> List[SegmentationUnit]:
        """Post-process frame-level labels into forced aligned units.
            Removes blanks.
        Args:
            net: The model used for alignment.
            labels: List of labels where labels[i] corresponds to the label for frame i.
        Returns:
            List[ForceAlignedUnit]: List of forced aligned units.
        """
        points_by_frames = net.points_by_frames()
        alignment_result: List[SegmentationUnit] = []
        if not labels:
            return alignment_result

        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                if labels[i - 1] != net.get_blank_id():
                    # only add non-blank labels
                    alignment_result.append(
                        SegmentationUnit(
                            start=start_idx * points_by_frames / net.sampling_rate,
                            end=i * points_by_frames / net.sampling_rate,
                            label=labels[i - 1],
                        )
                    )
                start_idx = i  # do irrespective

        if labels[-1] != net.get_blank_id():
            alignment_result.append(
                SegmentationUnit(
                    start=start_idx * points_by_frames / net.sampling_rate,
                    end=len(labels) * points_by_frames / net.sampling_rate,
                    label=labels[-1],
                )
            )
        return alignment_result

    @staticmethod
    @torch.no_grad()
    def greedy_decode(net, speech, speech_length, device="cpu") -> List[SegmentationUnit]:
        """Greedy CTC decode a single utterance without text input.

        Runs argmax over CTC logits frame-by-frame, then collapses repeated
        tokens and blanks while mapping frames back to time.

        Args:
            net: The model (must implement ctc_logits, points_by_frames, sampling_rate, get_blank_id).
            speech: (Length,) waveform tensor.
            speech_length: int or scalar tensor, number of valid samples.
            device: device string.
        Returns:
            List[SegmentationUnit]: collapsed phone segments with timestamps.
        """
        sp = speech[:int(speech_length)].unsqueeze(0).to(device)
        splen_t = torch.as_tensor([int(speech_length)], device=device)
        logits, logit_lengths = net.ctc_logits(sp, splen_t)
        frame_labels = logits.argmax(dim=-1).squeeze(0)[:int(logit_lengths[0])].tolist()
        return SegmentationInference.post_process_alignments(net, frame_labels)

    @staticmethod
    def prepare_inputs(
        speech: torch.Tensor,
        speech_length: torch.Tensor,
        target: torch.Tensor,
        target_length: torch.Tensor,
        device: str = "cpu",
        utt_id=None,
    ) -> dict:
        """Prepare inputs for forced alignment inference.

        Args:
            speech: (Length)
            speech_length: int
            target: (Length) tokenized
            target_length: int
            device: device to move tensors to
            utt_id: identifier for the utterance
        Returns:
            dict: Dictionary containing prepared inputs.
        """
        sp = speech[: int(speech_length)].unsqueeze(0).to(device)  # (1, T_s)
        txt = target[: int(target_length)].unsqueeze(0).to(device)  # (1, T_t)
        splen_t = torch.as_tensor([int(speech_length)], device=sp.device)
        txtlen_t = torch.as_tensor([int(target_length)], device=sp.device)
        log.info(
            f"Prepared inputs: uttid:{utt_id}, speech length: {splen_t.item()}, target length: {txtlen_t.item()}"
        )
        return sp, txt, splen_t, txtlen_t

    @torch.no_grad()
    def __call__(
        self,
        speech,
        speech_length,
        target=None,
        target_length=None,
        utt_id=None,
        *args,
        **kwargs,
    ) -> List[SegmentationUnit]:
        """Get segmentation for a single utterance.

        In greedy mode, decodes without text input via CTC argmax.
        In forced-align mode, requires target and target_length.

        Args:
            speech: (Length)
            speech_length: int
            target: (Length) tokenized — required if greedy=False
            target_length: int — required if greedy=False
            utt_id: str, identifier for the utterance
        Returns:
            List[SegmentationUnit]: list of segmentation units for the utterance.
        """
        if self.greedy:
            return SegmentationInference.greedy_decode(self.net, speech, speech_length, device=self.device)
        sp, txt, splen_t, txtlen_t = SegmentationInference.prepare_inputs(
            speech,
            speech_length,
            target,
            target_length,
            device=self.device,
            utt_id=utt_id,
        )
        align_label, _ = self.net.forced_align(
            sp, splen_t, txt, txtlen_t, utt_id=utt_id
        )
        labels = align_label.squeeze(0).detach().cpu().tolist()
        return SegmentationInference.post_process_alignments(self.net, labels)


def _boundary_flags_to_units(net, is_boundary, valid_len) -> List[SegmentationUnit]:
    """Convert per-frame boundary flags to SegmentationUnit segments.

    Args:
        net: The encoder model (provides points_by_frames and sampling_rate).
        is_boundary: List[bool] of length valid_len; True marks a phone start.
        valid_len: Number of valid (non-padded) frames.
    Returns:
        List[SegmentationUnit] with label=0 (no phone-class info in BCE mode).
    """
    points = net.points_by_frames()
    sr = net.sampling_rate
    units: List[SegmentationUnit] = []
    start = 0
    for i in range(1, valid_len):
        if is_boundary[i]:
            units.append(SegmentationUnit(
                start=start * points / sr,
                end=i * points / sr,
                label=0,
            ))
            start = i
    units.append(SegmentationUnit(
        start=start * points / sr,
        end=valid_len * points / sr,
        label=0,
    ))
    return units


class BoundaryInference:
    """Inference for the BCE boundary detection model.

    Applies sigmoid to per-frame boundary_head logits, thresholds to find
    phone boundaries, and returns SegmentationUnit objects with timestamps.

    Args:
        model: Encoder network (must implement encode, points_by_frames, sampling_rate).
        boundary_head: Linear(encoder_dim, 1) trained with BoundaryLoss.
        device: Torch device string.
        threshold: Sigmoid threshold for declaring a boundary (default 0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        boundary_head: nn.Module,
        device: str = "cpu",
        threshold: float = 0.5,
    ) -> None:
        self.net = model
        self.boundary_head = boundary_head
        self.threshold = threshold
        self.device = device
        self.net.to(self.device)
        self.boundary_head.to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        speech,
        speech_length,
        **kwargs,
    ) -> List[SegmentationUnit]:
        """Segment a single utterance using boundary detection.

        Args:
            speech: (Length,) waveform tensor.
            speech_length: int or scalar tensor, number of valid samples.
        Returns:
            List[SegmentationUnit]: phone segments with timestamps, label=0.
        """
        sp = speech[:int(speech_length)].unsqueeze(0).to(self.device)
        splen_t = torch.as_tensor([int(speech_length)], device=self.device)
        features, logit_lens = self.net.encode(sp, splen_t)
        if isinstance(features, tuple):
            features = features[0]
        boundary_probs = torch.sigmoid(
            self.boundary_head(features).squeeze(-1)
        ).squeeze(0)
        valid_len = int(logit_lens[0])
        is_boundary = (boundary_probs[:valid_len] > self.threshold).tolist()
        return _boundary_flags_to_units(self.net, is_boundary, valid_len)


def build_boundary_inference(
    net: nn.Module,
    ckpt_path: str,
    device: str = "cuda",
    threshold: float = 0.5,
) -> BoundaryInference:
    """Build BoundaryInference with net and boundary_head loaded from a Lightning checkpoint.

    Args:
        net: Encoder network instance (uninitialised weights).
        ckpt_path: Path to Lightning .ckpt file saved by SegmentationModel.
        device: Torch device string.
        threshold: Sigmoid threshold for boundary detection.
    Returns:
        BoundaryInference ready for inference.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict(
        {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")},
        strict=True,
    )
    boundary_head = nn.Linear(net.encoder_output_size(), 1)
    boundary_head.load_state_dict(
        {k[len("boundary_head."):]: v for k, v in state.items() if k.startswith("boundary_head.")},
        strict=True,
    )
    return BoundaryInference(
        model=net, boundary_head=boundary_head, device=device, threshold=threshold
    )


def build_segmentation_inference(
    net: nn.Module,
    ckpt_path: str,
    device: str = "cuda",
    greedy: bool = True,
) -> SegmentationInference:
    """Build SegmentationInference with ALL weights loaded from a Lightning checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    net_state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    net.load_state_dict(net_state, strict=True)
    return SegmentationInference(model=net, device=device, greedy=greedy)


if __name__ == "__main__":
    # python -m src.recipe.segmentation.inference
    from src.model.xeusphoneme.builders import build_xeus_pr_from_hf

    ckpt_path = '/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/runs/speech_segmentation/seg_pxeus_frac0_053/last.ckpt'
    net = build_xeus_pr_from_hf(
        work_dir='/work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/cache/xeus',
        hf_repo='espnet/xeus',
        load_ckpt=False,
        vocab_file='src/model/xeusphoneme/resources/ipa_vocab.json',
        interctc_weight=0.3,
        interctc_layer_idx=[4, 8, 12],
        interctc_use_conditioning=True,
        ctc_weight=1.0,
    )
    inference_module = build_segmentation_inference(net, ckpt_path, device='cuda', greedy=True)
    log.info("SegmentationInference module built successfully.")
    