"""Parity tests: SegmentRecognizeModel vs JointPRSegModel.

Verifies that the modular implementation produces identical loss
values and gradients as the original coupled implementation when
configured equivalently (bce_weight=1.0, seg_loss_weight=1.0,
pr_loss_weight=1.0).
"""

import copy

import torch
import torch.nn as nn
from types import SimpleNamespace

from src.recipe.joint.model_module import JointPRSegModel
from src.recipe.segment_recognize.model_module import (
    SegmentRecognizeModel,
)


# -- Shared dummy net -------------------------------------------

ENCODER_DIM = 16
VOCAB = 8
POINTS = 320


class DummyNet(nn.Module):
    """Deterministic encoder for reproducible comparisons."""

    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.ignore_id = -1
        self._enc = nn.Linear(ENCODER_DIM, ENCODER_DIM)
        self.ctc = SimpleNamespace(
            ctc_lo=nn.Linear(ENCODER_DIM, VOCAB),
        )

    def encode(self, speech, lengths):
        B = speech.shape[0]
        S = max(1, int(lengths.max()) // POINTS)
        features = self._enc(
            torch.randn(B, S, ENCODER_DIM),
        )
        logit_lens = (lengths // POINTS).clamp(min=1).long()
        return features, logit_lens

    def encoder_output_size(self):
        return ENCODER_DIM

    def points_by_frames(self):
        return POINTS

    def get_blank_id(self):
        return 0

    def _calc_ctc_loss(
        self, enc, enc_lens, text, text_lens, **kw,
    ):
        logits = self.ctc.ctc_lo(enc)
        log_p = nn.functional.log_softmax(logits, dim=-1)
        log_p = log_p.permute(1, 0, 2)
        text_clean = torch.where(text == -1, 0, text)
        loss = nn.functional.ctc_loss(
            log_p, text_clean, enc_lens, text_lens,
            blank=0, reduction="mean",
        )
        return loss, {"loss_ctc": loss.detach()}


# -- Helpers ----------------------------------------------------

BCE_TARGET = (
    "src.recipe.segment_recognize"
    ".heads.bce_boundary.BCEBoundaryHead"
)
CTC_TARGET = (
    "src.recipe.segment_recognize"
    ".heads.ctc_recognition.CTCRecognitionHead"
)


def _make_pair():
    """Create both models sharing the same net and weights.

    Returns ``(joint_model, sr_model)`` with identical
    boundary_head weights.
    """
    net = DummyNet()

    joint = JointPRSegModel(
        net=net,
        optimizer=torch.optim.Adam,
        bce_weight=1.0,
        seg_loss_weight=1.0,
        pr_loss_weight=1.0,
    )

    sr = SegmentRecognizeModel(
        net=net,
        optimizer=torch.optim.Adam,
        seg_losses={
            "bce": {"_target_": BCE_TARGET, "weight": 1.0},
        },
        pr_losses={
            "ctc": {"_target_": CTC_TARGET, "weight": 1.0},
        },
    )

    # Copy boundary_head weights so both models are identical.
    sr.seg_losses["bce"].boundary_head.load_state_dict(
        joint.boundary_head.state_dict(),
    )
    # SR uses nn.Identity() at resolution=1; make joint's Linear
    # act as identity so both produce the same features.
    with torch.no_grad():
        joint.upsample.weight.copy_(torch.eye(ENCODER_DIM))
        joint.upsample.bias.zero_()

    return joint, sr


def _seg_sub_batch(B=2, T_speech=6400, T_phones=4):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full(
            (B,), T_speech, dtype=torch.long,
        ),
        "text": torch.randint(1, VOCAB, (B, T_phones)),
        "text_length": torch.full(
            (B,), T_phones, dtype=torch.long,
        ),
        "target_start": torch.stack(
            [
                torch.arange(T_phones, dtype=torch.float32)
                * POINTS
            ]
            * B,
        ),
        "target_end": torch.stack(
            [
                torch.arange(T_phones, dtype=torch.float32)
                * POINTS
                + POINTS
                - 1
            ]
            * B,
        ),
        "utt_id": [f"seg_{i}" for i in range(B)],
    }


def _pr_sub_batch(B=2, T_speech=4800, T_phones=3):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full(
            (B,), T_speech, dtype=torch.long,
        ),
        "text": torch.randint(1, VOCAB, (B, T_phones)),
        "text_length": torch.full(
            (B,), T_phones, dtype=torch.long,
        ),
        "lang_sym": ["<eng>"] * B,
        "utt_id": [f"pr_{i}" for i in range(B)],
    }


# -- Parity tests -----------------------------------------------


def test_seg_loss_parity():
    """Seg-only batch produces matching loss values."""
    joint, sr = _make_pair()
    torch.manual_seed(0)
    seg = _seg_sub_batch()

    batch_j = {"segmentation": seg, "recognition": None}
    batch_s = {"segmentation": copy.deepcopy(seg), "recognition": None}

    # Seed the encoder identically for both.
    torch.manual_seed(42)
    loss_j = joint.training_step(batch_j, 0)
    torch.manual_seed(42)
    loss_s = sr.training_step(batch_s, 0)

    torch.testing.assert_close(
        loss_j, loss_s, atol=1e-5, rtol=1e-5,
    )


def test_pr_loss_parity():
    """PR-only batch produces matching loss values."""
    joint, sr = _make_pair()
    torch.manual_seed(0)
    pr = _pr_sub_batch()

    batch_j = {"segmentation": None, "recognition": pr}
    batch_s = {
        "segmentation": None,
        "recognition": copy.deepcopy(pr),
    }

    torch.manual_seed(42)
    loss_j = joint.training_step(batch_j, 0)
    torch.manual_seed(42)
    loss_s = sr.training_step(batch_s, 0)

    torch.testing.assert_close(
        loss_j, loss_s, atol=1e-5, rtol=1e-5,
    )


def test_mixed_loss_parity():
    """Mixed batch produces matching total loss."""
    joint, sr = _make_pair()
    torch.manual_seed(0)
    seg = _seg_sub_batch()
    pr = _pr_sub_batch()

    batch_j = {"segmentation": seg, "recognition": pr}
    batch_s = {
        "segmentation": copy.deepcopy(seg),
        "recognition": copy.deepcopy(pr),
    }

    torch.manual_seed(42)
    loss_j = joint.training_step(batch_j, 0)
    torch.manual_seed(42)
    loss_s = sr.training_step(batch_s, 0)

    torch.testing.assert_close(
        loss_j, loss_s, atol=1e-5, rtol=1e-5,
    )


def test_gradient_parity():
    """Encoder gradients match direction after backward."""
    joint, sr = _make_pair()
    torch.manual_seed(0)
    seg = _seg_sub_batch()
    pr = _pr_sub_batch()

    batch_j = {"segmentation": seg, "recognition": pr}
    batch_s = {
        "segmentation": copy.deepcopy(seg),
        "recognition": copy.deepcopy(pr),
    }

    torch.manual_seed(42)
    loss_j = joint.training_step(batch_j, 0)
    loss_j.backward()
    grad_j = joint.net._enc.weight.grad.clone()

    joint.net._enc.weight.grad = None
    torch.manual_seed(42)
    loss_s = sr.training_step(batch_s, 0)
    loss_s.backward()
    grad_s = sr.net._enc.weight.grad.clone()

    # Same direction (cosine similarity > 0.99).
    cos = nn.functional.cosine_similarity(
        grad_j.flatten().unsqueeze(0),
        grad_s.flatten().unsqueeze(0),
    )
    assert cos.item() > 0.99
