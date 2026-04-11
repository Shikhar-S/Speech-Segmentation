"""Integration tests for SegmentRecognizeModel."""

import torch
import torch.nn as nn
from types import SimpleNamespace

from src.recipe.segment_recognize.model_module import (
    SegmentRecognizeModel,
)


# -- Dummy net (same interface as joint tests) ------------------

ENCODER_DIM = 16
VOCAB = 8
POINTS = 320


class DummyNet(nn.Module):
    """Minimal encoder supporting both PR and seg interfaces."""

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


def _make_model(**overrides):
    defaults = dict(
        net=DummyNet(),
        optimizer=torch.optim.Adam,
        seg_losses={
            "bce": {"_target_": BCE_TARGET, "weight": 1.0},
        },
        pr_losses={
            "ctc": {"_target_": CTC_TARGET, "weight": 1.0},
        },
    )
    defaults.update(overrides)
    return SegmentRecognizeModel(**defaults)


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


# -- Tests ------------------------------------------------------


def test_training_step_pr_only():
    model = _make_model()
    batch = {
        "segmentation": None,
        "recognition": _pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_seg_only():
    model = _make_model()
    batch = {
        "segmentation": _seg_sub_batch(),
        "recognition": None,
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_mixed():
    model = _make_model()
    batch = {
        "segmentation": _seg_sub_batch(),
        "recognition": _pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_backward_mixed():
    """Gradients flow through shared encoder from both tasks."""
    model = _make_model()
    batch = {
        "segmentation": _seg_sub_batch(),
        "recognition": _pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    loss.backward()

    enc_grad = model.net._enc.weight.grad
    assert enc_grad is not None
    assert enc_grad.abs().sum() > 0

    bce = model.seg_losses["bce"]
    bh_grad = bce.boundary_head.weight.grad
    assert bh_grad is not None
    assert bh_grad.abs().sum() > 0


def test_on_validation_epoch_end_no_seg_updates():
    """No crash when val_seg_loss was never updated."""
    model = _make_model()
    model.on_validation_epoch_end()
    assert model.val_seg_loss.weight.item() == 0


def test_loss_weight_override():
    """Loss weights from config are respected."""
    model = _make_model(
        pr_losses={
            "ctc": {
                "_target_": CTC_TARGET,
                "weight": 0.1,
            },
        },
    )
    assert model.pr_losses["ctc"].weight == 0.1


def test_multiple_seg_losses():
    """Model supports multiple seg losses simultaneously."""
    fa_target = (
        "src.recipe.segment_recognize"
        ".heads.fa_segmentation.FASegmentationHead"
    )
    model = _make_model(
        seg_losses={
            "bce": {"_target_": BCE_TARGET, "weight": 0.7},
            "fa": {"_target_": fa_target, "weight": 0.3},
        },
    )
    assert "bce" in model.seg_losses
    assert "fa" in model.seg_losses

    batch = {
        "segmentation": _seg_sub_batch(),
        "recognition": None,
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
