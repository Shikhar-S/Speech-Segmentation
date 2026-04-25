"""Integration tests for SegmentRecognizeModel."""

import pytest
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

from functools import partial

from src.recipe.segment_recognize.heads.bce_boundary import (
    BCEBoundaryHead,
)
from src.recipe.segment_recognize.heads.ctc_recognition import (
    CTCRecognitionHead,
)
from src.recipe.segment_recognize.heads.frame_ce_segmentation import (
    FrameCESegmentationHead,
)


def _make_model(**overrides):
    defaults = dict(
        net=DummyNet(),
        optimizer=torch.optim.Adam,
        seg_losses={
            "bce": partial(BCEBoundaryHead, weight=1.0),
        },
        pr_losses={
            "ctc": partial(CTCRecognitionHead, weight=1.0),
        },
    )
    defaults.update(overrides)
    return SegmentRecognizeModel(**defaults)


def _seg_sub_batch(B=2, T_speech=6400, T_phones=4):
    arange = torch.arange(T_phones, dtype=torch.float32) * POINTS
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full((B,), T_speech, dtype=torch.long),
        "phones": torch.randint(1, VOCAB, (B, T_phones)),
        "phone_length": torch.full((B,), T_phones, dtype=torch.long),
        "phone_start": torch.stack([arange] * B),
        "phone_end": torch.stack([arange + POINTS - 1] * B),
        "utt_id": [f"seg_{i}" for i in range(B)],
    }


def _pr_sub_batch(B=2, T_speech=4800, T_phones=3):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full((B,), T_speech, dtype=torch.long),
        "phones": torch.randint(1, VOCAB, (B, T_phones)),
        "phone_length": torch.full((B,), T_phones, dtype=torch.long),
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
            "ctc": partial(CTCRecognitionHead, weight=0.1),
        },
    )
    assert model.pr_losses["ctc"].weight == 0.1


def test_multiple_seg_losses():
    """Model supports multiple seg losses simultaneously."""
    model = _make_model(
        seg_losses={
            "bce": partial(BCEBoundaryHead, weight=0.7),
            "fa": partial(FrameCESegmentationHead, weight=0.3),
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


# -- Refactor coverage ------------------------------------------


def test_head_name_collision_raises_at_init():
    """Head names colliding across seg/pr groups raise in ``__init__``."""
    with pytest.raises(ValueError, match="collision"):
        _make_model(
            seg_losses={"dup": partial(BCEBoundaryHead, weight=1.0)},
            pr_losses={"dup": partial(CTCRecognitionHead, weight=1.0)},
        )


def test_prepare_seg_targets_frame_math():
    """``_prepare_seg_targets`` floor-divides samples by effective_pbf."""
    model = _make_model()
    batch = {
        "phone_start": torch.tensor([[0.0, 639.0, 640.0, 1280.0]]),
        "phone_end": torch.tensor([[639.0, 1279.0, 1919.0, 2559.0]]),
    }
    model._prepare_seg_targets(batch)
    # POINTS=320 from DummyNet; resolution=1 → effective_pbf=320.
    assert torch.equal(
        batch["phone_start_idx"],
        torch.tensor([[0, 1, 2, 4]], dtype=torch.long),
    )
    assert torch.equal(
        batch["phone_end_idx"],
        torch.tensor([[1, 3, 5, 7]], dtype=torch.long),
    )


def test_apply_losses_sums_weighted_head_losses():
    """``_apply_losses`` returns ``sum(weight * head.forward.loss)``."""
    model = _make_model(
        seg_losses={
            "bce": partial(BCEBoundaryHead, weight=2.5),
        },
        pr_losses={
            "ctc": partial(CTCRecognitionHead, weight=0.25),
        },
    )
    batch = _seg_sub_batch()
    feat = torch.randn(2, 20, ENCODER_DIM)
    lens = torch.full((2,), 20, dtype=torch.long)
    model._prepare_seg_targets(batch)
    total = model._apply_losses(
        model.seg_losses, feat, lens, batch, "train", on_step=True,
    )
    # Single head: total should equal weight * raw loss.
    bce = model.seg_losses["bce"]
    raw = bce(feat, lens, batch)["loss"]
    torch.testing.assert_close(total, 2.5 * raw)


def test_validation_step_runs_seg_when_phone_start_present():
    """Val batch with seg targets populates ``val_seg_loss``."""
    model = _make_model()
    batch = {"segmentation": _seg_sub_batch(), "recognition": None}
    model.validation_step(batch, 0)
    assert model.val_seg_loss.weight.item() > 0


def test_predict_step_dispatches_and_drops_none():
    """``predict_step`` calls ``decode`` on each head and skips Nones."""
    class _NoDecodeHead(nn.Module):
        weight = 1.0

        def forward(self, features, feature_lens, batch, **ctx):
            return {"loss": torch.tensor(0.0)}

        @torch.no_grad()
        def decode(self, features, feature_lens, batch, **ctx):
            return None  # Should be skipped in results.

    model = _make_model()
    # Swap in a head that decodes to None.
    model.seg_losses["bce"].decode = (
        lambda *a, **kw: ["sentinel"] * a[0].shape[0]
    )
    model.pr_losses["ctc"].decode = lambda *a, **kw: None
    out = model.predict_step(
        {
            "speech": torch.randn(2, 4800),
            "speech_length": torch.full((2,), 4800, dtype=torch.long),
        },
        0,
    )
    assert "bce" in out
    assert "ctc" not in out
