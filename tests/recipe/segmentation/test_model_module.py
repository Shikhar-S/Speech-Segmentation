"""Integration tests for SegmentationModel loss-mode dispatch."""

import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace

from src.recipe.segmentation.model_module import SegmentationModel


# -------------------------------------------------------------------
# DUMMY NET
# -------------------------------------------------------------------

class DummyNet(nn.Module):
    """Minimal net satisfying the SegmentationModel interface."""

    ENCODER_DIM = 16
    VOCAB = 8
    POINTS = 320  # 20 ms frames at 16 kHz

    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self._enc = nn.Linear(self.ENCODER_DIM, self.ENCODER_DIM)
        self.ctc = SimpleNamespace(ctc_lo=nn.Linear(self.ENCODER_DIM, self.VOCAB))

    def encode(self, speech, lengths):
        B = speech.shape[0]
        S = max(1, int(lengths.max()) // self.POINTS)
        features = self._enc(torch.randn(B, S, self.ENCODER_DIM))
        logit_lens = (lengths // self.POINTS).clamp(min=1).long()
        return features, logit_lens

    def ctc_logits(self, speech, lengths):
        features, logit_lens = self.encode(speech, lengths)
        return self.ctc.ctc_lo(features), logit_lens

    def encoder_output_size(self):
        return self.ENCODER_DIM

    def points_by_frames(self):
        return self.POINTS

    def get_blank_id(self):
        return 0

    def forced_align(self, speech, speech_lengths, text, text_lengths):
        S = max(1, int(speech_lengths.max()) // self.POINTS)
        return torch.zeros(1, S, dtype=torch.long), None


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _make_batch(B=2, T_speech=6400, T_phones=4):
    """Synthetic batch matching the collate_fn output schema."""
    points = DummyNet.POINTS
    speech = torch.randn(B, T_speech)
    lengths = torch.full((B,), T_speech, dtype=torch.long)
    target = torch.randint(0, DummyNet.VOCAB, (B, T_phones))
    target_len = torch.full((B,), T_phones, dtype=torch.long)
    target_start = torch.stack(
        [torch.arange(T_phones, dtype=torch.float32) * points] * B
    )
    target_end = target_start + points - 1
    return {
        "speech": speech,
        "speech_length": lengths,
        "target": target,
        "target_length": target_len,
        "target_start": target_start,
        "target_end": target_end,
    }


def _make_model(bce_weight=0.0, pos_weight=1.0, resolution=1, audio_sr=16000):
    return SegmentationModel(
        net=DummyNet(),
        optimizer=torch.optim.Adam,
        scheduler=None,
        bce_weight=bce_weight,
        pos_weight=pos_weight,
        resolution=resolution,
        audio_sr=audio_sr,
    )


# -------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------

def test_fa_only_no_boundary_head():
    """bce_weight=0.0: boundary_head must not exist; bce_loss is None."""
    model = _make_model(bce_weight=0.0)
    assert not hasattr(model, "boundary_head")
    out = model.model_step(_make_batch())
    assert out["bce_loss"] is None
    assert out["fa_loss"] is not None
    assert torch.isfinite(out["loss"])
    assert "rval" in out
    assert out["rval"] is not None


def test_bce_only_no_fa_computed():
    """bce_weight=1.0: fa_loss and logits must be None."""
    model = _make_model(bce_weight=1.0)
    assert hasattr(model, "boundary_head")
    out = model.model_step(_make_batch())
    assert out["fa_loss"] is None
    assert out["logits"] is None
    assert out["bce_loss"] is not None
    assert torch.isfinite(out["loss"])


def test_combined_loss_weighted_sum():
    """bce_weight=0.5: loss must equal 0.5*fa_loss + 0.5*bce_loss."""
    model = _make_model(bce_weight=0.5)
    out = model.model_step(_make_batch())
    expected = 0.5 * out["fa_loss"] + 0.5 * out["bce_loss"]
    assert torch.isclose(out["loss"], expected)


def test_hparams_saved():
    """bce_weight and pos_weight must appear in Lightning hparams."""
    model = _make_model(bce_weight=0.3, pos_weight=2.0)
    assert model.hparams["bce_weight"] == pytest.approx(0.3)
    assert model.hparams["pos_weight"] == pytest.approx(2.0)


def test_fa_backward():
    """Backward pass in FA-only mode must produce finite gradients."""
    model = _make_model(bce_weight=0.0)
    out = model.model_step(_make_batch())
    out["loss"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.isfinite().all()


def test_bce_backward():
    """Backward pass in BCE-only mode must produce finite gradients."""
    model = _make_model(bce_weight=1.0)
    out = model.model_step(_make_batch())
    out["loss"].backward()
    assert model.boundary_head.weight.grad is not None
    assert model.boundary_head.weight.grad.isfinite().all()


def test_combined_backward():
    """Backward pass in combined mode must produce finite gradients."""
    model = _make_model(bce_weight=0.4)
    out = model.model_step(_make_batch())
    out["loss"].backward()
    assert model.boundary_head.weight.grad is not None
    assert model.boundary_head.weight.grad.isfinite().all()


def test_test_step_fa_only():
    """bce_weight=0: test_step populates fa and greedy, not boundary."""
    model = _make_model(bce_weight=0.0)
    model.on_test_start()
    model.test_step(_make_batch(), batch_idx=0)
    assert len(model.test_data["fa"]["predictions"]) > 0
    assert len(model.test_data["greedy"]["predictions"]) > 0
    assert len(model.test_data["boundary"]["predictions"]) == 0


def test_test_step_bce_only():
    """bce_weight=1: test_step populates boundary, not fa or greedy."""
    model = _make_model(bce_weight=1.0)
    model.on_test_start()
    model.test_step(_make_batch(), batch_idx=0)
    assert len(model.test_data["fa"]["predictions"]) == 0
    assert len(model.test_data["greedy"]["predictions"]) == 0
    assert len(model.test_data["boundary"]["predictions"]) > 0


def test_test_step_combined():
    """bce_weight=0.5: test_step populates all three modes."""
    model = _make_model(bce_weight=0.5)
    model.on_test_start()
    model.test_step(_make_batch(), batch_idx=0)
    assert len(model.test_data["fa"]["predictions"]) > 0
    assert len(model.test_data["greedy"]["predictions"]) > 0
    assert len(model.test_data["boundary"]["predictions"]) > 0


def test_upsample_resolution_2():
    """resolution=2: features double in time, effective_pbf halves."""
    model = _make_model(bce_weight=1.0, resolution=2)
    assert model.effective_pbf == DummyNet.POINTS / 2
    assert hasattr(model, "upsample")
    out = model.model_step(_make_batch())
    assert torch.isfinite(out["loss"])


def test_upsample_backward():
    """resolution=2: backward pass produces finite gradients on upsample."""
    model = _make_model(bce_weight=1.0, resolution=2)
    out = model.model_step(_make_batch())
    out["loss"].backward()
    assert model.upsample.weight.grad is not None
    assert model.upsample.weight.grad.isfinite().all()
    assert model.boundary_head.weight.grad is not None
