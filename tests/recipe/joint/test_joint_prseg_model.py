"""Integration tests for JointPRSegModel."""

import torch
import torch.nn as nn
from types import SimpleNamespace

from src.recipe.joint.model_module import JointPRSegModel


class DummyNet(nn.Module):
    ENCODER_DIM = 16
    VOCAB = 8
    POINTS = 320
    IGNORE_ID = -1

    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.ignore_id = self.IGNORE_ID
        self._enc = nn.Linear(self.ENCODER_DIM, self.ENCODER_DIM)
        self.ctc = SimpleNamespace(
            ctc_lo=nn.Linear(self.ENCODER_DIM, self.VOCAB),
        )

    def encode(self, speech, lengths):
        B = speech.shape[0]
        S = max(1, int(lengths.max()) // self.POINTS)
        features = self._enc(torch.randn(B, S, self.ENCODER_DIM))
        logit_lens = (lengths // self.POINTS).clamp(min=1).long()
        return features, logit_lens

    def encoder_output_size(self):
        return self.ENCODER_DIM

    def points_by_frames(self):
        return self.POINTS

    def get_blank_id(self):
        return 0

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, text, text_lengths, **kwargs):
        logits = self.ctc.ctc_lo(encoder_out)
        log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
        text_clean = torch.where(text == -1, 0, text)
        loss = nn.functional.ctc_loss(
            log_probs, text_clean, encoder_out_lens, text_lengths,
            blank=0, reduction="mean",
        )
        return loss, {"loss_ctc": loss.detach()}


POINTS = DummyNet.POINTS
VOCAB = DummyNet.VOCAB


def _make_model(**kwargs):
    defaults = dict(net=DummyNet(), optimizer=torch.optim.Adam, bce_weight=1.0)
    defaults.update(kwargs)
    return JointPRSegModel(**defaults)


def _make_seg_sub_batch(B=2, T_speech=6400, T_phones=4):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full((B,), T_speech, dtype=torch.long),
        "text": torch.randint(1, VOCAB, (B, T_phones)),
        "text_length": torch.full((B,), T_phones, dtype=torch.long),
        "target_start": torch.stack(
            [torch.arange(T_phones, dtype=torch.float32) * POINTS] * B
        ),
        "target_end": torch.stack(
            [torch.arange(T_phones, dtype=torch.float32) * POINTS + POINTS - 1] * B
        ),
        "utt_id": [f"seg_{i}" for i in range(B)],
    }


def _make_pr_sub_batch(B=2, T_speech=4800, T_phones=3):
    return {
        "speech": torch.randn(B, T_speech),
        "speech_length": torch.full((B,), T_speech, dtype=torch.long),
        "text": torch.randint(1, VOCAB, (B, T_phones)),
        "text_length": torch.full((B,), T_phones, dtype=torch.long),
        "lang_sym": ["<eng>"] * B,
        "utt_id": [f"pr_{i}" for i in range(B)],
    }


def test_training_step_pr_only():
    model = _make_model()
    batch = {"segmentation": None, "recognition": _make_pr_sub_batch()}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_seg_only():
    model = _make_model()
    batch = {"segmentation": _make_seg_sub_batch(), "recognition": None}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_mixed():
    model = _make_model()
    batch = {
        "segmentation": _make_seg_sub_batch(),
        "recognition": _make_pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_backward_mixed():
    """Gradients flow through shared encoder from both tasks."""
    model = _make_model()
    batch = {
        "segmentation": _make_seg_sub_batch(),
        "recognition": _make_pr_sub_batch(),
    }
    loss = model.training_step(batch, 0)
    loss.backward()
    enc_grad = model.net._enc.weight.grad
    assert enc_grad is not None
    assert enc_grad.abs().sum() > 0
    bh_grad = model.boundary_head.weight.grad
    assert bh_grad is not None
    assert bh_grad.abs().sum() > 0


def test_on_validation_epoch_end_no_seg_updates():
    """on_validation_epoch_end does not crash when val_seg_loss was
    never updated (no segmentation batches in the epoch)."""
    model = _make_model()
    model.on_validation_epoch_end()
    assert model.val_seg_loss.weight.item() == 0
