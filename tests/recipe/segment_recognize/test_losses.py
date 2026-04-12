"""Unit tests for individual TaskHead implementations."""

import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.recipe.segment_recognize.heads.base import TaskHead
from src.recipe.segment_recognize.heads.bce_boundary import (
    BCEBoundaryHead,
)
from src.recipe.segment_recognize.heads.count_ctc import (
    CountCTCHead,
    _build_targets,
    _parse_substitution,
)
from src.recipe.segment_recognize.heads.fa_segmentation import (
    FASegmentationHead,
)
from src.recipe.segment_recognize.heads.ctc_recognition import (
    CTCRecognitionHead,
)


# -- Fixtures ---------------------------------------------------

ENCODER_DIM = 16
VOCAB = 8
POINTS = 320
B, T, N_PHONES = 2, 20, 4


def _dummy_net():
    """Minimal net with ctc.ctc_lo and _calc_ctc_loss."""
    net = SimpleNamespace()
    net.ctc = SimpleNamespace(
        ctc_lo=nn.Linear(ENCODER_DIM, VOCAB),
    )

    def _calc_ctc_loss(enc, enc_lens, text, text_lens, **kw):
        logits = net.ctc.ctc_lo(enc)
        log_p = nn.functional.log_softmax(logits, dim=-1)
        log_p = log_p.permute(1, 0, 2)
        text_clean = torch.where(text == -1, 0, text)
        loss = nn.functional.ctc_loss(
            log_p, text_clean, enc_lens, text_lens,
            blank=0, reduction="mean",
        )
        return loss, {"loss_ctc": loss.detach()}

    net._calc_ctc_loss = _calc_ctc_loss
    return net


def _seg_batch():
    """Batch dict expected by seg heads."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full((B,), T * POINTS, dtype=torch.long),
        "text": torch.randint(1, VOCAB, (B, N_PHONES)),
        "text_length": torch.full(
            (B,), N_PHONES, dtype=torch.long,
        ),
        "target_start_idx": torch.stack(
            [torch.arange(N_PHONES)] * B,
        ),
        "target_end_idx": torch.stack(
            [torch.arange(N_PHONES) + 1] * B,
        ),
    }


def _pr_batch():
    """Batch dict expected by PR heads."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full((B,), T * POINTS, dtype=torch.long),
        "text": torch.randint(1, VOCAB, (B, 3)),
        "text_length": torch.full((B,), 3, dtype=torch.long),
        "lang_sym": ["<eng>"] * B,
    }


# -- BCEBoundaryHead ------------------------------------------


def test_bce_forward_returns_loss():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM)
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert "boundary_logits" in out
    assert out["boundary_logits"].shape == (B, T)


def test_bce_eval_metrics_keys():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM)
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert set(metrics.keys()) == {
        "precision", "recall", "f1", "rval",
    }


def test_bce_boundary_head_grad():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM)
    feat = torch.randn(B, T, ENCODER_DIM, requires_grad=True)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    out["loss"].backward()
    assert lm.boundary_head.weight.grad is not None
    assert lm.boundary_head.weight.grad.abs().sum() > 0


# -- FASegmentationHead ----------------------------------------


def test_fa_forward_returns_loss_and_accuracy():
    lm = FASegmentationHead()
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch(), net=net)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert "accuracy" in out


def test_fa_eval_metrics():
    lm = FASegmentationHead()
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert "fa_accuracy" in metrics


# -- CTCRecognitionHead ----------------------------------------


def test_ctc_forward_delegates_to_net():
    lm = CTCRecognitionHead()
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _pr_batch(), net=net)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert "loss_ctc" in out


def test_ctc_eval_metrics_extracts_scalars():
    lm = CTCRecognitionHead()
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _pr_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _pr_batch())
    assert "loss_ctc" in metrics
    assert isinstance(metrics["loss_ctc"], float)


# -- Base class ------------------------------------------------


def test_loss_module_weight():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, weight=0.5)
    assert lm.weight == 0.5


def test_log_output_calls_pl_log():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM)
    pl = MagicMock(spec=["log"])
    out = {"loss": torch.tensor(1.0)}
    metrics = {"precision": 0.8, "rval": 0.7}
    lm.log_output(pl, "train", out, metrics, on_step=True)
    logged_names = {c.args[0] for c in pl.log.call_args_list}
    assert "train/bce_loss" in logged_names
    assert "train/precision" in logged_names
    assert "train/rval" in logged_names


def test_kwargs_absorbed():
    """Extra kwargs (from inject dict) don't raise."""
    lm = BCEBoundaryHead(
        encoder_dim=16, effective_pbf=320.0,
        audio_sr=32000, unknown_param=42,
    )
    assert lm.audio_sr == 32000

    lm2 = CTCRecognitionHead(
        encoder_dim=16, effective_pbf=320.0,
    )
    assert lm2.weight == 1.0


# -- CountCTCHead ----------------------------------------------


def _count_ctc_seg_batch():
    """Seg-style batch with frame-aligned target_start_idx."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full(
            (B,), T * POINTS, dtype=torch.long,
        ),
        "text": torch.randint(1, VOCAB, (B, N_PHONES)),
        "text_length": torch.full(
            (B,), N_PHONES, dtype=torch.long,
        ),
        "target_start_idx": torch.stack(
            [torch.arange(N_PHONES) * 3] * B,
        ),
    }


def _count_ctc_pr_batch():
    """PR-style batch (no target_start_idx)."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full(
            (B,), T * POINTS, dtype=torch.long,
        ),
        "text": torch.randint(1, VOCAB, (B, N_PHONES)),
        "text_length": torch.full(
            (B,), N_PHONES, dtype=torch.long,
        ),
    }


def test_count_ctc_pair_forward():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, substitution="1 0")
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (B, T, 3)
    assert lm.num_classes == 3


def test_count_ctc_single_forward():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, substitution="1")
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (B, T, 2)
    assert lm.num_classes == 2


def test_count_ctc_custom_vocab():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM, substitution="a b c",
    )
    assert lm.vocab == {"a": 1, "b": 2, "c": 3}
    assert lm.per_phone_seq == [1, 2, 3]
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    assert out["logits"].shape == (B, T, 4)


def test_count_ctc_build_targets_pair():
    text_len = torch.tensor([3, 2], dtype=torch.long)
    targets, lengths = _build_targets(text_len, [1, 2])
    assert targets.tolist() == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    assert lengths.tolist() == [6, 4]


def test_count_ctc_build_targets_single():
    text_len = torch.tensor([3, 1], dtype=torch.long)
    targets, lengths = _build_targets(text_len, [1])
    assert targets.tolist() == [1, 1, 1, 1]
    assert lengths.tolist() == [3, 1]


def test_count_ctc_eval_metrics_with_boundaries():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM,
        substitution="1 0",
        boundary_token="1",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    metrics = lm.eval_metrics(out, lens, _count_ctc_seg_batch())
    assert set(metrics.keys()) == {
        "precision", "recall", "f1", "rval",
        "count_accuracy", "count_mae",
    }


def test_count_ctc_eval_metrics_no_boundary_token():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM, substitution="1 0",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    metrics = lm.eval_metrics(out, lens, _count_ctc_seg_batch())
    assert metrics == {}


def test_count_ctc_eval_metrics_pr_batch():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM,
        substitution="1 0",
        boundary_token="1",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_pr_batch())
    metrics = lm.eval_metrics(out, lens, _count_ctc_pr_batch())
    assert set(metrics.keys()) == {"count_accuracy", "count_mae"}


def test_count_ctc_grad():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, substitution="1 0")
    feat = torch.randn(B, T, ENCODER_DIM, requires_grad=True)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    out["loss"].backward()
    assert lm.proj.weight.grad is not None
    assert lm.proj.weight.grad.abs().sum() > 0


def test_count_ctc_skip_long_targets():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, substitution="1 0")
    # T=20 frames, N_PHONES=4 -> L*N=8, fits.
    # Make one utterance need 30 tokens (15 phones * 2) > 20 frames.
    batch = _count_ctc_seg_batch()
    batch["text_length"] = torch.tensor([N_PHONES, 15], dtype=torch.long)
    feat = torch.randn(B, T, ENCODER_DIM, requires_grad=True)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_count_ctc_invalid_boundary_token():
    with pytest.raises(ValueError):
        CountCTCHead(
            encoder_dim=ENCODER_DIM,
            substitution="1 0",
            boundary_token="z",
        )


def test_count_ctc_empty_substitution():
    with pytest.raises(ValueError):
        CountCTCHead(encoder_dim=ENCODER_DIM, substitution="")
    with pytest.raises(ValueError):
        CountCTCHead(encoder_dim=ENCODER_DIM, substitution="   ")


def test_count_ctc_parse_substitution():
    vocab, seq = _parse_substitution("1 0")
    assert vocab == {"1": 1, "0": 2}
    assert seq == [1, 2]

    vocab, seq = _parse_substitution("a a b")
    assert vocab == {"a": 1, "b": 2}
    assert seq == [1, 1, 2]
