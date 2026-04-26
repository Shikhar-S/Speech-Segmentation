"""Unit tests for individual TaskHead implementations."""

import torch
import torch.nn as nn
import pytest
from types import MethodType, SimpleNamespace
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
from src.recipe.segment_recognize.heads.fce_segmentation import (
    FrameCESegmentationHead,
)
from src.recipe.segment_recognize.heads.asg_recognition import (
    ASGRecognitionHead,
)
from src.recipe.segment_recognize.heads.ctc_recognition import (
    CTCRecognitionHead,
)
from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)


# -- Fixtures ---------------------------------------------------

ENCODER_DIM = 16
VOCAB = 8
POINTS = 320
B, T, N_PHONES = 2, 20, 4


def _ev():
    """Shared evaluator (tolerance_ms=20) for heads constructed in tests."""
    return SegmentationEvaluator(tolerance_ms=20)


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
    """Batch dict expected by seg heads (canonical keys)."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full((B,), T * POINTS, dtype=torch.long),
        "phones": torch.randint(1, VOCAB, (B, N_PHONES)),
        "phone_length": torch.full(
            (B,), N_PHONES, dtype=torch.long,
        ),
        "phone_start_idx": torch.stack(
            [torch.arange(N_PHONES)] * B,
        ),
        "phone_end_idx": torch.stack(
            [torch.arange(N_PHONES) + 1] * B,
        ),
    }


def _pr_batch():
    """Batch dict expected by PR heads (canonical keys)."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full((B,), T * POINTS, dtype=torch.long),
        "phones": torch.randint(1, VOCAB, (B, 3)),
        "phone_length": torch.full((B,), 3, dtype=torch.long),
        "lang_sym": ["<eng>"] * B,
    }


# -- BCEBoundaryHead ------------------------------------------


def test_bce_forward_returns_loss():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev())
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert "boundary_logits" in out
    assert out["boundary_logits"].shape == (B, T)


def test_bce_eval_metrics_keys():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev())
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert set(metrics.keys()) == {
        "precision", "recall", "f1", "rval",
    }


def test_bce_boundary_head_grad():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev())
    feat = torch.randn(B, T, ENCODER_DIM, requires_grad=True)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    out["loss"].backward()
    assert lm.boundary_head.weight.grad is not None
    assert lm.boundary_head.weight.grad.abs().sum() > 0


# -- FASegmentationHead ----------------------------------------


def test_fa_forward_returns_loss_and_accuracy():
    lm = FrameCESegmentationHead(evaluator=_ev())
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch(), net=net)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert "accuracy" in out


def test_fa_eval_metrics():
    lm = FrameCESegmentationHead(evaluator=_ev())
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert "fa_accuracy" in metrics


# -- CTCRecognitionHead ----------------------------------------


def test_ctc_forward_delegates_to_net():
    lm = CTCRecognitionHead(evaluator=_ev())
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _pr_batch(), net=net)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert "loss_ctc" in out


def test_ctc_eval_metrics_extracts_scalars():
    lm = CTCRecognitionHead(evaluator=_ev())
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _pr_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _pr_batch())
    assert "loss_ctc" in metrics
    assert isinstance(metrics["loss_ctc"], float)


# -- Base class ------------------------------------------------


def test_loss_module_weight():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), weight=0.5)
    assert lm.weight == 0.5


def test_log_output_calls_pl_log():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev())
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
        encoder_dim=16, evaluator=_ev(), effective_pbf=320.0,
        audio_sr=32000, unknown_param=42, tolerance_ms=25,
    )
    assert lm.audio_sr == 32000

    lm2 = CTCRecognitionHead(
        encoder_dim=16, evaluator=_ev(), effective_pbf=320.0,
        tolerance_ms=25,
    )
    assert lm2.weight == 1.0


# -- CountCTCHead ----------------------------------------------


def _count_ctc_seg_batch():
    """Seg-style batch with frame-aligned phone_start_idx (canonical keys)."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full(
            (B,), T * POINTS, dtype=torch.long,
        ),
        "phones": torch.randint(1, VOCAB, (B, N_PHONES)),
        "phone_length": torch.full(
            (B,), N_PHONES, dtype=torch.long,
        ),
        "phone_start_idx": torch.stack(
            [torch.arange(N_PHONES) * 3] * B,
        ),
    }


def _count_ctc_pr_batch():
    """PR-style batch (no phone_start_idx)."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full(
            (B,), T * POINTS, dtype=torch.long,
        ),
        "phones": torch.randint(1, VOCAB, (B, N_PHONES)),
        "phone_length": torch.full(
            (B,), N_PHONES, dtype=torch.long,
        ),
    }


def test_count_ctc_pair_forward():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="1 0")
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (B, T, 3)
    assert lm.num_classes == 3


def test_count_ctc_single_forward():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="1")
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (B, T, 2)
    assert lm.num_classes == 2


def test_count_ctc_custom_vocab():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="a b c",
    )
    assert lm.vocab == {"a": 1, "b": 2, "c": 3}
    assert lm.per_phone_seq == [1, 2, 3]
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    assert out["logits"].shape == (B, T, 4)


def test_count_ctc_build_targets_pair():
    phone_len = torch.tensor([3, 2], dtype=torch.long)
    targets, lengths = _build_targets(phone_len, [1, 2])
    assert targets.tolist() == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    assert lengths.tolist() == [6, 4]


def test_count_ctc_build_targets_single():
    phone_len = torch.tensor([3, 1], dtype=torch.long)
    targets, lengths = _build_targets(phone_len, [1])
    assert targets.tolist() == [1, 1, 1, 1]
    assert lengths.tolist() == [3, 1]


def test_count_ctc_eval_metrics_with_boundaries():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
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
        encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="1 0",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    metrics = lm.eval_metrics(out, lens, _count_ctc_seg_batch())
    assert metrics == {}


def test_count_ctc_eval_metrics_pr_batch():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
        substitution="1 0",
        boundary_token="1",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_pr_batch())
    metrics = lm.eval_metrics(out, lens, _count_ctc_pr_batch())
    assert set(metrics.keys()) == {"count_accuracy", "count_mae"}


def test_count_ctc_grad():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="1 0")
    feat = torch.randn(B, T, ENCODER_DIM, requires_grad=True)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _count_ctc_seg_batch())
    out["loss"].backward()
    assert lm.proj.weight.grad is not None
    assert lm.proj.weight.grad.abs().sum() > 0


def test_count_ctc_skip_long_targets():
    lm = CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="1 0")
    # T=20 frames, N_PHONES=4 -> L*N=8, fits.
    # Make one utterance need 30 tokens (15 phones * 2) > 20 frames.
    batch = _count_ctc_seg_batch()
    batch["phone_length"] = torch.tensor([N_PHONES, 15], dtype=torch.long)
    feat = torch.randn(B, T, ENCODER_DIM, requires_grad=True)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_count_ctc_invalid_boundary_token():
    with pytest.raises(ValueError):
        CountCTCHead(
            encoder_dim=ENCODER_DIM,
            evaluator=_ev(),
            substitution="1 0",
            boundary_token="z",
        )


def test_count_ctc_empty_substitution():
    with pytest.raises(ValueError):
        CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="")
    with pytest.raises(ValueError):
        CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="   ")


def test_count_ctc_parse_substitution():
    vocab, seq = _parse_substitution("1 0")
    assert vocab == {"1": 1, "0": 2}
    assert seq == [1, 2]

    vocab, seq = _parse_substitution("a a b")
    assert vocab == {"a": 1, "b": 2}
    assert seq == [1, 1, 2]


# -- decode() --------------------------------------------------


def _dummy_decode_net():
    """Dummy net with ctc.ctc_lo, get_blank_id, token_list for decode tests."""
    net = _dummy_net()
    net.get_blank_id = lambda: 0
    net.token_list = [f"p{i}" for i in range(VOCAB)]
    return net


def _decode_batch():
    """Minimal batch dict for decode (no supervision needed)."""
    return {
        "speech": torch.randn(B, T * POINTS),
        "speech_length": torch.full(
            (B,), T * POINTS, dtype=torch.long,
        ),
    }


def test_task_head_base_decode_returns_none():
    class _Dummy(TaskHead):
        def forward(self, features, feature_lens, batch, **ctx):
            return {"loss": torch.tensor(0.0)}

    lm = _Dummy()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    assert lm.decode(feat, lens, _decode_batch()) is None


def test_bce_decode_returns_segmentation_units():
    lm = BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev())
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm.decode(feat, lens, _decode_batch())
    assert isinstance(out, list)
    assert len(out) == B
    for item in out:
        assert set(item.keys()) == {"boundaries"}
        for u in item["boundaries"]:
            assert isinstance(u, SegmentationUnit)
            assert isinstance(u.start, float)
            assert isinstance(u.end, float)


def test_count_ctc_decode_returns_segmentation_units():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
        substitution="1 0",
        boundary_token="1",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm.decode(feat, lens, _decode_batch())
    assert isinstance(out, list)
    assert len(out) == B
    for item in out:
        assert set(item.keys()) == {"boundaries"}
        for u in item["boundaries"]:
            assert isinstance(u, SegmentationUnit)


def test_count_ctc_decode_without_boundary_token_raises():
    lm = CountCTCHead(
        encoder_dim=ENCODER_DIM, evaluator=_ev(), substitution="1 0",
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    with pytest.raises(NotImplementedError):
        lm.decode(feat, lens, _decode_batch())


def test_ctc_decode_returns_phone_ids():
    lm = CTCRecognitionHead(evaluator=_ev())
    net = _dummy_decode_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm.decode(feat, lens, _decode_batch(), net=net)
    assert isinstance(out, list)
    assert len(out) == B
    for item in out:
        assert set(item.keys()) >= {
            "phone_ids", "phones", "transcript",
        }
        assert isinstance(item["phone_ids"], list)
        assert isinstance(item["phones"], list)
        assert isinstance(item["transcript"], str)


def test_fa_decode_returns_none():
    lm = FrameCESegmentationHead(evaluator=_ev())
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    assert lm.decode(feat, lens, _decode_batch()) is None


def test_asg_decode_returns_phone_ids():
    lm = ASGRecognitionHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
        num_labels=VOCAB,
        repeat_idx=0,
        use_transitions=False,
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm.decode(feat, lens, _decode_batch())
    assert isinstance(out, list)
    assert len(out) == B
    for item in out:
        assert "phone_ids" in item
        assert isinstance(item["phone_ids"], list)


# -- model-level predict_step dispatch ---------------------------


def _bare_sr_model(seg_heads=None, pr_heads=None):
    """Build a ``SegmentRecognizeModel`` stub bypassing ``__init__``.

    Dispatch tests only need ``seg_losses``, ``pr_losses``, and
    stubbed-out encode / normalize methods.
    """
    from src.recipe.segment_recognize.model_module import (
        SegmentRecognizeModel,
    )
    model = SegmentRecognizeModel.__new__(SegmentRecognizeModel)
    nn.Module.__init__(model)
    model.seg_losses = nn.ModuleDict(seg_heads or {})
    model.pr_losses = nn.ModuleDict(pr_heads or {})
    model.net = _dummy_decode_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    model._encode = MethodType(lambda self, sp, sl: (feat, lens), model)
    return model


def test_predict_step_dispatches_all_heads():
    model = _bare_sr_model(
        seg_heads={
            "bce": BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev()),
            "count_ctc": CountCTCHead(
                encoder_dim=ENCODER_DIM,
                evaluator=_ev(),
                substitution="1 0",
                boundary_token="1",
            ),
        },
    )
    out = model.predict_step(_decode_batch(), 0)
    assert set(out.keys()) == {"bce", "count_ctc"}
    for key in ("bce", "count_ctc"):
        for item in out[key]:
            assert "boundaries" in item
            for u in item["boundaries"]:
                assert isinstance(u, SegmentationUnit)


def test_predict_step_fa_skipped():
    """FA.decode returns None, so predict_step filters it out."""
    model = _bare_sr_model(
        seg_heads={"fa": FrameCESegmentationHead(evaluator=_ev())},
    )
    out = model.predict_step(_decode_batch(), 0)
    assert out == {}


# -- Rval eval metrics per head ---------------------------------


def test_fa_eval_metrics_returns_rval():
    lm = FrameCESegmentationHead(evaluator=_ev(), effective_pbf=POINTS, audio_sr=16000)
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert "fa_accuracy" in metrics
    assert set(metrics.keys()) >= {
        "fa_accuracy", "precision", "recall", "f1", "rval",
    }


def test_fa_eval_metrics_no_seg_targets():
    lm = FrameCESegmentationHead(evaluator=_ev(), effective_pbf=POINTS, audio_sr=16000)
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    batch = _pr_batch()
    batch["phone_start_idx"] = torch.zeros(B, 3, dtype=torch.long)
    batch["phone_end_idx"] = torch.ones(B, 3, dtype=torch.long)
    out = lm(feat, lens, batch, net=net)
    no_seg = _pr_batch()
    out_no_seg = {"loss": out["loss"], "accuracy": out["accuracy"],
                  "logits": out["logits"]}
    metrics = lm.eval_metrics(out_no_seg, lens, no_seg)
    assert "fa_accuracy" in metrics
    assert "rval" not in metrics


def test_ctc_eval_metrics_returns_rval():
    lm = CTCRecognitionHead(
        evaluator=_ev(), effective_pbf=POINTS, audio_sr=16000,
    )
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert "loss_ctc" in metrics
    assert set(metrics.keys()) >= {
        "loss_ctc", "precision", "recall", "f1", "rval",
    }


def test_ctc_eval_metrics_no_seg_targets():
    lm = CTCRecognitionHead(
        evaluator=_ev(), effective_pbf=POINTS, audio_sr=16000,
    )
    net = _dummy_net()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _pr_batch(), net=net)
    metrics = lm.eval_metrics(out, lens, _pr_batch())
    assert "loss_ctc" in metrics
    assert "rval" not in metrics


def test_asg_eval_metrics_returns_rval():
    lm = ASGRecognitionHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
        num_labels=VOCAB,
        repeat_idx=0,
        use_transitions=False,
        effective_pbf=POINTS,
        audio_sr=16000,
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _seg_batch())
    metrics = lm.eval_metrics(out, lens, _seg_batch())
    assert set(metrics.keys()) == {
        "precision", "recall", "f1", "rval",
    }


def test_asg_eval_metrics_no_seg_targets():
    lm = ASGRecognitionHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
        num_labels=VOCAB,
        repeat_idx=0,
        use_transitions=False,
        effective_pbf=POINTS,
        audio_sr=16000,
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm(feat, lens, _pr_batch())
    metrics = lm.eval_metrics(out, lens, _pr_batch())
    assert metrics == {}


# -- Decode boundary output -------------------------------------


def _dummy_net_with_tokens():
    """Dummy net with ``token_list`` + ``get_blank_id`` for CTC decode."""
    net = _dummy_net()
    net.token_list = [f"t{i}" for i in range(VOCAB)]
    net.get_blank_id = lambda: 0
    return net


def test_ctc_decode_returns_boundaries():
    lm = CTCRecognitionHead(evaluator=_ev(), effective_pbf=POINTS, audio_sr=16000)
    net = _dummy_net_with_tokens()
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm.decode(feat, lens, _pr_batch(), net=net)
    assert len(out) == B
    for item in out:
        assert set(item.keys()) == {
            "phone_ids", "phones", "transcript", "boundaries",
        }
        assert all(
            isinstance(u, SegmentationUnit) for u in item["boundaries"]
        )


def test_asg_decode_returns_boundaries():
    lm = ASGRecognitionHead(
        encoder_dim=ENCODER_DIM,
        evaluator=_ev(),
        num_labels=VOCAB,
        repeat_idx=0,
        use_transitions=False,
        effective_pbf=POINTS,
        audio_sr=16000,
    )
    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    out = lm.decode(feat, lens, _pr_batch())
    assert len(out) == B
    for item in out:
        assert set(item.keys()) == {"phone_ids", "boundaries"}
        assert all(
            isinstance(u, SegmentationUnit) for u in item["boundaries"]
        )
