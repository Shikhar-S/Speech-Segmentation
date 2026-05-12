"""Unit tests for ``TieBCECountCTCProj`` callback.

Confirms that after ``setup`` the BCE head's projection is replaced by a
stateless wrapper around ``count_ctc.proj``, that no parameter is
double-counted, and that gradients from BOTH heads reach the shared
projection.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import SegmentationEvaluator
from src.recipe.segment_recognize.heads.bce_boundary import BCEBoundaryHead
from src.recipe.segment_recognize.heads.count_ctc import CountCTCHead
from src.recipe.segment_recognize.weight_tying import (
    BoundaryFromBinary,
    TieBCECountCTCProj,
)


ENCODER_DIM = 16
B, T, N_PHONES = 2, 8, 3


def _ev() -> SegmentationEvaluator:
    return SegmentationEvaluator(tolerance_ms=20)


def _make_module() -> nn.Module:
    """Mirror SegmentRecognizeModel's slot layout for the callback's lookup."""
    m = nn.Module()
    m.seg_losses = nn.ModuleDict(
        {"bce": BCEBoundaryHead(encoder_dim=ENCODER_DIM, evaluator=_ev())}
    )
    m.pr_losses = nn.ModuleDict(
        {"count_ctc": CountCTCHead(encoder_dim=ENCODER_DIM, evaluator=_ev())}
    )
    return m


def _seg_batch() -> dict:
    return {
        "speech": torch.randn(B, T * 320),
        "speech_length": torch.full((B,), T * 320, dtype=torch.long),
        "target_length": torch.full((B,), N_PHONES, dtype=torch.long),
        "target_start_idx": torch.stack(
            [torch.arange(N_PHONES, dtype=torch.long)] * B
        ),
        "target_end_idx": torch.stack(
            [torch.arange(N_PHONES, dtype=torch.long) + 1] * B
        ),
        "utt_id": [f"utt{i}" for i in range(B)],
    }


def test_tying_replaces_boundary_head():
    m = _make_module()
    TieBCECountCTCProj().setup(trainer=None, pl_module=m, stage="fit")
    bce = m.seg_losses["bce"]
    cc = m.pr_losses["count_ctc"]
    assert isinstance(bce.boundary_head, BoundaryFromBinary)
    assert bce.boundary_head._shared_proj is cc.proj


def test_tying_skips_when_stage_is_not_fit():
    m = _make_module()
    original = m.seg_losses["bce"].boundary_head
    TieBCECountCTCProj().setup(trainer=None, pl_module=m, stage="validate")
    assert m.seg_losses["bce"].boundary_head is original


def test_parameters_not_double_counted():
    m = _make_module()
    cc_proj = m.pr_losses["count_ctc"].proj
    TieBCECountCTCProj().setup(trainer=None, pl_module=m, stage="fit")

    param_ids = [id(p) for p in m.parameters()]
    assert param_ids.count(id(cc_proj.weight)) == 1
    assert param_ids.count(id(cc_proj.bias)) == 1


def test_gradient_via_bce_reaches_shared_proj():
    m = _make_module()
    TieBCECountCTCProj().setup(trainer=None, pl_module=m, stage="fit")
    bce = m.seg_losses["bce"]
    cc = m.pr_losses["count_ctc"]

    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    cc.proj.weight.grad = None
    out = bce(feat, lens, _seg_batch())
    out["loss"].backward()

    assert cc.proj.weight.grad is not None
    assert cc.proj.weight.grad.abs().sum() > 0


def test_gradient_via_count_ctc_reaches_shared_proj():
    m = _make_module()
    TieBCECountCTCProj().setup(trainer=None, pl_module=m, stage="fit")
    cc = m.pr_losses["count_ctc"]

    feat = torch.randn(B, T, ENCODER_DIM)
    lens = torch.full((B,), T, dtype=torch.long)
    cc.proj.weight.grad = None
    out = cc(feat, lens, _seg_batch())
    out["loss"].backward()

    assert cc.proj.weight.grad is not None
    assert cc.proj.weight.grad.abs().sum() > 0
